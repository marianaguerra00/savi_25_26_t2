
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from model import ModelBetterCNN
import cv2
import random
import time
from tqdm import tqdm


# -------------------------------------------------
# Detection configuration
# -------------------------------------------------
class DetectionConfig:
    windowSize = 36
    stride = 4

    minForegroundRatio = 0.015
    foregroundThreshold = 30
    minMaxIntensityFactor = 2.0

    confidenceThreshold = 0.7
    confidenceMargin = 0.25

    borderMarginRatio = 0.05


cfg = DetectionConfig()


# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def computeBoxCenter(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def groupBoundingBoxes(boxes, labels, distanceThreshold):
    groups = []

    for box, label in zip(boxes, labels):
        cx, cy = computeBoxCenter(box)
        assigned = False

        for group in groups:
            gcx, gcy = group["center"]
            if np.hypot(cx - gcx, cy - gcy) < distanceThreshold:
                group["boxes"].append(box)
                group["labels"].append(label)

                xs = [b[0] + b[2] / 2 for b in group["boxes"]]
                ys = [b[1] + b[3] / 2 for b in group["boxes"]]
                group["center"] = (np.mean(xs), np.mean(ys))

                assigned = True
                break

        if not assigned:
            groups.append({
                "boxes": [box],
                "labels": [label],
                "center": (cx, cy)
            })

    return groups


def averageBoundingBox(boxes):
    return tuple(map(int, np.mean(boxes, axis=0)))


def majorityVote(labels):
    return max(set(labels), key=labels.count)


# -------------------------------------------------
# Dataset loading
# -------------------------------------------------
def loadImagesFromUbyte(filePath, numberOfImages, imageSize):
    data = np.fromfile(filePath, dtype=np.uint8)
    return data.reshape(numberOfImages, imageSize, imageSize)


def loadLabelsUbyte(filePath):
    annotationsPerImage = []

    with open(filePath, "rb") as file:
        while True:
            numObjectsBytes = file.read(1)
            if not numObjectsBytes:
                break

            numObjects = np.frombuffer(numObjectsBytes, dtype=np.uint8)[0]
            annotations = []

            for _ in range(numObjects):
                digitLabel = np.frombuffer(file.read(1), dtype=np.uint8)[0]
                x, y, w, h = np.frombuffer(file.read(8), dtype=np.uint16)
                annotations.append((digitLabel, int(x), int(y), int(w), int(h)))

            annotationsPerImage.append(annotations)

    return annotationsPerImage


# -------------------------------------------------
# Image processing
# -------------------------------------------------
def slidingWindow(image, windowSize, stride):
    h, w = image.shape
    for y in range(0, h - windowSize + 1, stride):
        for x in range(0, w - windowSize + 1, stride):
            yield x, y, image[y:y + windowSize, x:x + windowSize]


def mnistStyleResize(crop, threshold=40):
    ys, xs = np.where(crop > threshold)

    if len(xs) == 0:
        return cv2.resize(crop, (28, 28))

    digit = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = digit.shape
    size = max(h, w)

    padded = np.zeros((size, size), dtype=np.uint8)
    padded[(size - h)//2:(size - h)//2 + h,
           (size - w)//2:(size - w)//2 + w] = digit

    resized = cv2.resize(padded, (20, 20))
    final = np.zeros((28, 28), dtype=np.uint8)
    final[4:24, 4:24] = resized

    return final


def rejectWindow(crop):
    if np.max(crop) < np.mean(crop) + cfg.minMaxIntensityFactor * np.std(crop):
        return True

    fgRatio = np.count_nonzero(crop > cfg.foregroundThreshold) / crop.size
    if fgRatio < cfg.minForegroundRatio:
        return True

    ys, xs = np.where(crop > cfg.foregroundThreshold)
    if len(xs) == 0:
        return True

    margin = int(cfg.borderMarginRatio * crop.shape[0])
    if (
        xs.min() <= margin or xs.max() >= crop.shape[1] - margin or
        ys.min() <= margin or ys.max() >= crop.shape[0] - margin
    ):
        return True

    return False


# -------------------------------------------------
# Detection evaluation
# -------------------------------------------------
def computeIoU(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    unionArea = aw * ah + bw * bh - interArea

    return interArea / unionArea if unionArea > 0 else 0.0


def matchDetections(predBoxes, predLabels, gtAnnotations, iouThreshold=0.3):

    matchedGT = set()
    matchedPred = {}
    predToGT = {}

    tp = 0
    fp = 0

    # -------------------------------------------------
    # Loop over predictions (PREDICTION-DRIVEN)
    # -------------------------------------------------
    for pIdx, (pBox, pLabel) in enumerate(zip(predBoxes, predLabels)):

        bestIou = 0.0
        bestGT = -1
        bestGTLabel = None

        # Find best GT for this prediction
        for gtIdx, (gtLabel, gx, gy, gw, gh) in enumerate(gtAnnotations):
            if gtIdx in matchedGT:
                continue

            iou = computeIoU(pBox, (gx, gy, gw, gh))
            if iou > bestIou:
                bestIou = iou
                bestGT = gtIdx
                bestGTLabel = gtLabel

        # -------------------------------------------------
        # Decide TP / FP
        # -------------------------------------------------
        if bestGT != -1 and bestIou >= iouThreshold:
            matchedGT.add(bestGT)

            correctClass = (pLabel == bestGTLabel)

            matchedPred[pIdx] = {
                "gtIdx": bestGT,
                "correctClass": correctClass
            }

            predToGT[pIdx] = bestGTLabel

            if correctClass:
                tp += 1
            else:
                # Correct localization, wrong digit
                fp += 1
        else:
            # No GT matched with sufficient IoU
            fp += 1

    # -------------------------------------------------
    # False negatives = GT never matched
    # -------------------------------------------------
    fn = len(gtAnnotations) - len(matchedGT)
    

    return tp, fp, fn, matchedPred, matchedGT, predToGT




# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    # -------------------------
    # Argument parsing
    # -------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="versionD")
    parser.add_argument("--modelPath", default="./mnist_cnn.pth")
    parser.add_argument("--numImages", type=int, default=30)
    parser.add_argument(
        "--maxImages",
        type=int,
        default=-1,
        help="Maximum number of images to process (-1 = all)"
    )
    args = parser.parse_args()

    startTime = time.time()

    # -------------------------
    # Model setup
    # -------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelBetterCNN().to(device)
    model.load_state_dict(torch.load(args.modelPath, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # -------------------------
    # Load dataset
    # -------------------------

    baseDir = os.path.dirname(os.path.abspath(__file__))

    images = loadImagesFromUbyte(
        os.path.join(baseDir, "..", "improved_dataset",
                     args.version, "test-images-ubyte.bin"),
        10000, 128
    )

    gtAnnotations = loadLabelsUbyte(
        os.path.join(baseDir, "..", "improved_dataset",
                     args.version, "test-labels-ubyte.bin")
    )

    # -------------------------
    # Detection loop with progress bar
    # -------------------------

    results = []

    TP = FP = FN = 0
    allConfidences = []

    totalImages = len(images) if args.maxImages == -1 else min(args.maxImages, len(images))

    for idx in tqdm(range(totalImages), desc="Processing images"):
        image = images[idx]

        boxes, labels, windowStats = [], [], []

        for x, y, crop in slidingWindow(image, cfg.windowSize, cfg.stride):
            if rejectWindow(crop):
                continue

            crop28 = mnistStyleResize(crop)
            tensor = transform(crop28).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1)
                top2 = torch.topk(probs, 2).values.squeeze()
                conf, label = torch.max(probs, dim=1)

            if conf.item() < cfg.confidenceThreshold:
                continue

            if (top2[0] - top2[1]).item() < cfg.confidenceMargin:
                continue

            boxes.append((x, y, cfg.windowSize, cfg.windowSize))
            labels.append(label.item())
            windowStats.append(conf.item())

        groups = groupBoundingBoxes(boxes, labels, cfg.windowSize * 0.7)
        finalBoxes = [averageBoundingBox(g["boxes"]) for g in groups]
        finalLabels = [majorityVote(g["labels"]) for g in groups]

        tp, fp, fn, matchedPred, matchedGT, predToGT = matchDetections(
            finalBoxes, finalLabels, gtAnnotations[idx]
        )

        TP += tp
        FP += fp
        FN += fn
    
        for pIdx in matchedPred:
            allConfidences.append(windowStats[pIdx])

        if idx < args.numImages:
            results.append({
                "image": image,
                "boxes": finalBoxes,
                "labels": finalLabels,
                "matchedPred": matchedPred,
                "matchedGT": matchedGT,
                "gtBoxes": [(x, y, w, h) for (_, x, y, w, h) in gtAnnotations[idx]],
                "predToGT": predToGT,
            })

    totalTime = time.time() - startTime
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1Score = 2 * precision * recall / max(precision + recall, 1e-8)
    
    random.shuffle(results)

    # -------------------------------------------------
    # Visualization 
    # -------------------------------------------------

    currentImageIndex = [0]
    currentStatIndex = [0]

    fig = plt.figure(figsize=(9, 6))

    axImage = fig.add_axes([0.10, 0.45, 0.40, 0.40])
    axStats = fig.add_axes([0.08, 0.05, 0.45, 0.30])

    axNextImg = fig.add_axes([0.60, 0.65, 0.35, 0.10])
    axPrevImg = fig.add_axes([0.60, 0.53, 0.35, 0.10])

    axNextStat = fig.add_axes([0.60, 0.28, 0.35, 0.10])
    axPrevStat = fig.add_axes([0.60, 0.16, 0.35, 0.10])

    fig.text(
        0.98, 0.98,
        f"Total processing time: {totalTime:.2f} s",
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7)
    )


    def drawImage():
        

        idx = currentImageIndex[0]
        axImage.clear()
        data = results[idx]

        axImage.imshow(data["image"], cmap="gray")
        axImage.axis("off")

        for pIdx, ((x, y, w, h), label) in enumerate(zip(data["boxes"], data["labels"])):
            gtLabel = data["predToGT"].get(pIdx, None)
            if pIdx in data["matchedPred"]:
                if data["matchedPred"][pIdx]["correctClass"]:
                    color = "lime"
                    tag = "TP"
                else:
                    color = "orange"
                    tag = "FP (WR)"
            else:
                color = "red"
                tag = "FP"

            axImage.add_patch(
                patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
            )

            textY = y - 6 if y - 6 > 0 else y + h + 10
            axImage.text(
                x + 2, textY,
                f"{label} ({tag})",
                color="white",
                fontsize=4,
                bbox=dict(facecolor=color, alpha=0.85)
            )

            
            # Ground-truth label (NEW)
            if gtLabel is not None:
                axImage.text(
                    x + 25,
                    textY,
                    f"GT: {gtLabel}",
                    color="black",
                    fontsize=4,
                    bbox=dict(facecolor="yellow", alpha=0.9)
                )

        for gtIdx, (x, y, w, h) in enumerate(data["gtBoxes"]):
            if gtIdx not in data["matchedGT"]:
                axImage.add_patch(
                    patches.Rectangle((x, y), w, h, linewidth=2,
                                      edgecolor="blue", linestyle="--", facecolor="none")
                )

        axImage.set_title(f"Image {idx + 1}")
        fig.canvas.draw_idle()

    def drawMetrics():
        axStats.clear()
        vals = [precision, recall, f1Score]
        axStats.bar([f"Precision {round(precision, 2)}", f"Recall {round(recall, 2)}", f"F1 {round(f1Score, 2)}"], vals)
        axStats.set_ylim(0, 1)

    def drawConfidence():
        axStats.clear()
        axStats.hist(allConfidences, bins=30, range=(0, 1))
        axStats.set_title("Confidence distribution")

    statsFns = [drawMetrics, drawConfidence]

    def drawStats():
        statsFns[currentStatIndex[0]]()
        fig.canvas.draw_idle()

    def onNextImage(event):
        currentImageIndex[0] = min(len(results) - 1, currentImageIndex[0] + 1)
        drawImage()

    def onPrevImage(event):
        currentImageIndex[0] = max(0, currentImageIndex[0] - 1)
        drawImage()

    def onNextStat(event):
        currentStatIndex[0] = (currentStatIndex[0] + 1) % len(statsFns)
        drawStats()

    def onPrevStat(event):
        currentStatIndex[0] = (currentStatIndex[0] - 1) % len(statsFns)
        drawStats()

    # ------------------------
    # Buttons 
    # ------------------------

    btnNextImg = Button(axNextImg, "Next image")
    btnPrevImg = Button(axPrevImg, "Previous image")

    btnNextStat = Button(axNextStat, "Next stats")
    btnPrevStat = Button(axPrevStat, "Previous stats")

    btnNextImg.on_clicked(onNextImage)
    btnPrevImg.on_clicked(onPrevImage)

    btnNextStat.on_clicked(onNextStat)
    btnPrevStat.on_clicked(onPrevStat)

    drawImage()
    drawStats()
    plt.show()


if __name__ == "__main__":
    main()
