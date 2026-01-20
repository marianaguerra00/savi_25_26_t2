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
import matplotlib.gridspec as gridspec
import random



# -------------------------------------------------
# Detection configuration
# -------------------------------------------------

class DetectionConfig:
    windowSize = 36
    stride = 4

    minForegroundRatio = 0.015
    foregroundThreshold = 30
    minMaxIntensityFactor = 2.0

    confidenceThreshold = 0.6
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
                annotations.append((digitLabel, x, y, w, h))

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
# Detection evaluation utilities
# -------------------------------------------------

def computeIoU(boxA, boxB):
    ax, ay, aw, ah = map(int, boxA)
    bx, by, bw, bh = map(int, boxB)

    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    unionArea = aw * ah + bw * bh - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def matchDetections(predBoxes, predLabels, gtAnnotations, iouThreshold=0.5):
    matchedCorrect = set()
    matchedPred = set()          # <<< CHANGED
    tp = fp = 0

    for pIdx, (pBox, pLabel) in enumerate(zip(predBoxes, predLabels)):
        bestIou = 0
        bestIdx = -1

        for idx, (gtLabel, x, y, w, h) in enumerate(gtAnnotations):
            iou = computeIoU(pBox, (x, y, w, h))
            if iou > bestIou:
                bestIou = iou
                bestIdx = idx

        if bestIou >= iouThreshold and bestIdx not in matchedCorrect:
            if pLabel == gtAnnotations[bestIdx][0]:
                tp += 1
                matchedCorrect.add(bestIdx)
                matchedPred.add(pIdx)   # <<< CHANGED
            else:
                fp += 1
        else:
            fp += 1

    fn = len(gtAnnotations) - len(matchedCorrect)
    return tp, fp, fn, matchedPred   # <<< CHANGED


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="versionD")
    parser.add_argument("--modelPath", default="./mnist_cnn.pth")
    parser.add_argument("--numImages", type=int, default=30)
    args = parser.parse_args()

    maxDebugImages = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelBetterCNN().to(device)
    model.load_state_dict(torch.load(args.modelPath, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

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

    results = []

    totalTruePositives = 0
    totalFalsePositives = 0
    totalFalseNegatives = 0

    allConfidences = []
    allMargins = []
    allEntropies = []

    for idx, image in enumerate(images):

        if maxDebugImages > 0 and idx >= maxDebugImages:
            break

        boxes, labels, windowStats = [], [], []

        for x, y, crop in slidingWindow(image, cfg.windowSize, cfg.stride):
            if rejectWindow(crop):
                continue

            crop28 = mnistStyleResize(crop)
            tensor = transform(crop28).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1)
                top2 = torch.topk(probs, 2).values.squeeze()
                confidence, label = torch.max(probs, dim=1)

            if confidence.item() < cfg.confidenceThreshold:
                continue

            if (top2[0] - top2[1]).item() < cfg.confidenceMargin:
                continue

            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

            boxes.append((x, y, cfg.windowSize, cfg.windowSize))
            labels.append(label.item())
            windowStats.append((confidence.item(),
                                (top2[0] - top2[1]).item(),
                                entropy))

        groups = groupBoundingBoxes(boxes, labels, cfg.windowSize * 0.7)

        finalBoxes = [averageBoundingBox(g["boxes"]) for g in groups]
        finalLabels = [majorityVote(g["labels"]) for g in groups]

        tp, fp, fn, matchedPred = matchDetections(
            finalBoxes, finalLabels, gtAnnotations[idx]
        )

        totalTruePositives += tp
        totalFalsePositives += fp
        totalFalseNegatives += fn

        # <<< CHANGED: only collect stats from TRUE POSITIVES
        for pIdx in matchedPred:
            group = groups[pIdx]
            groupStats = []
            for box in group["boxes"]:
                idxBox = boxes.index(box)
                groupStats.append(windowStats[idxBox])

            best = max(groupStats, key=lambda x: x[0])
            allConfidences.append(best[0])
            allMargins.append(best[1])
            allEntropies.append(best[2])

        if idx < args.numImages:
            results.append({
                "image": image,
                "boxes": finalBoxes,
                "labels": finalLabels
            })

    

    precision = totalTruePositives / max(
        totalTruePositives + totalFalsePositives, 1
    )

    recall = totalTruePositives / max(
        totalTruePositives + totalFalseNegatives, 1
    )

    f1Score = 2 * precision * recall / max(
        precision + recall, 1e-8
    )

    print("===== TEST SET EVALUATION =====")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1Score:.3f}")
    print(f"Mean confidence: {np.mean(allConfidences):.3f}")
    print(f"Mean margin:     {np.mean(allMargins):.3f}")
    print(f"Mean entropy:    {np.mean(allEntropies):.3f}")

    random.shuffle(results)
    # -------------------------------------------------
    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------

    # ------------------------
    # State (Matplotlib-safe)
    # ------------------------

    currentImageIndex = [0]
    currentStatIndex = [0]

    # ------------------------
    # Figure and axes
    # ------------------------

    fig = plt.figure(figsize=(9, 6))

    # Image axis (smaller, centered)
    axImage = fig.add_axes([0.12, 0.48, 0.38, 0.38])

    # Stats axis (bottom)
    axStats = fig.add_axes([0.08, 0.05, 0.45, 0.32])



    # Buttons (right side)
    axNextImg = fig.add_axes([0.62, 0.65, 0.32, 0.10])
    axPrevImg = fig.add_axes([0.62, 0.53, 0.32, 0.10])

    axNextStat = fig.add_axes([0.62, 0.28, 0.32, 0.10])
    axPrevStat = fig.add_axes([0.62, 0.16, 0.32, 0.10])

    # ------------------------
    # Image drawing
    # ------------------------

    def drawImage():
        idx = currentImageIndex[0]
        axImage.clear()

        axImage.imshow(results[idx]["image"], cmap="gray")
        axImage.axis("off")

        for (x, y, w, h), label in zip(
            results[idx]["boxes"],
            results[idx]["labels"]
        ):
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            axImage.add_patch(rect)

            axImage.text(
                x,
                y - 5,
                str(label),
                color="white",
                fontsize=10,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="red", edgecolor="none", pad=2)
            )

        axImage.set_title(f"Image {idx + 1}")
        fig.canvas.draw_idle()

    # ------------------------
    # Stats plot 1 — Detection metrics
    # ------------------------

    def drawMetricsStats():
        axStats.clear()

        values = [precision, recall, f1Score]
        labels = ["Precision", "Recall", "F1-score"]

        axStats.bar(labels, values)
        axStats.set_ylim(0, 1)
        axStats.set_ylabel("Score")
        axStats.set_title("Detection performance metrics")

        for i, v in enumerate(values):
            axStats.text(i, v + 0.02, f"{v:.3f}", ha="center")

    # ------------------------
    # Stats plot 2 — Confidence distribution
    # ------------------------

    def drawConfidenceDistribution():
        axStats.clear()

        axStats.hist(
            allConfidences,
            bins=30,
            range=(0.0, 1.0)
        )

        axStats.set_xlabel("Confidence")
        axStats.set_ylabel("Number of detections")
        axStats.set_title("Confidence distribution")

    # ------------------------
    # Stats controller
    # ------------------------

    statsDrawFunctions = [
        drawMetricsStats,
        drawConfidenceDistribution
    ]

    def drawStats():
        statsDrawFunctions[currentStatIndex[0]]()
        fig.canvas.draw_idle()

    # ------------------------
    # Button callbacks
    # ------------------------

    def onNextImage(event):
        currentImageIndex[0] = min(
            len(results) - 1,
            currentImageIndex[0] + 1
        )
        drawImage()

    def onPrevImage(event):
        currentImageIndex[0] = max(
            0,
            currentImageIndex[0] - 1
        )
        drawImage()

    def onNextStat(event):
        currentStatIndex[0] = (
            currentStatIndex[0] + 1
        ) % len(statsDrawFunctions)
        drawStats()

    def onPrevStat(event):
        currentStatIndex[0] = (
            currentStatIndex[0] - 1
        ) % len(statsDrawFunctions)
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

    # ------------------------
    # Initial draw
    # ------------------------

    drawImage()
    drawStats()
    plt.show()







if __name__ == "__main__":
    main()
