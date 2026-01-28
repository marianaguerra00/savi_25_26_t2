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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns


def loadImagesFromUbyte(filePath, numberOfImages, imageSize):
    """Load images from ubyte binary format"""
    data = np.fromfile(filePath, dtype=np.uint8)
    return data.reshape(numberOfImages, imageSize, imageSize)


def loadLabelsUbyte(filePath):
    """Load annotations from custom ubyte format"""
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


def slidingWindow(image, windowSize, stride):
    """Generate sliding windows over the image"""
    h, w = image.shape
    for y in range(0, h - windowSize + 1, stride):
        for x in range(0, w - windowSize + 1, stride):
            yield x, y, image[y:y + windowSize, x:x + windowSize]


def mnistStyleResize(crop, threshold=40):
    """Resize crop to 28x28 matching MNIST preprocessing"""
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


def rejectWindow(crop, foregroundThreshold, minForegroundRatio, 
                 minMaxIntensityFactor, borderMarginRatio):
    """Filter out low-quality windows using multiple heuristics"""
    if np.max(crop) < np.mean(crop) + minMaxIntensityFactor * np.std(crop):
        return True

    fgRatio = np.count_nonzero(crop > foregroundThreshold) / crop.size
    if fgRatio < minForegroundRatio:
        return True

    ys, xs = np.where(crop > foregroundThreshold)
    if len(xs) == 0:
        return True

    margin = int(borderMarginRatio * crop.shape[0])
    if (xs.min() <= margin or xs.max() >= crop.shape[1] - margin or
        ys.min() <= margin or ys.max() >= crop.shape[0] - margin):
        return True

    return False


def computeBoxCenter(box):
    """Calculate center coordinates of a bounding box"""
    x, y, w, h = box
    return x + w / 2, y + h / 2


def groupBoundingBoxes(boxes, labels, distanceThreshold):
    """Group nearby bounding boxes into single detections"""
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
    """Compute average bounding box from a group"""
    return tuple(map(int, np.mean(boxes, axis=0)))


def majorityVote(labels):
    """Determine label by majority voting"""
    return max(set(labels), key=labels.count)


def computeIoU(boxA, boxB):
    """Calculate Intersection over Union between two boxes"""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    unionArea = aw * ah + bw * bh - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def matchDetections(predBoxes, predLabels, gtAnnotations, iouThreshold):
    """
    Match predictions to ground truth using IoU threshold.
    Returns match info and classification data for matched detections.
    """
    matchedGT = set()
    matchedPred = {}
    predToGT = {}
    allTrueLabels = []
    allPredLabels = []
    allIoUs = []

    for pIdx, (pBox, pLabel) in enumerate(zip(predBoxes, predLabels)):
        bestIou = 0.0
        bestGT = -1
        bestGTLabel = None

        for gtIdx, (gtLabel, gx, gy, gw, gh) in enumerate(gtAnnotations):
            if gtIdx in matchedGT:
                continue
            iou = computeIoU(pBox, (gx, gy, gw, gh))
            if iou > bestIou:
                bestIou = iou
                bestGT = gtIdx
                bestGTLabel = gtLabel

        if bestGT != -1 and bestIou >= iouThreshold:
            matchedGT.add(bestGT)
            correctClass = (pLabel == bestGTLabel)
            matchedPred[pIdx] = {"gtIdx": bestGT, "correctClass": correctClass}
            predToGT[pIdx] = bestGTLabel
            allIoUs.append(bestIou)
            
            # Track for classification metrics
            allTrueLabels.append(bestGTLabel)
            allPredLabels.append(pLabel)

    return matchedPred, matchedGT, predToGT, allTrueLabels, allPredLabels, allIoUs


def detectImage(image, model, device, transform, windowSize, stride, 
                foregroundThreshold, minForegroundRatio, minMaxIntensityFactor, 
                borderMarginRatio, confidenceThreshold, confidenceMargin, groupingDistance):
    """
    Perform sliding window detection on a single image.
    Returns final bounding boxes, labels, and confidence scores.
    """
    boxes, labels, windowStats = [], [], []

    for x, y, crop in slidingWindow(image, windowSize, stride):
        if rejectWindow(crop, foregroundThreshold, minForegroundRatio, 
                       minMaxIntensityFactor, borderMarginRatio):
            continue

        crop28 = mnistStyleResize(crop)
        tensor = transform(crop28).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)
            top2 = torch.topk(probs, 2).values.squeeze()
            conf, label = torch.max(probs, dim=1)

        if conf.item() < confidenceThreshold:
            continue
        if (top2[0] - top2[1]).item() < confidenceMargin:
            continue

        boxes.append((x, y, windowSize, windowSize))
        labels.append(label.item())
        windowStats.append(conf.item())

    groups = groupBoundingBoxes(boxes, labels, groupingDistance)
    finalBoxes = [averageBoundingBox(g["boxes"]) for g in groups]
    finalLabels = [majorityVote(g["labels"]) for g in groups]

    return finalBoxes, finalLabels, windowStats


def visualize(results, accuracy, precision, recall, f1Score, cm, 
             meanIoU, detectionAcc, totalTime):
    """
    Interactive visualization UI matching FPN interface.
    Provides 3 metric views: Localization, Classification, Confusion Matrix.
    """
    currentImageIndex = [0]
    currentStatsIndex = [0]

    fig = plt.figure(figsize=(10, 7))
    fig.canvas.manager.set_window_title('Sliding Window Detection Results')

    axImage = fig.add_axes([0.15, 0.50, 0.45, 0.40])
    axStats = fig.add_axes([0.08, 0.10, 0.60, 0.30])

    axNextImg = fig.add_axes([0.72, 0.65, 0.25, 0.10])
    axPrevImg = fig.add_axes([0.72, 0.53, 0.25, 0.10])
    axNextStat = fig.add_axes([0.72, 0.25, 0.25, 0.10])
    axPrevStat = fig.add_axes([0.72, 0.13, 0.25, 0.10])

    fig.text(
        0.98, 0.98,
        f"Processing time: {totalTime:.2f}s",
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7),
        color="white"
    )

    def drawImage():
        axImage.clear()
        idx = currentImageIndex[0]
        data = results[idx]

        axImage.imshow(data["image"], cmap="gray")
        axImage.axis("off")

        # Draw unmatched ground truth (blue dashed)
        for gtIdx, (x, y, w, h) in enumerate(data["gtBoxes"]):
            if gtIdx not in data["matchedGT"]:
                axImage.add_patch(
                    patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1.5,
                        edgecolor="blue",
                        facecolor="none",
                        linestyle="--"
                    )
                )

        # Draw predictions (green=correct, red=incorrect)
        for pIdx, ((x, y, w, h), label) in enumerate(zip(data["boxes"], data["labels"])):
            gtLabel = data["predToGT"].get(pIdx, None)
            
            if pIdx in data["matchedPred"]:
                isCorrect = data["matchedPred"][pIdx]["correctClass"]
                color = "lime" if isCorrect else "red"
            else:
                color = "red"
                isCorrect = False

            axImage.add_patch(
                patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none"
                )
            )

            textY = y - 6 if y > 10 else y + h + 6
            labelText = f"{label}"
            if gtLabel is not None:
                labelText += f" (GT: {gtLabel})"
            
            axImage.text(
                x + 2, textY, labelText,
                color="white",
                fontsize=6,
                bbox=dict(facecolor=color, alpha=0.85)
            )

        axImage.set_title(
            f"Image {idx + 1}/{len(results)}\n"
            f"Blue Dash=GT | Green=Correct | Red=Wrong"
        )
        fig.canvas.draw_idle()

    def drawLocalizationMetrics():
        """View 0: Localization Metrics"""
        axStats.clear()
        
        metrics = [meanIoU, detectionAcc]
        labels = ["Mean IoU", "Detection Acc (IoU > 0.5)"]
        colors = ['#3498db', '#2ecc71']
        
        bars = axStats.bar(labels, metrics, color=colors, alpha=0.7, width=0.5)
        axStats.set_ylim(0, 1)
        axStats.set_title("Localization Metrics (BBox Quality)", 
                         fontweight='bold', pad=10)
        axStats.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            axStats.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axStats.text(0.50, -0.15,
                    "IoU measures bbox overlap | Detection Acc = % boxes with IoU > 0.5",
                    ha='center', va='top', transform=axStats.transAxes,
                    fontsize=8, style='italic', color='gray')
        fig.canvas.draw_idle()

    def drawClassificationMetrics():
        """View 1: Classification Metrics"""
        axStats.clear()
        values = [accuracy, precision, recall, f1Score]
        labels = [
            f"Accuracy\n{accuracy:.3f}",
            f"Precision\n{precision:.3f}",
            f"Recall\n{recall:.3f}",
            f"F1\n{f1Score:.3f}"
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = axStats.bar(labels, values, color=colors, alpha=0.7)
        
        axStats.set_ylim(0, 1.05)
        axStats.set_title("Classification Metrics (Digit Recognition)", 
                        fontweight='bold', pad=15)
        axStats.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            axStats.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        fig.canvas.draw_idle()

    def drawConfusionMatrix():
        """View 2: Confusion Matrix"""
        axStats.clear()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            ax=axStats, cbar=False, square=False
        )
        axStats.set_title("Confusion Matrix (Digit Classes)", fontweight='bold')
        axStats.set_xlabel("Predicted")
        axStats.set_ylabel("True")
        fig.canvas.draw_idle()

    statsFns = [drawLocalizationMetrics, drawClassificationMetrics, drawConfusionMatrix]

    def drawStats():
        statsFns[currentStatsIndex[0]]()

    def onNextImage(event):
        currentImageIndex[0] = min(len(results) - 1, currentImageIndex[0] + 1)
        drawImage()

    def onPrevImage(event):
        currentImageIndex[0] = max(0, currentImageIndex[0] - 1)
        drawImage()

    def onNextStats(event):
        currentStatsIndex[0] = (currentStatsIndex[0] + 1) % len(statsFns)
        drawStats()

    def onPrevStats(event):
        currentStatsIndex[0] = (currentStatsIndex[0] - 1) % len(statsFns)
        drawStats()

    btnNextImg = Button(axNextImg, "Next →")
    btnPrevImg = Button(axPrevImg, "← Previous")
    btnNextStat = Button(axNextStat, "Next Stats →")
    btnPrevStat = Button(axPrevStat, "← Previous Stats")

    btnNextImg.on_clicked(onNextImage)
    btnPrevImg.on_clicked(onPrevImage)
    btnNextStat.on_clicked(onNextStats)
    btnPrevStat.on_clicked(onPrevStats)

    drawImage()
    drawStats()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='MNIST Sliding Window Detection with Interactive Visualization'
    )
    parser.add_argument("--version", default="versionD",
                       help="Dataset version folder (default: versionD)")
    parser.add_argument("--modelPath", default="./mnist_cnn.pth",
                       help="Path to trained model (default: ./mnist_cnn.pth)")
    parser.add_argument("--numImages", type=int, default=30,
                       help="Number of images to visualize (default: 30)")
    parser.add_argument("--maxImages", type=int, default=-1,
                       help="Max images to process, -1 for all (default: -1)")
    args = parser.parse_args()

    # Detection parameters
    windowSize = 36
    stride = 4
    minForegroundRatio = 0.015
    foregroundThreshold = 30
    minMaxIntensityFactor = 2.0
    confidenceThreshold = 0.7
    confidenceMargin = 0.25
    borderMarginRatio = 0.05
    groupingDistance = windowSize * 0.7
    iouThreshold = 0.3

    print("\n" + "="*60)
    print("MNIST SLIDING WINDOW DETECTION")
    print("="*60)
    print(f"Window Size: {windowSize}x{windowSize}")
    print(f"Stride: {stride}")
    print(f"Confidence Threshold: {confidenceThreshold}")
    print(f"IoU Threshold: {iouThreshold}")
    print("="*60 + "\n")

    startTime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    model = ModelBetterCNN().to(device)
    model.load_state_dict(torch.load(args.modelPath, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load dataset
    baseDir = os.path.dirname(os.path.abspath(__file__))
    images = loadImagesFromUbyte(
        os.path.join(baseDir, "..", "improved_dataset", args.version, "test-images-ubyte.bin"),
        10000, 128
    )
    gtAnnotations = loadLabelsUbyte(
        os.path.join(baseDir, "..", "improved_dataset", args.version, "test-labels-ubyte.bin")
    )

    results = []
    allTrueLabels = []
    allPredLabels = []
    allIoUs = []
    detectionCorrect = 0
    detectionTotal = 0
    
    totalImages = len(images) if args.maxImages == -1 else min(args.maxImages, len(images))

    # Process images
    for idx in tqdm(range(totalImages), desc="Processing images"):
        image = images[idx]
        finalBoxes, finalLabels, windowStats = detectImage(
            image, model, device, transform, windowSize, stride, 
            foregroundThreshold, minForegroundRatio, minMaxIntensityFactor, 
            borderMarginRatio, confidenceThreshold, confidenceMargin, groupingDistance
        )

        matchedPred, matchedGT, predToGT, trueLabels, predLabels, ious = matchDetections(
            finalBoxes, finalLabels, gtAnnotations[idx], iouThreshold
        )
        
        allTrueLabels.extend(trueLabels)
        allPredLabels.extend(predLabels)
        allIoUs.extend(ious)
        
        for iou in ious:
            detectionTotal += 1
            if iou > 0.5:
                detectionCorrect += 1

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
    
    # Calculate metrics
    accuracy = accuracy_score(allTrueLabels, allPredLabels) if allTrueLabels else 0.0
    precision = precision_score(allTrueLabels, allPredLabels, average="macro", zero_division=0)
    recall = recall_score(allTrueLabels, allPredLabels, average="macro", zero_division=0)
    f1Score = f1_score(allTrueLabels, allPredLabels, average="macro", zero_division=0)
    cm = confusion_matrix(allTrueLabels, allPredLabels) if allTrueLabels else np.zeros((10,10))
    
    meanIoU = np.mean(allIoUs) if allIoUs else 0.0
    detectionAcc = (detectionCorrect / detectionTotal) if detectionTotal > 0 else 0.0
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print("Classification Metrics:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1Score:.4f}")
    print("\nDetection Metrics:")
    print(f"  Mean IoU        : {meanIoU:.4f}")
    print(f"  Detection Acc   : {detectionAcc:.4f} (IoU > 0.5)")
    print(f"  Total detections: {detectionTotal}")
    print(f"\nEvaluation time: {totalTime:.2f} s")
    print("=" * 60 + "\n")
    
    if len(results) == 0:
        print("No images available for visualization.")
        return
    
    random.shuffle(results)
    visualize(results, accuracy, precision, recall, f1Score, cm, 
             meanIoU, detectionAcc, totalTime)


if __name__ == "__main__":
    main()