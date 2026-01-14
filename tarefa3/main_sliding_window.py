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


# -------------------------------------------------
# Utility functions for bounding box grouping
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
            distance = np.sqrt((cx - gcx)**2 + (cy - gcy)**2)

            if distance < distanceThreshold:
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
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    ws = [b[2] for b in boxes]
    hs = [b[3] for b in boxes]

    return (
        int(np.mean(xs)),
        int(np.mean(ys)),
        int(np.mean(ws)),
        int(np.mean(hs))
    )


def majorityVote(labels):
    return max(set(labels), key=labels.count)


# -------------------------------------------------
# Load binary images
# -------------------------------------------------
def loadImagesFromUbyte(filePath, numberOfImages, imageSize):
    data = np.fromfile(filePath, dtype=np.uint8)
    return data.reshape(numberOfImages, imageSize, imageSize)


# -------------------------------------------------
# Sliding window generator
# -------------------------------------------------
def slidingWindow(image, windowSize, stride):
    height, width = image.shape

    for y in range(0, height - windowSize + 1, stride):
        for x in range(0, width - windowSize + 1, stride):
            crop = image[y:y + windowSize, x:x + windowSize]
            yield x, y, crop


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    
    parser = argparse.ArgumentParser("Sliding Window Digit Detection")
    parser.add_argument("--version", required=True)
    parser.add_argument("--modelPath", required=False, default = "./mnist_cnn.pth")
    parser.add_argument("--numImages", type=int, default=20)
    args = parser.parse_args()

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------
    windowSize = 36  
    stride = 4
    confidenceThreshold = 0.9
    imageSize = 128

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelBetterCNN().to(device)
    model.load_state_dict(torch.load(args.modelPath, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------
    baseDir = os.path.dirname(os.path.abspath(__file__))
    binPath = os.path.join(
        baseDir,
        "..",
        "improved_dataset",
        args.version,
        "test-images-ubyte.bin"
    )

    images = loadImagesFromUbyte(binPath, 10000, imageSize)

    # -------------------------------------------------
    # Process multiple images
    # -------------------------------------------------
    results = []

    for imageIndex in range(args.numImages):
        image = images[imageIndex]

        boxes = []
        labels = []

        for x, y, crop in slidingWindow(image, windowSize, stride):

            if np.max(crop) < 40:
                continue

            if np.count_nonzero(crop > 40) < 50:
                continue

            ys, xs = np.where(crop > 40)
            if len(xs) == 0:
                continue

            if (
                xs.min() <= 1 or
                xs.max() >= crop.shape[1] - 1 or
                ys.min() <= 1 or
                ys.max() >= crop.shape[0] - 1
            ):
                continue

            resizedCrop = cv2.resize(crop, (28, 28))
            cropTensor = transform(resizedCrop).unsqueeze(0).to(device)


            with torch.no_grad():
                logits = model(cropTensor)
                probs = F.softmax(logits, dim=1)
                confidence, label = torch.max(probs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))

            if entropy.item() > 1.5:
                continue

            if confidence.item() > confidenceThreshold:
                boxes.append((x, y, windowSize, windowSize))
                labels.append(label.item())

        groups = groupBoundingBoxes(
            boxes,
            labels,
            distanceThreshold=windowSize * 0.6
        )

        finalBoxes = []
        finalLabels = []

        for group in groups:
            finalBoxes.append(averageBoundingBox(group["boxes"]))
            finalLabels.append(majorityVote(group["labels"]))

        results.append({
            "image": image,
            "boxes": finalBoxes,
            "labels": finalLabels
        })

    # -------------------------------------------------
    # MATLAB
    # -------------------------------------------------
    currentIndex = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)

    def drawImage(index):
        ax.clear()
        ax.imshow(results[index]["image"], cmap="gray")
        ax.axis("off")

        for (x, y, w, h), label in zip(
            results[index]["boxes"],
            results[index]["labels"]
        ):
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1,
                edgecolor="green",
                facecolor="none"
            )
            ax.add_patch(rect)

            ax.text(
                x, y - 2,
                str(label),
                color="red",
                fontsize=8
            )

        ax.set_title(f"Image {index + 1} / {len(results)}")
        fig.canvas.draw_idle()

    def nextImage(event):
        nonlocal currentIndex
        if currentIndex < len(results) - 1:
            currentIndex += 1
            drawImage(currentIndex)

    def previousImage(event):
        nonlocal currentIndex
        if currentIndex > 0:
            currentIndex -= 1
            drawImage(currentIndex)

    axPrev = plt.axes([0.25, 0.05, 0.2, 0.075])
    axNext = plt.axes([0.55, 0.05, 0.2, 0.075])

    btnPrev = Button(axPrev, "Previous")
    btnNext = Button(axNext, "Next")

    btnPrev.on_clicked(previousImage)
    btnNext.on_clicked(nextImage)

    drawImage(currentIndex)
    plt.show()


if __name__ == "__main__":
    main()
