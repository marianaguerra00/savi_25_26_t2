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
    # Compute the center (cx, cy) of a bounding box
    x, y, w, h = box
    return x + w / 2, y + h / 2


def groupBoundingBoxes(boxes, labels, distanceThreshold):
    # Group nearby bounding boxes to remove duplicate detections
    groups = []

    for box, label in zip(boxes, labels):
        # Compute center of the current box
        cx, cy = computeBoxCenter(box)
        assigned = False

        # Try to match the box with an existing group
        for group in groups:
            gcx, gcy = group["center"]

            # Euclidean distance between centers
            distance = np.sqrt((cx - gcx)**2 + (cy - gcy)**2)

            # If close enough, merge into the group
            if distance < distanceThreshold:
                group["boxes"].append(box)
                group["labels"].append(label)

                # Recompute group center as the mean of all box centers
                xs = [b[0] + b[2] / 2 for b in group["boxes"]]
                ys = [b[1] + b[3] / 2 for b in group["boxes"]]
                group["center"] = (np.mean(xs), np.mean(ys))

                assigned = True
                break

        # If not assigned to any group, create a new group
        if not assigned:
            groups.append({
                "boxes": [box],
                "labels": [label],
                "center": (cx, cy)
            })

    return groups


def averageBoundingBox(boxes):
    # Average multiple bounding boxes into a single box
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
    # Return the most common label in the group
    return max(set(labels), key=labels.count)


# -------------------------------------------------
# Load binary images
# -------------------------------------------------

def loadImagesFromUbyte(filePath, numberOfImages, imageSize):
    # Load raw binary image data and reshape to (N, H, W)
    data = np.fromfile(filePath, dtype=np.uint8)
    return data.reshape(numberOfImages, imageSize, imageSize)


# -------------------------------------------------
# Sliding window generator
# -------------------------------------------------

def slidingWindow(image, windowSize, stride):
    # Slide a window over the image and yield crops
    height, width = image.shape

    for y in range(0, height - windowSize + 1, stride):
        for x in range(0, width - windowSize + 1, stride):
            crop = image[y:y + windowSize, x:x + windowSize]
            yield x, y, crop


def mnistStyleResize(crop, threshold=40):
    # Find foreground pixels using a threshold
    ys, xs = np.where(crop > threshold)

    # If no foreground pixels, fallback to direct resize
    if len(xs) == 0:
        return cv2.resize(crop, (28, 28))

    # Compute bounding box around the digit
    xMin, xMax = xs.min(), xs.max()
    yMin, yMax = ys.min(), ys.max()

    digit = crop[yMin:yMax + 1, xMin:xMax + 1]

    # Make the digit square by padding
    h, w = digit.shape
    size = max(h, w)

    padded = np.zeros((size, size), dtype=np.uint8)
    yOffset = (size - h) // 2
    xOffset = (size - w) // 2
    padded[yOffset:yOffset + h, xOffset:xOffset + w] = digit

    # Resize digit to 20x20 (MNIST format)
    resizedDigit = cv2.resize(padded, (20, 20))

    # Place digit in the center of a 28x28 canvas
    finalImage = np.zeros((28, 28), dtype=np.uint8)
    finalImage[4:24, 4:24] = resizedDigit

    return finalImage


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser("Sliding Window Digit Detection")
    parser.add_argument("--version", required=True)
    parser.add_argument("--modelPath", required=False, default="./mnist_cnn.pth")
    parser.add_argument("--numImages", type=int, default=20)
    args = parser.parse_args()

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    windowSize = 36              # Sliding window size
    stride = 4                   # Step size between windows
    confidenceThreshold = 0.5    # Minimum softmax confidence
    imageSize = 128              # Input image resolution

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained CNN model
    model = ModelBetterCNN().to(device)
    model.load_state_dict(torch.load(args.modelPath, map_location=device))
    model.eval()

    # MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    # Build path to binary test images
    baseDir = os.path.dirname(os.path.abspath(__file__))
    binPath = os.path.join(
        baseDir,
        "..",
        "improved_dataset",
        args.version,
        "test-images-ubyte.bin"
    )

    # Load images from binary file
    images = loadImagesFromUbyte(binPath, 10000, imageSize)

    # -------------------------------------------------
    # Process multiple images
    # -------------------------------------------------

    results = []

    for imageIndex in range(args.numImages):
        image = images[imageIndex]

        boxes = []
        labels = []

        # Run sliding window detection
        for x, y, crop in slidingWindow(image, windowSize, stride):

            # Skip dark or empty windows
            if np.max(crop) < 25:
                continue

            if np.count_nonzero(crop > 30) < 25:
                continue

            # Extract foreground pixels
            ys, xs = np.where(crop > 40)
            if len(xs) == 0:
                continue

            # Reject crops touching window borders
            if (
                xs.min() <= 0 or
                xs.max() >= crop.shape[1] - 1 or
                ys.min() <= 0 or
                ys.max() >= crop.shape[0] - 1
            ):
                continue

            # Resize crop to MNIST-style input
            processedCrop = mnistStyleResize(crop)
            cropTensor = transform(processedCrop).unsqueeze(0).to(device)

            # Run model inference
            with torch.no_grad():
                logits = model(cropTensor)
                probs = F.softmax(logits, dim=1)
                confidence, label = torch.max(probs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))

            # Reject uncertain predictions
            if entropy.item() > 1.5:
                continue

            # Keep high-confidence detections
            if confidence.item() > confidenceThreshold:
                boxes.append((x, y, windowSize, windowSize))
                labels.append(label.item())

        # Group overlapping detections
        groups = groupBoundingBoxes(
            boxes,
            labels,
            distanceThreshold=windowSize * 0.6
        )

        finalBoxes = []
        finalLabels = []

        # Average boxes and vote labels per group
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
        # Draw image and bounding boxes
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
        # Go to next image
        nonlocal currentIndex
        if currentIndex < len(results) - 1:
            currentIndex += 1
            drawImage(currentIndex)

    def previousImage(event):
        # Go to previous image
        nonlocal currentIndex
        if currentIndex > 0:
            currentIndex -= 1
            drawImage(currentIndex)

    # Create navigation buttons
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
