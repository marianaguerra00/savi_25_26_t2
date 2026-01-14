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
# Detection configuration 
# -------------------------------------------------

class DetectionConfig:
    # Sliding window parameters
    windowSize = 36
    stride = 4

    # Foreground filtering parameters
    minForegroundRatio = 0.015        # Minimum ratio of active pixels
    foregroundThreshold = 30          # Pixel intensity threshold
    minMaxIntensityFactor = 2.0       # Std-based intensity threshold

    # Model confidence filtering
    confidenceThreshold = 0.6         # Minimum softmax confidence
    confidenceMargin = 0.25           # Difference between top-1 and top-2

    # Border rejection parameters
    borderMarginRatio = 0.05          # Margin as percentage of window size

# Instantiate configuration
cfg = DetectionConfig()


# -------------------------------------------------
# Utility functions for bounding box grouping
# -------------------------------------------------

def computeBoxCenter(box):
    # Compute center coordinates of a bounding box
    x, y, w, h = box
    return x + w / 2, y + h / 2


def groupBoundingBoxes(boxes, labels, distanceThreshold):
    # Group overlapping bounding boxes based on center distance
    groups = []

    for box, label in zip(boxes, labels):
        # Compute center of current bounding box
        cx, cy = computeBoxCenter(box)
        assigned = False

        # Try to assign the box to an existing group
        for group in groups:
            gcx, gcy = group["center"]
            distance = np.sqrt((cx - gcx)**2 + (cy - gcy)**2)

            # Merge if distance is below threshold
            if distance < distanceThreshold:
                group["boxes"].append(box)
                group["labels"].append(label)

                # Update group center using mean of box centers
                xs = [b[0] + b[2] / 2 for b in group["boxes"]]
                ys = [b[1] + b[3] / 2 for b in group["boxes"]]
                group["center"] = (np.mean(xs), np.mean(ys))

                assigned = True
                break

        # Create new group if not assigned
        if not assigned:
            groups.append({
                "boxes": [box],
                "labels": [label],
                "center": (cx, cy)
            })

    return groups


def averageBoundingBox(boxes):
    # Compute average bounding box from a group of boxes
    return tuple(map(int, np.mean(boxes, axis=0)))


def majorityVote(labels):
    # Return the most frequent label in the group
    return max(set(labels), key=labels.count)


# -------------------------------------------------
# Image utilities
# -------------------------------------------------

def loadImagesFromUbyte(filePath, numberOfImages, imageSize):
    # Load raw binary image data and reshape
    data = np.fromfile(filePath, dtype=np.uint8)
    return data.reshape(numberOfImages, imageSize, imageSize)


def slidingWindow(image, windowSize, stride):
    # Generate sliding window crops over the image
    h, w = image.shape
    for y in range(0, h - windowSize + 1, stride):
        for x in range(0, w - windowSize + 1, stride):
            yield x, y, image[y:y + windowSize, x:x + windowSize]


def mnistStyleResize(crop, threshold=40):
    # Extract foreground pixels
    ys, xs = np.where(crop > threshold)

    # Fallback if no foreground detected
    if len(xs) == 0:
        return cv2.resize(crop, (28, 28))

    # Crop bounding box around digit
    digit = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = digit.shape
    size = max(h, w)

    # Pad digit to square shape
    padded = np.zeros((size, size), dtype=np.uint8)
    padded[(size - h)//2:(size - h)//2 + h,
           (size - w)//2:(size - w)//2 + w] = digit

    # Resize to MNIST digit size
    resized = cv2.resize(padded, (20, 20))

    # Center digit in 28x28 image
    final = np.zeros((28, 28), dtype=np.uint8)
    final[4:24, 4:24] = resized
    return final


# -------------------------------------------------
# Window rejection logic (DECLARATIVE)
# -------------------------------------------------

def rejectWindow(crop):
    # Reject window if max intensity is too low compared to local statistics
    if np.max(crop) < np.mean(crop) + cfg.minMaxIntensityFactor * np.std(crop):
        return True

    # Reject window if foreground pixel ratio is too small
    fgRatio = np.count_nonzero(crop > cfg.foregroundThreshold) / crop.size
    if fgRatio < cfg.minForegroundRatio:
        return True

    # Extract foreground coordinates for border checking
    ys, xs = np.where(crop > cfg.foregroundThreshold)
    if len(xs) == 0:
        return True

    # Reject window if digit touches window border
    margin = int(cfg.borderMarginRatio * crop.shape[0])
    if (
        xs.min() <= margin or
        xs.max() >= crop.shape[1] - margin or
        ys.min() <= margin or
        ys.max() >= crop.shape[0] - margin
    ):
        return True

    return False


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=False, default= "versionD")
    parser.add_argument("--modelPath", default="./mnist_cnn.pth")
    parser.add_argument("--numImages", type=int, default=30)
    args = parser.parse_args()

    # Select computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained CNN model
    model = ModelBetterCNN().to(device)
    model.load_state_dict(torch.load(args.modelPath, map_location=device))
    model.eval()

    # Define MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Build dataset path
    baseDir = os.path.dirname(os.path.abspath(__file__))
    binPath = os.path.join(
        baseDir, "..", "improved_dataset",
        args.version, "test-images-ubyte.bin"
    )

    # Load images
    images = loadImagesFromUbyte(binPath, 10000, 128)

    results = []

    # Process selected number of images
    for idx in range(args.numImages):
        image = images[idx]
        boxes, labels = [], []

        # Run sliding window detection
        for x, y, crop in slidingWindow(image, cfg.windowSize, cfg.stride):

            # Reject invalid windows
            if rejectWindow(crop):
                continue

            # Resize crop to MNIST format
            crop28 = mnistStyleResize(crop)
            tensor = transform(crop28).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1)
                top2 = torch.topk(probs, 2).values.squeeze()
                confidence, label = torch.max(probs, dim=1)


            if confidence.item() < cfg.confidenceThreshold:
                continue

            if (top2[0] - top2[1]).item() < cfg.confidenceMargin:
                continue

            # Store valid detection
            boxes.append((x, y, cfg.windowSize, cfg.windowSize))
            labels.append(label.item())

        # Group overlapping detections
        groups = groupBoundingBoxes(
            boxes, labels,
            distanceThreshold=cfg.windowSize * 0.7
        )

        # Store final averaged results
        results.append({
            "image": image,
            "boxes": [averageBoundingBox(g["boxes"]) for g in groups],
            "labels": [majorityVote(g["labels"]) for g in groups]
        })


    # -------------------------------------------------
    # MATLAB
    # -------------------------------------------------

    currentIndex = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)

    def drawImage(index):
        # Display image and predicted bounding boxes
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
        # Move to next image
        nonlocal currentIndex
        if currentIndex < len(results) - 1:
            currentIndex += 1
            drawImage(currentIndex)

    def previousImage(event):
        # Move to previous image
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
