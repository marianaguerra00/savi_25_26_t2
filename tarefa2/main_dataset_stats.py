import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Slider, Button


# -------------------------------------------------
# LOAD UBYTE IMAGES
# -------------------------------------------------

def loadImagesUbyte(filePath, numberOfImages, imageSize):
    # Load raw uint8 image data from binary file
    data = np.fromfile(filePath, dtype=np.uint8)
    return data.reshape(numberOfImages, imageSize, imageSize)


# -------------------------------------------------
# LOAD UBYTE LABELS
# -------------------------------------------------

def loadLabelsUbyte(filePath):
    # Load labels and bounding boxes from binary file
    annotationsPerImage = []

    with open(filePath, "rb") as file:
        while True:
            # Read number of objects in the image
            numObjectsBytes = file.read(1)
            if not numObjectsBytes:
                break

            numObjects = np.frombuffer(
                numObjectsBytes, dtype=np.uint8
            )[0]

            annotations = []

            # Read each object annotation
            for _ in range(numObjects):
                digitLabel = np.frombuffer(
                    file.read(1), dtype=np.uint8
                )[0]

                x, y, w, h = np.frombuffer(
                    file.read(8), dtype=np.uint16
                )

                annotations.append(
                    (digitLabel, x, y, w, h)
                )

            annotationsPerImage.append(annotations)

    return annotationsPerImage


# -------------------------------------------------
# VISUALIZATION Interface
# -------------------------------------------------

def runInterface(images, annotations, digitCounts, classCounts, boxSizes):

    # Mutable indices for callbacks
    currentImageIndex = [0]
    currentStatIndex = [0]

    fig = plt.figure(figsize=(12, 7))

    # --------------------
    # LEFT SIDE (CONTENT)
    # --------------------

    # Image display axis
    axImage = plt.axes([0.10, 0.55, 0.45, 0.4])
    axImage.axis("off")

    # Statistics display axis
    axStats = plt.axes([0.10, 0.10, 0.45, 0.35])

    # ---------------------------
    # RIGHT SIDE (BUTTONS)
    # ---------------------------

    buttonWidth = 0.20
    buttonHeight = 0.08
    buttonX = 0.60

    axNextImg = plt.axes([buttonX, 0.75, buttonWidth, buttonHeight])
    axPrevImg = plt.axes([buttonX, 0.65, buttonWidth, buttonHeight])
    
    axNextStat = plt.axes([buttonX, 0.30, buttonWidth, buttonHeight])
    axPrevStat = plt.axes([buttonX, 0.20, buttonWidth, buttonHeight])
    

    # ------------------
    # DRAW FUNCTIONS
    # ------------------

    def drawImage():
        # Draw current image with bounding boxes
        axImage.clear()
        axImage.axis("off")
        axImage.set_title(
            f"Image {currentImageIndex[0]} with Bounding Boxes",
            fontsize=13
        )

        axImage.imshow(images[currentImageIndex[0]], cmap="gray")

        for digitLabel, x, y, w, h in annotations[currentImageIndex[0]]:
            rect = plt.Rectangle(
                (x, y), w, h,
                edgecolor="red",
                facecolor="none",
                linewidth=2
            )
            axImage.add_patch(rect)

    def drawStats():
        # Draw selected statistics plot
        axStats.clear()

        if currentStatIndex[0] == 0:
            # Distribution of number of digits per image
            uniqueCounts, counts = np.unique(
                digitCounts, return_counts=True
            )
            rel = counts / counts.sum()

            axStats.bar(uniqueCounts, rel, edgecolor="black", alpha=0.8)
            axStats.set_title("Number of digits per image")
            axStats.set_xlabel("Digits per image")
            axStats.set_ylabel("Relative frequency")
            axStats.set_xticks(uniqueCounts)

        elif currentStatIndex[0] == 1:
            # Digit class distribution
            axStats.bar(range(10), classCounts, edgecolor="black", alpha=0.8)
            axStats.set_title("Digit class distribution")
            axStats.set_xlabel("Digit class")
            axStats.set_ylabel("Number of occurrences")
            axStats.set_xticks(range(10))

        elif currentStatIndex[0] == 2:
            # Bounding box size distribution
            counts, bins = np.histogram(
                boxSizes,
                bins=range(min(boxSizes), max(boxSizes) + 2)
            )
            rel = counts / counts.sum()
            axStats.bar(bins[:-1], rel, edgecolor="black", alpha=0.8)
            axStats.set_title("Bounding box size distribution")
            axStats.set_xlabel("Box size (pixels)")
            axStats.set_ylabel("Relative frequency")

        axStats.grid(True, linestyle="--", alpha=0.5)


    # ------------------------
    # INITIAL DRAW
    # ------------------------

    drawImage()
    drawStats()


    # ------------------------
    # BUTTONS
    # ------------------------

    btnPrevImg = Button(axPrevImg, "Previous image")
    btnNextImg = Button(axNextImg, "Next image")

    btnPrevStat = Button(axPrevStat, "Previous stats")
    btnNextStat = Button(axNextStat, "Next stats")

    def onPrevImage(event):
        # Show previous image
        currentImageIndex[0] = max(0, currentImageIndex[0] - 1)
        drawImage()
        fig.canvas.draw_idle()

    def onNextImage(event):
        # Show next image
        currentImageIndex[0] = min(
            len(images) - 1, currentImageIndex[0] + 1
        )
        drawImage()
        fig.canvas.draw_idle()

    def onPrevStat(event):
        # Show previous statistics plot
        currentStatIndex[0] = (currentStatIndex[0] - 1) % 3
        drawStats()
        fig.canvas.draw_idle()

    def onNextStat(event):
        # Show next statistics plot
        currentStatIndex[0] = (currentStatIndex[0] + 1) % 3
        drawStats()
        fig.canvas.draw_idle()

    btnNextImg.on_clicked(onNextImage)
    btnPrevImg.on_clicked(onPrevImage)
    
    btnNextStat.on_clicked(onNextStat)
    btnPrevStat.on_clicked(onPrevStat)

    plt.show()



# -------------------------------------------------
# STATISTICS Computation
# -------------------------------------------------

def computeStatistics(allAnnotations):
    # Compute dataset statistics
    digitCounts = []
    classCounts = np.zeros(10)
    boxSizes = []

    for annotations in allAnnotations:
        # Number of digits per image
        digitCounts.append(len(annotations))

        for digitLabel, x, y, w, h in annotations:
            # Count digit classes
            classCounts[digitLabel] += 1

            # Store bounding box size
            boxSizes.append(w)

    return digitCounts, classCounts, boxSizes


def setupPlot(title, xlabel, ylabel):
    # Configure plot appearance
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="MNIST Object Detection Dataset Statistics"
    )

    parser.add_argument(
        "--version",
        required=True,
        choices=["versionA", "versionB", "versionC", "versionD"],
        help="Dataset version to analyze"
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "test"],
        help="Dataset split to analyze"
    )

    args = parser.parse_args()

    # Build dataset paths
    baseDirectory = os.path.dirname(os.path.abspath(__file__))
    versionDirectory = os.path.join(
        baseDirectory, "..\\improved_dataset", args.version
    )

    imageFile = os.path.join(
        versionDirectory,
        f"{args.split}-images-ubyte.bin"
    )

    labelFile = os.path.join(
        versionDirectory,
        f"{args.split}-labels-ubyte.bin"
    )

    # Dataset parameters
    numberOfImages = 60000 if args.split == "train" else 10000
    imageSize = 128

    # Load images and annotations
    images = loadImagesUbyte(
        imageFile, numberOfImages, imageSize
    )

    annotations = loadLabelsUbyte(labelFile)

    # Compute dataset statistics
    digitCounts, classCounts, boxSizes = computeStatistics(annotations)

    # Run interactive visualization (subset of images)
    runInterface(
        images=images[:50],
        annotations=annotations[:50],
        digitCounts=digitCounts,
        classCounts=classCounts,
        boxSizes=boxSizes
    )


if __name__ == "__main__":
    main()
