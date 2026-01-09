import os
import struct
import random
import numpy as np
import cv2
import argparse
from torchvision import transforms, datasets


def getMnistData(dataPath):
    transform = transforms.ToTensor()

    trainDataset = datasets.MNIST(
        root=dataPath,
        train=True,
        download=True,
        transform=transform
    )

    testDataset = datasets.MNIST(
        root=dataPath,
        train=False,
        download=True,
        transform=transform
    )

    trainImages = []
    trainLabels = []

    for image, label in trainDataset:
        # image is [1, 28, 28] tensor in [0,1]
        trainImages.append((image.numpy()[0] * 255).astype(np.uint8))
        trainLabels.append(label)

    testImages = []
    testLabels = []

    for image, label in testDataset:
        testImages.append((image.numpy()[0] * 255).astype(np.uint8))
        testLabels.append(label)

    return (
        np.array(trainImages),
        np.array(trainLabels),
        np.array(testImages),
        np.array(testLabels)
    )





# ------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------
def boxesOverlap(firstBox, secondBox):
    """
    Check if two bounding boxes overlap.
    Each box is (x, y, width, height).
    """
    x1, y1, w1, h1 = firstBox
    x2, y2, w2, h2 = secondBox

    return not (
        x1 + w1 <= x2 or
        x2 + w2 <= x1 or
        y1 + h1 <= y2 or
        y2 + h2 <= y1
    )


# -----------------------------------
# DATASET GENERATION
# ----------------------------------
def generateSplit(
    images,
    labels,
    outputDirectory,
    splitName,
    numberOfImages,
    outputImageSize,
    minimumDigits,
    maximumDigits,
    minimumDigitSize,
    maximumDigitSize,
    allowScaleVariation
):

    os.makedirs(outputDirectory, exist_ok=True)

    maximumPlacementAttempts = 100

    allImages = []
    allAnnotations = []

    for imageIndex in range(numberOfImages):

        canvasImage = np.zeros(
            (outputImageSize, outputImageSize),
            dtype=np.uint8
        )

        boundingBoxes = []
        annotations = []

        numberOfDigits = random.randint(minimumDigits, maximumDigits)

        for _ in range(numberOfDigits):

            for attempt in range(maximumPlacementAttempts):

                randomIndex = random.randint(0, len(images) - 1)
                digitImage = images[randomIndex]
                digitLabel = labels[randomIndex]

                digitSize = (
                    random.randint(minimumDigitSize, maximumDigitSize)
                    if allowScaleVariation
                    else minimumDigitSize
                )

                digitImage = cv2.resize(
                    digitImage, (digitSize, digitSize)
                )

                positionX = random.randint(
                    0, outputImageSize - digitSize
                )
                positionY = random.randint(
                    0, outputImageSize - digitSize
                )

                candidateBox = (
                    positionX, positionY, digitSize, digitSize
                )

                if any(
                    boxesOverlap(candidateBox, box)
                    for box in boundingBoxes
                ):
                    continue

                canvasImage[
                    positionY:positionY + digitSize,
                    positionX:positionX + digitSize
                ] = np.maximum(
                    canvasImage[
                        positionY:positionY + digitSize,
                        positionX:positionX + digitSize
                    ],
                    digitImage
                )

                boundingBoxes.append(candidateBox)
                annotations.append(
                    (digitLabel, candidateBox)
                )
                break

        allImages.append(canvasImage)
        allAnnotations.append(annotations)

        if imageIndex % 1000 == 0:
            print(f"[{splitName}] {imageIndex}/{numberOfImages}")

    # ---- SAVE 2 FILES ONLY ----

    saveImagesAsUbyte(
        np.stack(allImages),
        os.path.join(
            outputDirectory, f"{splitName}-images-ubyte.bin"
        )
    )

    saveLabelsAsUbyte(
        allAnnotations,
        os.path.join(
            outputDirectory, f"{splitName}-labels-ubyte.bin"
        )
    )


# -----------------------------------
# Save as Ubyte
# ----------------------------------

def saveImagesAsUbyte(imagesArray, outputFilePath):
    imagesArray = np.asarray(imagesArray, dtype=np.uint8)
    imagesArray.tofile(outputFilePath)


def saveLabelsAsUbyte(allAnnotations, outputFilePath):
    with open(outputFilePath, "wb") as file:
        for annotations in allAnnotations:

            numberOfObjects = len(annotations)
            file.write(
                np.array([numberOfObjects], dtype=np.uint8).tobytes()
            )

            for digitLabel, (x, y, w, h) in annotations:
                file.write(
                    np.array([digitLabel], dtype=np.uint8).tobytes()
                )
                file.write(
                    np.array([x, y, w, h], dtype=np.uint16).tobytes()
                )


# -----------------
# MAIN
# ------------------

def main():

    parser = argparse.ArgumentParser(
        description="MNIST Object Detection Dataset Generator"
    )

    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["B", "D"],
        help="Dataset versions to generate (A, B, C, D)"
    )

    args = parser.parse_args()

    baseDirectory = os.path.dirname(os.path.abspath(__file__))
    rawMnistDirectory = os.path.join(baseDirectory, "rawMNIST")
    outputDirectory = os.path.join(baseDirectory, "output")

    dataDirectory = os.path.join(baseDirectory, "data")

    (
        trainImages,
        trainLabels,
        testImages,
        testLabels
    ) = getMnistData(dataDirectory)


    outputImageSize = 128
    numberOfImagesTest = 10000
    numberOfImagesTrain = 60000

    versionConfigurations = {
    "A": {
        "name": "versionA",
        "minimumDigits": 1,
        "maximumDigits": 1,
        "minimumDigitSize": 28,
        "maximumDigitSize": 28,
        "allowScaleVariation": False
    },
    "B": {
        "name": "versionB",
        "minimumDigits": 1,
        "maximumDigits": 1,
        "minimumDigitSize": 22,
        "maximumDigitSize": 36,
        "allowScaleVariation": True
    },
    "C": {
        "name": "versionC",
        "minimumDigits": 3,
        "maximumDigits": 5,
        "minimumDigitSize": 28,
        "maximumDigitSize": 28,
        "allowScaleVariation": False
    },
    "D": {
        "name": "versionD",
        "minimumDigits": 3,
        "maximumDigits": 5,
        "minimumDigitSize": 22,
        "maximumDigitSize": 36,
        "allowScaleVariation": True
    }
}

    for versionKey in args.versions:

        config = versionConfigurations[versionKey]

        versionDirectory = os.path.join(
            outputDirectory, config["name"]
        )

        generateSplit(
            images=trainImages,
            labels=trainLabels,
            outputDirectory=versionDirectory,
            splitName="train",
            numberOfImages=numberOfImagesTrain,
            outputImageSize=outputImageSize,
            minimumDigits=config["minimumDigits"],
            maximumDigits=config["maximumDigits"],
            minimumDigitSize=config["minimumDigitSize"],
            maximumDigitSize=config["maximumDigitSize"],
            allowScaleVariation=config["allowScaleVariation"]
        )

        generateSplit(
            images=testImages,
            labels=testLabels,
            outputDirectory=versionDirectory,
            splitName="test",
            numberOfImages=numberOfImagesTest,
            outputImageSize=outputImageSize,
            minimumDigits=config["minimumDigits"],
            maximumDigits=config["maximumDigits"],
            minimumDigitSize=config["minimumDigitSize"],
            maximumDigitSize=config["maximumDigitSize"],
            allowScaleVariation=config["allowScaleVariation"]
        )



if __name__ == "__main__":
    main()
