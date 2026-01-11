import os
import struct
import random
import numpy as np
import cv2
import argparse
from torchvision import transforms, datasets


# -------------------------------------------------
# LOAD MNIST DATA
# -------------------------------------------------
def getMnistData(dataPath):
    # Convert images to tensors
    transform = transforms.ToTensor()

    # Load MNIST training dataset
    trainDataset = datasets.MNIST(
        root=dataPath,
        train=True,
        download=True,
        transform=transform
    )

    # Load MNIST test dataset
    testDataset = datasets.MNIST(
        root=dataPath,
        train=False,
        download=True,
        transform=transform
    )

    trainImages = []
    trainLabels = []

    # Convert training images to uint8 numpy arrays
    for image, label in trainDataset:
        # image is [1, 28, 28] tensor in [0,1]
        trainImages.append((image.numpy()[0] * 255).astype(np.uint8))
        trainLabels.append(label)

    testImages = []
    testLabels = []

    # Convert test images to uint8 numpy arrays
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
    # Check if two bounding boxes overlap
    # Each box is defined as (x, y, width, heigh
    x1, y1, w1, h1 = firstBox
    x2, y2, w2, h2 = secondBox

    # Return True if boxes overlap, False otherwise
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

    # Create output directory if it does not exis
    os.makedirs(outputDirectory, exist_ok=True)

    maximumPlacementAttempts = 100

    allImages = []
    allAnnotations = []

    # Generate images
    for imageIndex in range(numberOfImages):

        # Create empty canvas
        canvasImage = np.zeros(
            (outputImageSize, outputImageSize),
            dtype=np.uint8
        )

        boundingBoxes = []
        annotations = []

        # Random number of digits per image
        numberOfDigits = random.randint(minimumDigits, maximumDigits)


        # Place each digit
        for _ in range(numberOfDigits):

            for attempt in range(maximumPlacementAttempts):

                # Pick random MNIST digit
                randomIndex = random.randint(0, len(images) - 1)
                digitImage = images[randomIndex]
                digitLabel = labels[randomIndex]

                # Choose digit size
                digitSize = (
                    random.randint(minimumDigitSize, maximumDigitSize)
                    if allowScaleVariation
                    else minimumDigitSize
                )

                # Resize digit
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

                # Check for overlap with existing boxes
                if any(
                    boxesOverlap(candidateBox, box)
                    for box in boundingBoxes
                ):
                    continue
                
                # Place digit on canvas
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

                # Save bounding box and annotation
                boundingBoxes.append(candidateBox)
                annotations.append(
                    (digitLabel, candidateBox)
                )
                break

        allImages.append(canvasImage)
        allAnnotations.append(annotations)

         # Print progress
        if imageIndex % 1000 == 0:
            print(f"[{splitName}] {imageIndex}/{numberOfImages}")

    # -----------------------------------
    # SAVE FILES
    # -----------------------------------
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
    # Save images as raw uint8 binary file
    imagesArray = np.asarray(imagesArray, dtype=np.uint8)
    imagesArray.tofile(outputFilePath)


def saveLabelsAsUbyte(allAnnotations, outputFilePath):
    # Save labels and bounding boxes as binary file
    with open(outputFilePath, "wb") as file:
        for annotations in allAnnotations:
            
            # Write number of objects in image
            numberOfObjects = len(annotations)
            file.write(
                np.array([numberOfObjects], dtype=np.uint8).tobytes()
            )

            # Write each annotation
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

    # Parse command-line arguments
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


    # Define directories
    baseDirectory = os.path.dirname(os.path.abspath(__file__))
    outputDirectory = os.path.join(baseDirectory, "output")
    dataDirectory = os.path.join(baseDirectory, "data")

    # Load MNIST data
    (
        trainImages,
        trainLabels,
        testImages,
        testLabels
    ) = getMnistData(dataDirectory)


    outputImageSize = 128
    numberOfImagesTest = 10000
    numberOfImagesTrain = 60000

    # Dataset version configurations
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


    # Generate datasets for selected versions
    for versionKey in args.versions:

        config = versionConfigurations[versionKey]

        versionDirectory = os.path.join(
            outputDirectory, config["name"]
        )

        # Generate training split
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

        # Generate test split
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
