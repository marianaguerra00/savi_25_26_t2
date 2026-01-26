import numpy as np
import torch
from torch.utils.data import Dataset


def loadImagesFromUbyte(filePath, imageSize):
    data = np.fromfile(filePath, dtype=np.uint8)
    pixelsPerImage = imageSize * imageSize
    numberOfImages = data.size // pixelsPerImage
    data = data[:numberOfImages * pixelsPerImage]
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


class ImprovedDigitsDataset(Dataset):
    def __init__(self, imagesPath, labelsPath, imageSize=128, stride=4):
        self.images = loadImagesFromUbyte(imagesPath, imageSize)
        self.annotations = loadLabelsUbyte(labelsPath)

        self.imageSize = imageSize
        self.gridSize = imageSize // stride   # 128 // 4 = 32
        self.cellSize = imageSize / self.gridSize

        self.datasetLength = min(len(self.images), len(self.annotations))


    def __len__(self):
        return self.datasetLength

    def __getitem__(self, index):
        image = torch.tensor(
            self.images[index],
            dtype=torch.float32
        ) / 255.0

        image = image.unsqueeze(0)

        confTarget = torch.zeros(self.gridSize, self.gridSize)
        classTarget = torch.full(
            (self.gridSize, self.gridSize),
            -1,
            dtype=torch.long
        )
        bboxTarget = torch.zeros(4, self.gridSize, self.gridSize)

        for digitLabel, x, y, w, h in self.annotations[index]:
            cx = x + w / 2
            cy = y + h / 2

            gridX = int(cx / self.cellSize)
            gridY = int(cy / self.cellSize)

            if gridX >= self.gridSize or gridY >= self.gridSize:
                continue

            confTarget[gridY, gridX] = 1.0
            classTarget[gridY, gridX] = digitLabel

            bboxTarget[:, gridY, gridX] = torch.tensor([
                x / self.imageSize,
                y / self.imageSize,
                w / self.imageSize,
                h / self.imageSize
            ])

            if gridX >= self.gridSize or gridY >= self.gridSize:
                continue

            confTarget[gridY, gridX] = 1.0
            classTarget[gridY, gridX] = digitLabel

            bboxTarget[:, gridY, gridX] = torch.tensor([
                x / self.imageSize,
                y / self.imageSize,
                w / self.imageSize,
                h / self.imageSize
            ])

        return image, confTarget, classTarget, bboxTarget


def getImprovedDatasets(basePath="../improved_dataset/versionD"):
    trainDataset = ImprovedDigitsDataset(
        basePath + "/train-images-ubyte.bin",
        basePath + "/train-labels-ubyte.bin"
    )

    testDataset = ImprovedDigitsDataset(
        basePath + "/test-images-ubyte.bin",
        basePath + "/test-labels-ubyte.bin"
    )

    return trainDataset, testDataset
