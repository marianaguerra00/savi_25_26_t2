import numpy as np
import torch
from torch.utils.data import Dataset

def loadImagesFromUbyte(filePath, imageSize):
    """Load images from binary file"""
    data = np.fromfile(filePath, dtype=np.uint8)
    pixelsPerImage = imageSize * imageSize
    numberOfImages = data.size // pixelsPerImage
    data = data[:numberOfImages * pixelsPerImage]
    return data.reshape(numberOfImages, imageSize, imageSize)

def loadLabelsUbyte(filePath):
    """Load annotations from binary file"""
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
    """
    Dataset for digit detection and classification.
    
    Encoding strategy (YOLO-style):
    - Grid: 32x32 cells (stride=4 from 128x128 image)
    - Each cell predicts:
        * Confidence: 1 if object center is in cell, 0 otherwise
        * Class: digit label (0-9)
        * Bbox offsets: (tx, ty, tw, th) where:
            - tx, ty: offset of object center within cell (0-1)
            - tw, th: width/height normalized by image size (0-1)
    """
    def __init__(self, imagesPath, labelsPath, imageSize=128, stride=4):
        self.images = loadImagesFromUbyte(imagesPath, imageSize)
        self.annotations = loadLabelsUbyte(labelsPath)
        self.imageSize = imageSize
        self.gridSize = imageSize // stride   # 128 // 4 = 32
        self.cellSize = imageSize / self.gridSize  # 4 pixels per cell
        self.datasetLength = min(len(self.images), len(self.annotations))
    
    def __len__(self):
        return self.datasetLength
    
    def __getitem__(self, index):
        # Load and normalize image
        image = torch.tensor(
            self.images[index],
            dtype=torch.float32
        ) / 255.0
        image = image.unsqueeze(0)  # [1, 128, 128]
        
        # Initialize targets
        confTarget = torch.zeros(self.gridSize, self.gridSize)
        classTarget = torch.full(
            (self.gridSize, self.gridSize),
            -1,
            dtype=torch.long
        )
        bboxTarget = torch.zeros(4, self.gridSize, self.gridSize)
        
        # Encode annotations into grid
        for digitLabel, x, y, w, h in self.annotations[index]:
            # Calculate center of bounding box
            cx = x + w / 2.0
            cy = y + h / 2.0
            
            # Find which grid cell the center falls into
            gridX = int(cx / self.cellSize)
            gridY = int(cy / self.cellSize)
            
            # Skip if outside grid (shouldn't happen with valid data)
            if gridX >= self.gridSize or gridY >= self.gridSize:
                continue
            
            # Mark this cell as containing an object
            confTarget[gridY, gridX] = 1.0
            classTarget[gridY, gridX] = digitLabel
            
            # Calculate offset within the cell (YOLO-style)
            # tx, ty: how far into the cell the center is (0-1)
            cell_left = gridX * self.cellSize
            cell_top = gridY * self.cellSize
            tx = (cx - cell_left) / self.cellSize  # 0-1
            ty = (cy - cell_top) / self.cellSize   # 0-1
            
            # tw, th: width and height normalized by image size
            tw = w / self.imageSize  # 0-1
            th = h / self.imageSize  # 0-1
            
            bboxTarget[:, gridY, gridX] = torch.tensor([tx, ty, tw, th])
        
        return image, confTarget, classTarget, bboxTarget


def getImprovedDatasets(basePath="../improved_dataset/versionD"):
    """Load training and test datasets"""
    trainDataset = ImprovedDigitsDataset(
        basePath + "/train-images-ubyte.bin",
        basePath + "/train-labels-ubyte.bin"
    )
    testDataset = ImprovedDigitsDataset(
        basePath + "/test-images-ubyte.bin",
        basePath + "/test-labels-ubyte.bin"
    )
    return trainDataset, testDataset


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset...")
    trainDataset, testDataset = getImprovedDatasets()
    
    print(f"Train size: {len(trainDataset)}")
    print(f"Test size: {len(testDataset)}")
    
    # Test one sample
    image, conf, cls, bbox = trainDataset[0]
    print(f"\nSample shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Confidence: {conf.shape}")
    print(f"  Class: {cls.shape}")
    print(f"  BBox: {bbox.shape}")
    
    # Count objects in first sample
    num_objects = (conf > 0.5).sum()
    print(f"\nObjects in first sample: {num_objects}")
    
    if num_objects > 0:
        obj_cells = (conf > 0.5).nonzero(as_tuple=True)
        for i in range(num_objects):
            cy, cx = obj_cells[0][i].item(), obj_cells[1][i].item()
            label = cls[cy, cx].item()
            tx, ty, tw, th = bbox[:, cy, cx].tolist()
            print(f"  Cell ({cy},{cx}): class={label}, offsets=({tx:.3f}, {ty:.3f}, {tw:.3f}, {th:.3f})")