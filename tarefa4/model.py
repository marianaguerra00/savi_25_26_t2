import torch.nn as nn
import torch.nn.functional as F


class ModelImprovedDetector(nn.Module):
    def __init__(self, numClasses=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.detectorHead = nn.Conv2d(
            128,
            1 + numClasses + 4,
            kernel_size=1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detectorHead(x)
        return x
