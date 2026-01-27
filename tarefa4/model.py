import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelImprovedDetector(nn.Module):
    """
    Fully Convolutional Network (FCN) for simultaneous detection and classification.
    
    Architecture improvements:
    - Deeper feature extraction (5 conv blocks instead of 3)
    - Batch normalization for training stability
    - Multi-scale features before detection head
    - Predicts: confidence (1) + class (10) + bbox offsets (4) = 15 channels
    """
    def __init__(self, numClasses=10):
        super().__init__()
        
        # Encoder: Extract features with increasing depth
        # Input: 1x128x128 -> Output: 256x32x32
        self.encoder = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 128 -> 256 (no pooling - keep 32x32)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Multi-scale feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Detection head: predicts for each grid cell
        # Output: 15 channels (1 conf + 10 classes + 4 bbox)
        self.detectorHead = nn.Conv2d(
            128,
            1 + numClasses + 4,
            kernel_size=1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, 128, 128] input images
        
        Returns:
            [B, 15, 32, 32] predictions:
                - channel 0: objectness confidence
                - channels 1-10: class logits
                - channels 11-14: bbox offsets (tx, ty, tw, th)
        """
        features = self.encoder(x)      # [B, 256, 32, 32]
        refined = self.refinement(features)  # [B, 128, 32, 32]
        output = self.detectorHead(refined)  # [B, 15, 32, 32]
        
        return output
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelImprovedDetectorLite(nn.Module):
    """
    LIGHTWEIGHT version with ~300K parameters
    Better for limited GPU memory and faster training
    
    Key differences:
    - Fewer channels (16->32->64->128 instead of 32->64->128->256)
    - Single conv per block
    - No refinement layer
    """
    def __init__(self, numClasses=10):
        super().__init__()
        
        # Lighter encoder
        self.encoder = nn.Sequential(
            # Block 1: 1 -> 16
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            
            # Block 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # Block 3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 4: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Direct detection head (no refinement)
        self.detectorHead = nn.Conv2d(
            128,
            1 + numClasses + 4,
            kernel_size=1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.encoder(x)        # [B, 128, 32, 32]
        output = self.detectorHead(features)  # [B, 15, 32, 32]
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelImprovedDetectorV2(nn.Module):
    """
    Alternative: ResNet-inspired architecture with skip connections
    More parameters, potentially better accuracy
    """
    def __init__(self, numClasses=10):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # 128 -> 32
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        
        # Detection head
        self.detectorHead = nn.Conv2d(
            256,
            1 + numClasses + 4,
            kernel_size=1
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)       # [B, 64, 32, 32]
        x = self.layer1(x)      # [B, 64, 32, 32]
        x = self.layer2(x)      # [B, 128, 32, 32]
        x = self.layer3(x)      # [B, 256, 32, 32]
        x = self.detectorHead(x)  # [B, 15, 32, 32]
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


if __name__ == "__main__":
    # Test models
    print("=== Testing ModelImprovedDetectorLite (RECOMMENDED) ===")
    model_lite = ModelImprovedDetectorLite(numClasses=10)
    x = torch.randn(2, 1, 128, 128)
    output_lite = model_lite(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_lite.shape}")
    print(f"Parameters: {model_lite.count_parameters():,}")
    
    print("\n=== Testing ModelImprovedDetector ===")
    model1 = ModelImprovedDetector(numClasses=10)
    output1 = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters: {model1.count_parameters():,}")
    
    print("\n=== Testing ModelImprovedDetectorV2 (ResNet-style) ===")
    model2 = ModelImprovedDetectorV2(numClasses=10)
    output2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters: {model2.count_parameters():,}")