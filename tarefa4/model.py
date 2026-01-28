import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelImprovedDetector(nn.Module):
    """
    Lightweight FCN detector for digit detection and classification.
    
    Architecture:
    - Encoder: 4 blocks (16->32->64->128 channels)
    - FPN: Feature Pyramid Network with two scales (P3: 32x32, P4: 16x16)
    - Heads: Separate detection heads for each pyramid level
    
    Output per scale:
    - Channel 0: objectness confidence
    - Channels 1-10: class logits
    - Channels 11-14: bbox offsets (tx, ty, tw, th)
    """
    def __init__(self, numClasses=10):
        super().__init__()
        
        # Encoder blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 -> 64
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64 -> 32
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # FPN lateral connections
        self.lateralC3 = nn.Conv2d(32, 64, 1)    # C3 -> P3
        self.lateralC4 = nn.Conv2d(128, 64, 1)   # C4 -> P4
        
        # Detection heads for each pyramid level
        self.headP3 = nn.Conv2d(64, 1 + numClasses + 4, 1)
        self.headP4 = nn.Conv2d(64, 1 + numClasses + 4, 1)
        
        self._initWeights()
    
    def _initWeights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through encoder and FPN.
        
        Args:
            x: [B, 1, 128, 128] input images
        
        Returns:
            outP3: [B, 15, 32, 32] predictions at scale 1
            outP4: [B, 15, 16, 16] predictions at scale 2
        """
        # Encoder forward pass
        x = self.block1(x)   # [B, 16, 64, 64]
        c3 = self.block2(x)  # [B, 32, 32, 32]
        x = self.block3(c3)  # [B, 64, 32, 32]
        c4 = self.block4(x)  # [B, 128, 32, 32]
        
        # Downsample C4 for pyramid
        c4Down = F.max_pool2d(c4, 2)  # [B, 128, 16, 16]
        
        # FPN top-down pathway
        p4 = self.lateralC4(c4Down)  # [B, 64, 16, 16]
        p3 = self.lateralC3(c3) + F.interpolate(
            p4, scale_factor=2, mode="nearest"
        )  # [B, 64, 32, 32]
        
        # Detection heads
        outP3 = self.headP3(p3)  # [B, 15, 32, 32]
        outP4 = self.headP4(p4)  # [B, 15, 16, 16]
        
        # Apply sigmoid to bbox width/height (tw, th) for stability
        # This ensures w,h stay in [0,1] range
        outP3 = self._applyBboxConstraints(outP3)
        outP4 = self._applyBboxConstraints(outP4)
        
        return outP3, outP4
    
    def _applyBboxConstraints(self, output):
        """
        Apply sigmoid to bbox width/height channels for stability.
        
        Args:
            output: [B, 15, H, W] raw predictions
        
        Returns:
            output: [B, 15, H, W] with constrained bbox dimensions
        """
        # Split channels
        conf = output[:, 0:1]           # [B, 1, H, W]
        classes = output[:, 1:11]       # [B, 10, H, W]
        bboxOffsets = output[:, 11:13]  # [B, 2, H, W] - tx, ty
        bboxSizes = output[:, 13:15]    # [B, 2, H, W] - tw, th
        
        # Apply sigmoid to width/height and clamp to minimum value
        bboxSizes = torch.sigmoid(bboxSizes)
        bboxSizes = torch.clamp(bboxSizes, min=0.08)  # Prevent boxes from being too small
        
        # Concatenate back
        output = torch.cat([conf, classes, bboxOffsets, bboxSizes], dim=1)
        
        return output
    
    def countParameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing ModelImprovedDetector with FPN...")
    
    model = ModelImprovedDetector(numClasses=10)
    x = torch.randn(2, 1, 128, 128)
    
    outP3, outP4 = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"P3 output shape: {outP3.shape}")
    print(f"P4 output shape: {outP4.shape}")
    print(f"Total parameters: {model.countParameters():,}")
    
    # Verify bbox constraints are applied
    print(f"\nP3 bbox width/height range: [{outP3[:, 13:15].min():.4f}, {outP3[:, 13:15].max():.4f}]")
    print(f"P4 bbox width/height range: [{outP4[:, 13:15].min():.4f}, {outP4[:, 13:15].max():.4f}]")
    print("\n✓ Model test passed!")