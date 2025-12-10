"""
U-Net for Segmentation - Starting Point

Implement U-Net from scratch.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # TODO: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class UNet(nn.Module):
    """U-Net for image segmentation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 2, features: list = None):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            features: Channel sizes for each encoder level [64, 128, 256, 512]
        """
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        
        # TODO: Build encoder (contracting path)
        # Each level: DoubleConv -> MaxPool
        # Save intermediate features for skip connections
        
        # TODO: Bottleneck
        
        # TODO: Build decoder (expansive path)
        # Each level: UpConv -> Concat(skip) -> DoubleConv
        
        # TODO: Final 1x1 conv for classification
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, in_channels, H, W)
        Returns:
            Segmentation logits (B, out_channels, H, W)
        """
        # TODO: Encoder path (save skip connections)
        
        # TODO: Bottleneck
        
        # TODO: Decoder path (use skip connections)
        
        # TODO: Final classification
        
        pass


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth (B, H, W) or one-hot (B, C, H, W)
        """
        # TODO: Compute Dice loss
        # Dice = 2 * |A ∩ B| / (|A| + |B|)
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create U-Net
    unet = UNet(in_channels=1, out_channels=2, features=[64, 128, 256, 512])
    
    # Test forward
    x = torch.randn(2, 1, 256, 256)
    out = unet(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")
