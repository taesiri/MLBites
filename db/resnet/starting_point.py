"""
ResNet from Scratch - Starting Point

Implement ResNet from scratch.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        
        # TODO: First conv: 3x3, stride, padding=1
        # TODO: BatchNorm
        # TODO: ReLU
        # TODO: Second conv: 3x3, stride=1, padding=1
        # TODO: BatchNorm
        
        # TODO: Downsample shortcut if needed
        self.downsample = downsample
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # TODO: Main path
        
        # TODO: Shortcut (downsample if needed)
        
        # TODO: Add and ReLU
        
        pass


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        
        # TODO: 1x1 conv (reduce)
        # TODO: 3x3 conv
        # TODO: 1x1 conv (expand by factor of 4)
        
        self.downsample = downsample
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ResNet(nn.Module):
    """ResNet architecture."""
    
    def __init__(self, block, layers: list[int], num_classes: int = 1000):
        """
        Args:
            block: BasicBlock or Bottleneck
            layers: Number of blocks in each layer [2, 2, 2, 2] for ResNet-18
            num_classes: Number of output classes
        """
        super().__init__()
        self.in_channels = 64
        
        # TODO: Stem: Conv7x7, BN, ReLU, MaxPool
        
        # TODO: Residual layers
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # TODO: Classification head
        
        pass
    
    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int = 1):
        """Create a residual layer with multiple blocks."""
        # TODO: Create downsample if needed
        # TODO: First block may have stride > 1
        # TODO: Remaining blocks have stride = 1
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Stem
        # TODO: Layers 1-4
        # TODO: Pool and classify
        pass


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test ResNet-18
    model = resnet18(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
