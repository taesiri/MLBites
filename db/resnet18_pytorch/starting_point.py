from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """A basic residual block for ResNet-18/34.

    Architecture:
        x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> (+skip) -> ReLU -> out

    The skip connection either:
        - Directly adds input (if in_channels == out_channels and stride == 1)
        - Applies 1x1 conv + BN to match dimensions (otherwise)

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride for the first conv (default 1)

    Returns:
        forward(x) returns tensor of shape (batch, out_channels, H', W')
        where H' = H // stride, W' = W // stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # TODO: define conv1 (3x3, in_channels -> out_channels, stride)
        # TODO: define bn1
        # TODO: define conv2 (3x3, out_channels -> out_channels, stride=1)
        # TODO: define bn2
        # TODO: define relu
        # TODO: define downsample if needed (1x1 conv + bn for dimension matching)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BasicBlock.

        Args:
            x: input tensor of shape (batch, in_channels, H, W)

        Returns:
            output tensor of shape (batch, out_channels, H//stride, W//stride)
        """
        # TODO: save identity for skip connection
        # TODO: apply conv1 -> bn1 -> relu
        # TODO: apply conv2 -> bn2
        # TODO: apply downsample to identity if needed
        # TODO: add skip connection and final relu
        raise NotImplementedError


class ResNet18(nn.Module):
    """ResNet-18 for image classification.

    Architecture:
        - Initial: Conv(7x7, 64, stride=2) -> BN -> ReLU -> MaxPool(3x3, stride=2)
        - Layer1: 2 BasicBlocks with 64 filters
        - Layer2: 2 BasicBlocks with 128 filters (first stride=2)
        - Layer3: 2 BasicBlocks with 256 filters (first stride=2)
        - Layer4: 2 BasicBlocks with 512 filters (first stride=2)
        - Output: AdaptiveAvgPool -> Flatten -> FC(512 -> num_classes)

    Args:
        num_classes: number of output classes (default 1000)

    Returns:
        forward(x) returns logits of shape (batch_size, num_classes)
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        # TODO: define initial conv (7x7, 3->64, stride=2, padding=3)
        # TODO: define initial bn
        # TODO: define relu
        # TODO: define maxpool (3x3, stride=2, padding=1)
        # TODO: define layer1 (2 BasicBlocks, 64->64)
        # TODO: define layer2 (2 BasicBlocks, 64->128, first stride=2)
        # TODO: define layer3 (2 BasicBlocks, 128->256, first stride=2)
        # TODO: define layer4 (2 BasicBlocks, 256->512, first stride=2)
        # TODO: define adaptive avgpool
        # TODO: define fc (512 -> num_classes)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18.

        Args:
            x: input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            logits tensor of shape (batch_size, num_classes)
        """
        # TODO: apply initial conv -> bn -> relu -> maxpool
        # TODO: apply layer1, layer2, layer3, layer4
        # TODO: apply avgpool -> flatten -> fc
        raise NotImplementedError


