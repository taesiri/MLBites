from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """A basic residual block for ResNet-18/34."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # TODO: implement BasicBlock
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BasicBlock."""
        # TODO: implement forward pass
        raise NotImplementedError


class ResNet18(nn.Module):
    """ResNet-18 for image classification."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        # TODO: implement ResNet-18 architecture
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18."""
        # TODO: implement forward pass
        raise NotImplementedError




