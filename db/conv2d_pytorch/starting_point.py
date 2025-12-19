from __future__ import annotations

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    """2D Convolution layer.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the square convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D convolution to the input.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        raise NotImplementedError

