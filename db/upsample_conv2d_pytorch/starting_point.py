from __future__ import annotations

import torch
import torch.nn as nn


class UpsampleConv2d(nn.Module):
    """Upsample-Convolution layer.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the square convolving kernel.
        scale_factor: Upsampling scale factor.
        mode: Interpolation mode for upsampling ("nearest" or "bilinear").
        padding: Zero-padding added to both sides after upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: int = 2,
        mode: str = "nearest",
        padding: int = 0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling followed by 2D convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        raise NotImplementedError



