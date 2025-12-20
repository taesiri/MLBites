from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConv2d(nn.Module):
    """Upsample-Convolution layer.

    Combines upsampling (via interpolation) with a standard convolution.
    This pattern is commonly used in decoder networks and generative models
    to avoid checkerboard artifacts that can occur with transposed convolutions.

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
        # Store upsampling configuration
        self.scale_factor = scale_factor
        self.mode = mode

        # Create the convolution layer
        # We use nn.Conv2d directly since the goal is understanding the pattern
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling followed by 2D convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        # Step 1: Upsample the input using interpolation
        # For bilinear mode, we need align_corners=False for standard behavior
        if self.mode == "bilinear":
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        else:
            # Nearest neighbor doesn't use align_corners
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        # Step 2: Apply convolution to the upsampled feature map
        x = self.conv(x)

        return x



