from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    """2D Convolution layer.

    Applies a 2D convolution over an input signal composed of several input planes.

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weight with shape (out_channels, in_channels, kernel_size, kernel_size)
        # Using Kaiming uniform initialization (same as PyTorch default)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )

        # Initialize bias with shape (out_channels,)
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Kaiming uniform initialization for weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Uniform initialization for bias (matching PyTorch Conv2d)
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D convolution to the input.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        batch_size, _, in_height, in_width = x.shape

        # Apply zero padding if needed
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Get padded dimensions
        _, _, padded_height, padded_width = x.shape

        # Compute output dimensions
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1

        # Initialize output tensor
        output = torch.zeros(
            batch_size, self.out_channels, out_height, out_width,
            dtype=x.dtype, device=x.device
        )

        # Perform convolution by sliding the kernel over the input
        for h in range(out_height):
            for w in range(out_width):
                # Compute the top-left corner of the current receptive field
                h_start = h * self.stride
                w_start = w * self.stride

                # Extract the patch: (batch, in_channels, kernel_size, kernel_size)
                patch = x[
                    :,
                    :,
                    h_start : h_start + self.kernel_size,
                    w_start : w_start + self.kernel_size,
                ]

                # Compute output for all filters at this position
                # patch: (batch, in_channels, kh, kw)
                # weight: (out_channels, in_channels, kh, kw)
                # We need: (batch, out_channels)
                # Use einsum: sum over in_channels, kh, kw
                output[:, :, h, w] = torch.einsum(
                    "bchw,ochw->bo", patch, self.weight
                ) + self.bias

        return output



