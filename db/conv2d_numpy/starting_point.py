from __future__ import annotations

import numpy as np


def conv2d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Compute the forward pass of a 2D convolution layer.

    Args:
        x: Input tensor of shape (N, C_in, H, W).
        weight: Convolution kernels of shape (C_out, C_in, kH, kW).
        bias: Optional bias of shape (C_out,).
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of input.

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out).
    """
    raise NotImplementedError



