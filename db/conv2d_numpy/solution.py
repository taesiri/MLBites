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
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    # Compute output dimensions
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # Apply zero-padding to input (only on H and W dimensions)
    if padding > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        x_pad = x

    # Initialize output
    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    # Perform convolution
    for h in range(H_out):
        for w in range(W_out):
            # Extract the window from padded input
            h_start = h * stride
            w_start = w * stride
            window = x_pad[:, :, h_start:h_start + kH, w_start:w_start + kW]

            # Compute output for all samples and all output channels
            # window: (N, C_in, kH, kW)
            # weight: (C_out, C_in, kH, kW)
            # We want: (N, C_out) at position (h, w)
            for c_out in range(C_out):
                # Sum over C_in, kH, kW
                out[:, c_out, h, w] = np.sum(window * weight[c_out], axis=(1, 2, 3))

    # Add bias if provided
    if bias is not None:
        # Reshape bias to (1, C_out, 1, 1) for broadcasting
        out += bias.reshape(1, -1, 1, 1)

    return out



