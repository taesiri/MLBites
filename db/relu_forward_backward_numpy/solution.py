from __future__ import annotations

import numpy as np


def relu_forward(x: np.ndarray):
    """ReLU forward pass.

    Args:
        x: np.ndarray of any shape

    Returns:
        y: np.ndarray of same shape as x, where y = max(0, x)
        cache: values needed for backward pass
    """
    y = np.maximum(0, x)
    # Cache a boolean mask; derivative at x == 0 is defined as 0 via (x > 0).
    cache = x > 0
    return y, cache


def relu_backward(dy: np.ndarray, cache) -> np.ndarray:
    """ReLU backward pass.

    Args:
        dy: np.ndarray of same shape as x (upstream gradient)
        cache: from relu_forward (boolean mask where x > 0)

    Returns:
        dx: np.ndarray of same shape as x
    """
    mask = np.asarray(cache, dtype=bool)
    if dy.shape != mask.shape:
        raise ValueError(f"dy must have shape {mask.shape}, got {dy.shape}")
    dx = dy * mask
    return dx


