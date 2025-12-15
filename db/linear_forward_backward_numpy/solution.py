from __future__ import annotations

import numpy as np


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Linear (affine) layer forward pass.

    Args:
        x: (N, D)
        W: (D, M)
        b: (M,)

    Returns:
        y: (N, M)
        cache: tuple of values needed for backward
    """
    y = x @ W + b
    cache = (x, W, b)
    return y, cache


def linear_backward(dy: np.ndarray, cache):
    """Linear (affine) layer backward pass.

    Args:
        dy: (N, M) upstream gradient
        cache: from linear_forward

    Returns:
        dx: (N, D)
        dW: (D, M)
        db: (M,)
    """
    x, W, _b = cache
    dx = dy @ W.T
    dW = x.T @ dy
    db = np.sum(dy, axis=0)
    return dx, dW, db


