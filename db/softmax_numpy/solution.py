from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically-stable softmax over a given axis.

    Args:
        x: Input logits as a NumPy array of any shape.
        axis: Axis along which to compute softmax (default: last axis).

    Returns:
        A NumPy array of the same shape as `x`, where values sum to 1 along `axis`.
    """
    # Subtracting the max improves numerical stability: exp(x - max(x)) avoids overflow.
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp_shifted = np.exp(shifted)
    denom = np.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / denom


