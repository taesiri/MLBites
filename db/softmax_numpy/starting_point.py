from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically-stable softmax over a given axis.

    Softmax converts logits/scores into probabilities that sum to 1 along `axis`.

    Args:
        x: Input logits as a NumPy array of any shape.
        axis: Axis along which to compute softmax (default: last axis).

    Returns:
        A NumPy array of the same shape as `x`, where values sum to 1 along `axis`.
    """
    # TODO: implement a numerically-stable softmax:
    # - subtract the max along `axis` before exponentiating
    # - use keepdims=True to make broadcasting work
    raise NotImplementedError


