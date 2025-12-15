from __future__ import annotations

import numpy as np


def relu_forward(x):
    """ReLU forward pass.

    Computes y = max(0, x) elementwise.

    Args:
        x: np.ndarray of any shape

    Returns:
        y: np.ndarray of same shape as x
        cache: values needed for backward pass
    """
    # TODO: implement (likely y = np.maximum(0, x) and cache = something)
    raise NotImplementedError


def relu_backward(dy, cache):
    """ReLU backward pass.

    For this question, define the derivative at exactly x == 0 as 0
    (i.e., use the mask x > 0).

    Args:
        dy: np.ndarray of same shape as x (upstream gradient)
        cache: from relu_forward

    Returns:
        dx: np.ndarray of same shape as x
    """
    # TODO: implement
    raise NotImplementedError


