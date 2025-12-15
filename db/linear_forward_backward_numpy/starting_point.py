from __future__ import annotations

import numpy as np


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Linear (affine) layer forward pass.

    Computes y = x @ W + b for a batch of inputs.

    Args:
        x: (N, D) input
        W: (D, M) weights
        b: (M,) bias

    Returns:
        y: (N, M) output
        cache: any values needed for backward (e.g., x, W, b)
    """
    # TODO: implement
    raise NotImplementedError


def linear_backward(dy: np.ndarray, cache):
    """Linear (affine) layer backward pass.

    Args:
        dy: (N, M) upstream gradient
        cache: cache returned by linear_forward

    Returns:
        dx: (N, D) gradient w.r.t. x
        dW: (D, M) gradient w.r.t. W
        db: (M,) gradient w.r.t. b
    """
    # TODO: implement
    raise NotImplementedError


