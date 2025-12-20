from __future__ import annotations

from typing import Any

import numpy as np


def batchnorm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the forward pass of batch normalization.

    Args:
        x: Input tensor of shape (N, D).
        gamma: Scale parameter of shape (D,).
        beta: Shift parameter of shape (D,).
        eps: Small constant for numerical stability.

    Returns:
        out: Normalized output of shape (N, D).
        cache: Dictionary containing values needed for backward pass.
    """
    raise NotImplementedError


def batchnorm_backward(
    dout: np.ndarray,
    cache: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the backward pass of batch normalization.

    Args:
        dout: Upstream gradient of shape (N, D).
        cache: Dictionary from forward pass.

    Returns:
        dx: Gradient with respect to input x, shape (N, D).
        dgamma: Gradient with respect to gamma, shape (D,).
        dbeta: Gradient with respect to beta, shape (D,).
    """
    raise NotImplementedError



