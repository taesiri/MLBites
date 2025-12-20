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
    N = x.shape[0]

    # Step 1: Compute batch mean
    mu = np.mean(x, axis=0)

    # Step 2: Compute batch variance
    var = np.var(x, axis=0)

    # Step 3: Compute standard deviation (with eps for stability)
    std = np.sqrt(var + eps)

    # Step 4: Normalize
    x_norm = (x - mu) / std

    # Step 5: Scale and shift
    out = gamma * x_norm + beta

    # Cache values needed for backward pass
    cache = {
        "x": x,
        "x_norm": x_norm,
        "mu": mu,
        "var": var,
        "std": std,
        "gamma": gamma,
        "eps": eps,
        "N": N,
    }

    return out, cache


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
    x = cache["x"]
    x_norm = cache["x_norm"]
    mu = cache["mu"]
    std = cache["std"]
    gamma = cache["gamma"]
    eps = cache["eps"]
    N = cache["N"]

    # Gradient w.r.t. beta: dbeta = sum(dout) over batch
    dbeta = np.sum(dout, axis=0)

    # Gradient w.r.t. gamma: dgamma = sum(dout * x_norm) over batch
    dgamma = np.sum(dout * x_norm, axis=0)

    # Gradient w.r.t. x_norm
    dx_norm = dout * gamma

    # Gradient w.r.t. variance
    # d(x_norm)/d(var) = (x - mu) * (-0.5) * (var + eps)^(-3/2)
    dvar = np.sum(dx_norm * (x - mu) * (-0.5) * (std ** -3), axis=0)

    # Gradient w.r.t. mean
    # d(x_norm)/d(mu) = -1/std + dvar * (-2/N) * sum(x - mu)
    # Note: sum(x - mu) = 0, so the second term vanishes
    dmu = np.sum(dx_norm * (-1.0 / std), axis=0)

    # Gradient w.r.t. x
    # dx = dx_norm * (1/std) + dvar * (2/N) * (x - mu) + dmu * (1/N)
    dx = dx_norm / std + dvar * (2.0 / N) * (x - mu) + dmu / N

    return dx, dgamma, dbeta



