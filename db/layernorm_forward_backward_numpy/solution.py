from __future__ import annotations

import numpy as np


def layernorm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
):
    """LayerNorm forward pass.

    Normalizes across the feature dimension D for each sample (row) independently:
        x_hat = (x - mean) / sqrt(var + eps)
        y = x_hat * gamma + beta

    Args:
        x: (N, D)
        gamma: (D,) or (1, D)
        beta: (D,) or (1, D)
        eps: numerical stability constant

    Returns:
        y: (N, D)
        cache: values needed for backward pass
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N,D), got shape={x.shape}")
    n, d = x.shape
    if d == 0:
        raise ValueError("x must have D>0 features")

    gamma_in_shape = gamma.shape
    beta_in_shape = beta.shape

    gamma_2d = gamma.reshape(1, d) if gamma.ndim == 1 else gamma
    beta_2d = beta.reshape(1, d) if beta.ndim == 1 else beta

    if gamma_2d.shape != (1, d):
        raise ValueError(f"gamma must be (D,) or (1,D); got shape={gamma.shape}")
    if beta_2d.shape != (1, d):
        raise ValueError(f"beta must be (D,) or (1,D); got shape={beta.shape}")

    mu = np.mean(x, axis=1, keepdims=True)  # (N,1)
    var = np.var(x, axis=1, keepdims=True)  # (N,1)
    inv_std = 1.0 / np.sqrt(var + eps)  # (N,1)
    x_hat = (x - mu) * inv_std  # (N,D)

    y = x_hat * gamma_2d + beta_2d
    cache = (x_hat, inv_std, gamma_2d, gamma_in_shape, beta_in_shape)
    return y, cache


def layernorm_backward(dy: np.ndarray, cache):
    """LayerNorm backward pass.

    Args:
        dy: (N, D) upstream gradient
        cache: from layernorm_forward

    Returns:
        dx: (N, D)
        dgamma: same shape as gamma input to forward
        dbeta: same shape as beta input to forward
    """
    x_hat, inv_std, gamma_2d, gamma_in_shape, beta_in_shape = cache

    if dy.shape != x_hat.shape:
        raise ValueError(f"dy must have shape {x_hat.shape}, got {dy.shape}")

    n, d = dy.shape

    dbeta_2d = np.sum(dy, axis=0, keepdims=True)  # (1,D)
    dgamma_2d = np.sum(dy * x_hat, axis=0, keepdims=True)  # (1,D)

    dxhat = dy * gamma_2d  # (N,D)
    sum_dxhat = np.sum(dxhat, axis=1, keepdims=True)  # (N,1)
    sum_dxhat_xhat = np.sum(dxhat * x_hat, axis=1, keepdims=True)  # (N,1)

    dx = (inv_std / d) * (d * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)

    dgamma = (
        dgamma_2d.reshape(gamma_in_shape) if len(gamma_in_shape) == 1 else dgamma_2d
    )
    dbeta = dbeta_2d.reshape(beta_in_shape) if len(beta_in_shape) == 1 else dbeta_2d
    return dx, dgamma, dbeta
