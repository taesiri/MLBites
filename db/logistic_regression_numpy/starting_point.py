from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function element-wise.

    Args:
        z: Input array of any shape.

    Returns:
        Sigmoid values, same shape as input.
    """
    raise NotImplementedError


def compute_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """Compute the binary cross-entropy loss.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary labels of shape (n_samples,).
        w: Weight vector of shape (n_features,).
        b: Bias term (scalar).

    Returns:
        Mean binary cross-entropy loss.
    """
    raise NotImplementedError


def compute_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """Compute gradients of the loss with respect to weights and bias.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary labels of shape (n_samples,).
        w: Weight vector of shape (n_features,).
        b: Bias term (scalar).

    Returns:
        Tuple (dw, db) where dw has shape (n_features,) and db is a scalar.
    """
    raise NotImplementedError


def train(
    X: np.ndarray, y: np.ndarray, lr: float, n_iters: int
) -> tuple[np.ndarray, float]:
    """Train logistic regression using gradient descent.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary labels of shape (n_samples,).
        lr: Learning rate.
        n_iters: Number of gradient descent iterations.

    Returns:
        Tuple (w, b) of trained weights and bias.
    """
    raise NotImplementedError
