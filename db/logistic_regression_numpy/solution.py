from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function element-wise.

    Args:
        z: Input array of any shape.

    Returns:
        Sigmoid values, same shape as input.
    """
    # sigmoid(z) = 1 / (1 + exp(-z))
    return 1.0 / (1.0 + np.exp(-z))


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
    n = X.shape[0]

    # Compute predictions: p = sigmoid(X @ w + b)
    z = X @ w + b
    p = sigmoid(z)

    # Clip predictions to avoid log(0)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(loss)


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
    n = X.shape[0]

    # Forward pass
    z = X @ w + b
    p = sigmoid(z)

    # Gradient of BCE loss w.r.t. z is (p - y)
    # dL/dw = (1/n) * X^T @ (p - y)
    # dL/db = (1/n) * sum(p - y)
    error = p - y
    dw = (1 / n) * (X.T @ error)
    db = float((1 / n) * np.sum(error))

    return dw, db


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
    n_features = X.shape[1]

    # Initialize weights and bias to zeros
    w = np.zeros(n_features)
    b = 0.0

    # Gradient descent loop
    for _ in range(n_iters):
        dw, db = compute_gradients(X, y, w, b)
        w = w - lr * dw
        b = b - lr * db

    return w, b
