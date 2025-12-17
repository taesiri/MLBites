from __future__ import annotations

import numpy as np


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit linear regression weights using the normal equation.

    Args:
        X: Shape (n_samples, n_features). Feature matrix.
        y: Shape (n_samples,). Target values.

    Returns:
        Optimal weights of shape (n_features,).
    """
    # Normal equation: w = (X^T X)^(-1) X^T y
    # Using np.linalg.solve is more numerically stable than explicit inverse
    # Solve: (X^T X) w = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    weights = np.linalg.solve(XtX, Xty)
    return weights


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute predictions given features and weights.

    Args:
        X: Shape (n_samples, n_features). Feature matrix.
        weights: Shape (n_features,). Model weights.

    Returns:
        Predictions of shape (n_samples,).
    """
    # Linear prediction: y = X @ w
    return X @ weights
