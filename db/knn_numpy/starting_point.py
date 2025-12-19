from __future__ import annotations

import numpy as np


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
) -> np.ndarray:
    """Predict labels for test points using K-Nearest Neighbors.

    Args:
        X_train: Shape (n_train, n_features). Training data points.
        y_train: Shape (n_train,). Training labels (integers).
        X_test: Shape (n_test, n_features). Test data points.
        k: Number of nearest neighbors to consider.

    Returns:
        Predicted labels of shape (n_test,) with integer class indices.
    """
    raise NotImplementedError


