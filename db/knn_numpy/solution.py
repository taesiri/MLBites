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
    n_test = X_test.shape[0]

    # Compute squared Euclidean distances using the identity:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b.T
    X_train_sq = np.sum(X_train ** 2, axis=1, keepdims=True)  # (n_train, 1)
    X_test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)    # (n_test, 1)
    cross = X_test @ X_train.T                                 # (n_test, n_train)

    # Squared distances: (n_test, n_train)
    dist_sq = X_test_sq + X_train_sq.T - 2 * cross

    # Get indices of k nearest neighbors for each test point
    # argsort returns indices that would sort the array
    nearest_indices = np.argsort(dist_sq, axis=1)[:, :k]  # (n_test, k)

    # Get labels of k nearest neighbors
    neighbor_labels = y_train[nearest_indices]  # (n_test, k)

    # Majority voting: find the most common label for each test point
    predictions = np.empty(n_test, dtype=y_train.dtype)

    for i in range(n_test):
        labels = neighbor_labels[i]
        # Count occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Find max count
        max_count = counts.max()
        # Get all labels with max count (for tie-breaking)
        tied_labels = unique_labels[counts == max_count]
        # Return smallest label in case of tie
        predictions[i] = tied_labels.min()

    return predictions




