from __future__ import annotations

import numpy as np


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to the nearest centroid.

    Args:
        X: Shape (n_samples, n_features). Data points.
        centroids: Shape (k, n_features). Current centroid positions.

    Returns:
        Cluster assignments of shape (n_samples,) with integer indices in [0, k-1].
    """
    # Compute squared Euclidean distances using the identity:
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c.T
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n_samples, 1)
    C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, k)
    cross = X @ centroids.T  # (n_samples, k)

    # Squared distances: (n_samples, k)
    dist_sq = X_sq + C_sq - 2 * cross

    # Assign each point to the nearest centroid
    return np.argmin(dist_sq, axis=1)


def update_centroids(X: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
    """Compute new centroids as the mean of assigned points.

    Args:
        X: Shape (n_samples, n_features). Data points.
        assignments: Shape (n_samples,). Cluster assignment for each point.
        k: Number of clusters.

    Returns:
        Updated centroids of shape (k, n_features).
    """
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features), dtype=X.dtype)

    for j in range(k):
        # Find all points assigned to cluster j
        mask = assignments == j
        if np.any(mask):
            # Compute mean of assigned points
            centroids[j] = X[mask].mean(axis=0)
        # else: centroid stays at origin (zeros)

    return centroids


