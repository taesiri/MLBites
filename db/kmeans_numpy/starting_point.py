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
    raise NotImplementedError


def update_centroids(X: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
    """Compute new centroids as the mean of assigned points.

    Args:
        X: Shape (n_samples, n_features). Data points.
        assignments: Shape (n_samples,). Cluster assignment for each point.
        k: Number of clusters.

    Returns:
        Updated centroids of shape (k, n_features).
    """
    raise NotImplementedError


