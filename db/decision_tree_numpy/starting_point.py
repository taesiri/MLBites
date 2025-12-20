from __future__ import annotations

import numpy as np


def compute_gini(y: np.ndarray) -> float:
    """Compute the Gini impurity of a label array.

    Args:
        y: Shape (n_samples,). Integer class labels.

    Returns:
        Gini impurity value between 0 (pure) and 1 - 1/n_classes (max impurity).
    """
    raise NotImplementedError


def find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Find the best feature and threshold to split the data.

    Args:
        X: Shape (n_samples, n_features). Feature matrix.
        y: Shape (n_samples,). Integer class labels.

    Returns:
        Tuple of (best_feature_idx, best_threshold, best_gini).
        If no valid split exists, returns (-1, 0.0, float('inf')).
    """
    raise NotImplementedError




