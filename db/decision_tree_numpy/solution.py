from __future__ import annotations

import numpy as np


def compute_gini(y: np.ndarray) -> float:
    """Compute the Gini impurity of a label array.

    Args:
        y: Shape (n_samples,). Integer class labels.

    Returns:
        Gini impurity value between 0 (pure) and 1 - 1/n_classes (max impurity).
    """
    n = len(y)
    if n == 0:
        return 0.0

    # Count occurrences of each class
    _, counts = np.unique(y, return_counts=True)

    # Compute class probabilities
    probs = counts / n

    # Gini = 1 - sum(p_i^2)
    return 1.0 - np.sum(probs ** 2)


def find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Find the best feature and threshold to split the data.

    Args:
        X: Shape (n_samples, n_features). Feature matrix.
        y: Shape (n_samples,). Integer class labels.

    Returns:
        Tuple of (best_feature_idx, best_threshold, best_gini).
        If no valid split exists, returns (-1, 0.0, float('inf')).
    """
    n_samples, n_features = X.shape

    best_feature = -1
    best_threshold = 0.0
    best_gini = float("inf")

    # Try each feature
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]

        # Get unique values as potential thresholds
        unique_values = np.unique(feature_values)

        # Try each unique value as a threshold
        for threshold in unique_values:
            # Split: left gets samples where feature <= threshold
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            # Skip if split results in empty child
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            if n_left == 0 or n_right == 0:
                continue

            # Compute weighted Gini impurity
            gini_left = compute_gini(y[left_mask])
            gini_right = compute_gini(y[right_mask])
            weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples

            # Update best split if this is better
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gini


