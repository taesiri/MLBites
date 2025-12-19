from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax.

    Args:
        logits: Shape (n_samples, n_classes).

    Returns:
        Probabilities of shape (n_samples, n_classes), rows sum to 1.
    """
    raise NotImplementedError


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute numerically stable cross-entropy loss.

    Args:
        logits: Shape (n_samples, n_classes).
        targets: Shape (n_samples,), integer class indices.

    Returns:
        Mean cross-entropy loss (scalar).
    """
    raise NotImplementedError


