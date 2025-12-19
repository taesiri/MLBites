from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax.

    Args:
        logits: Shape (n_samples, n_classes).

    Returns:
        Probabilities of shape (n_samples, n_classes), rows sum to 1.
    """
    # Subtract max per row for numerical stability (prevents overflow in exp)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    # Normalize to get probabilities
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute numerically stable cross-entropy loss.

    Args:
        logits: Shape (n_samples, n_classes).
        targets: Shape (n_samples,), integer class indices.

    Returns:
        Mean cross-entropy loss (scalar).
    """
    n_samples = logits.shape[0]

    # Compute log-softmax using the log-sum-exp trick for stability:
    # log_softmax = logits - log(sum(exp(logits)))
    # = logits - (max + log(sum(exp(logits - max))))
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = max_logits.squeeze(axis=1) + np.log(np.sum(np.exp(shifted), axis=1))

    # Get the logit for the correct class for each sample
    correct_logits = logits[np.arange(n_samples), targets]

    # Cross-entropy = -log(softmax[correct_class]) = log_sum_exp - correct_logit
    losses = log_sum_exp - correct_logits

    # Return mean loss
    return float(np.mean(losses))


