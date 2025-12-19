from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute numerically stable softmax along the specified dimension.

    Args:
        x: Input tensor of any shape containing raw logits.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of same shape as x with softmax probabilities (sums to 1 along dim).
    """
    # Subtract max for numerical stability (prevents overflow in exp)
    # This doesn't change the result since softmax(x) = softmax(x - c) for any constant c
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    # Compute exp and normalize
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute numerically stable log-softmax along the specified dimension.

    Args:
        x: Input tensor of any shape containing raw logits.
        dim: Dimension along which to compute log-softmax.

    Returns:
        Tensor of same shape as x with log-probabilities.
    """
    # Use log-sum-exp trick for numerical stability:
    # log_softmax(x) = x - log(sum(exp(x)))
    #                = x - log(sum(exp(x - max))) - max
    #                = x - max - log(sum(exp(x - max)))
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    # Compute log(sum(exp(x_shifted))) which is now numerically stable
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=dim, keepdim=True))

    # log_softmax = x - max - log_sum_exp = x_shifted - log_sum_exp
    return x_shifted - log_sum_exp


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for classification.

    Args:
        logits: Raw logits of shape (N, C) where N is batch size, C is num classes.
        targets: Integer class indices of shape (N,) with values in [0, C-1].

    Returns:
        Scalar tensor containing mean cross-entropy loss.
    """
    # Compute log-softmax of logits
    log_probs = log_softmax(logits, dim=-1)

    # Get the log-probability of the correct class for each sample
    # Using advanced indexing: log_probs[i, targets[i]] for each i
    n_samples = logits.shape[0]
    correct_log_probs = log_probs[torch.arange(n_samples), targets]

    # Cross-entropy = -log(p_correct), return mean over batch
    return -correct_log_probs.mean()


