from __future__ import annotations

import numpy as np


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    num_steps: int = 1000,
    l2: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Fit binary logistic regression with full-batch gradient descent.

    Model:
        p(y=1|x) = sigmoid(x @ w + b)

    Loss (average over examples) with optional L2 on w:
        L(w,b) = mean( -y*log(p) - (1-y)*log(1-p) ) + 0.5*l2*||w||^2

    Args:
        X: Array of shape (n, d) of float features.
        y: Array of shape (n,) with binary labels in {0, 1}.
        lr: Learning rate.
        num_steps: Number of gradient descent steps.
        l2: L2 regularization strength (applied to w only).

    Returns:
        (w, b) where w has shape (d,) and b is a float.
    """
    # TODO:
    # - Initialize w = zeros(d,), b = 0.0
    # - For num_steps steps:
    #   - logits = X @ w + b
    #   - p = sigmoid(logits) = 1 / (1 + exp(-logits))
    #   - grad_logits = (p - y) / n
    #   - grad_w = X.T @ grad_logits + l2 * w
    #   - grad_b = grad_logits.sum()
    #   - w -= lr * grad_w
    #   - b -= lr * grad_b
    # - Return w, b
    raise NotImplementedError


