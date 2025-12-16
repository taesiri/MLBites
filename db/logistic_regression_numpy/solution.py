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

    Uses the average binary cross-entropy loss plus optional L2 regularization:
        mean( -y*log(p) - (1-y)*log(1-p) ) + 0.5*l2*||w||^2
    where p = sigmoid(X @ w + b).
    """
    # Use the input dtype when it's already floating, but ensure at least float32.
    # (Candidates shouldn't be forced into float64 to pass tests.)
    dtype = np.result_type(X, y, np.float32)
    Xf = np.asarray(X, dtype=dtype)
    yf = np.asarray(y, dtype=dtype)

    n, d = Xf.shape
    w = np.zeros(d, dtype=dtype)
    b = dtype.type(0.0)

    for _ in range(num_steps):
        logits = Xf @ w + b
        p = dtype.type(1.0) / (dtype.type(1.0) + np.exp(-logits))

        grad_logits = (p - yf) / dtype.type(n)
        grad_w = Xf.T @ grad_logits + dtype.type(l2) * w
        grad_b = grad_logits.sum()

        w -= lr * grad_w
        b -= lr * grad_b

    return w, float(b)


