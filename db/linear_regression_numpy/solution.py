from __future__ import annotations

import numpy as np


def _as_1d_targets(y: np.ndarray, *, N: int) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        if y.shape != (N,):
            raise ValueError(f"y must have shape (N,) where N={N}; got {y.shape}")
        return y
    if y.ndim == 2 and y.shape == (N, 1):
        return y[:, 0]
    raise ValueError(f"y must have shape (N,) or (N,1); got {y.shape}")


def linear_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2_reg: float = 0.0,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float]:
    """Fit linear regression (optionally ridge-regularized) parameters.

    Solves:
        min_{w,b} ||Xw + b - y||^2 + l2_reg * ||w||^2

    Notes:
    - If fit_intercept is False, returns b = 0.0.
    - The intercept is NOT regularized.
    """
    if l2_reg < 0.0:
        raise ValueError("l2_reg must be non-negative")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (N,D); got {X.ndim}D")
    N, D = X.shape
    if N <= 0 or D <= 0:
        raise ValueError(f"X must have shape (N,D) with N>0 and D>0; got {X.shape}")

    y1 = _as_1d_targets(y, N=N).astype(np.float64, copy=False)

    if fit_intercept:
        ones = np.ones((N, 1), dtype=X.dtype)
        X_aug = np.concatenate([X, ones], axis=1)  # (N, D+1); intercept is last
        A = X_aug.T @ X_aug  # (D+1, D+1)
        rhs = X_aug.T @ y1  # (D+1,)

        if l2_reg > 0.0:
            reg = l2_reg * np.eye(D + 1, dtype=X.dtype)
            reg[-1, -1] = 0.0  # don't regularize intercept
            theta = np.linalg.solve(A + reg, rhs)
        else:
            # lstsq is robust when A is singular / poorly-conditioned.
            theta = np.linalg.lstsq(X_aug, y1, rcond=None)[0]

        w = theta[:-1]
        b = float(theta[-1])
        return w, b

    # No intercept
    A = X.T @ X  # (D, D)
    rhs = X.T @ y1  # (D,)

    if l2_reg > 0.0:
        w = np.linalg.solve(A + l2_reg * np.eye(D, dtype=X.dtype), rhs)
    else:
        w = np.linalg.lstsq(X, y1, rcond=None)[0]
    return w, 0.0


def linear_regression_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict targets for a linear model y = Xw + b."""
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,D); got {X.ndim}D")
    if w.ndim != 1:
        raise ValueError(f"w must be 1D (D,); got {w.ndim}D")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"shape mismatch: X is {X.shape}, w is {w.shape}")
    return X @ w + float(b)


