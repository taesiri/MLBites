from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_close(
    a: np.ndarray | float,
    b: np.ndarray | float,
    *,
    atol: float,
    rtol: float,
    msg: str,
) -> None:
    if isinstance(a, np.ndarray) != isinstance(b, np.ndarray):
        raise AssertionError(f"{msg}: type mismatch {type(a)} vs {type(b)}")

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            raise AssertionError(f"{msg}: shape mismatch {a.shape} vs {b.shape}")
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            max_abs = float(np.max(np.abs(a - b)))
            raise AssertionError(f"{msg}: not close (max_abs={max_abs:.3e})")
        return

    # floats
    af = float(a)
    bf = float(b)
    if not np.isclose(af, bf, atol=atol, rtol=rtol):
        raise AssertionError(f"{msg}: not close ({af} vs {bf})")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required functions
            for this question (same signature as in starting_point.py).

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "linear_regression_fit"):
        raise AssertionError("candidate must define linear_regression_fit")
    if not hasattr(candidate, "linear_regression_predict"):
        raise AssertionError("candidate must define linear_regression_predict")

    fit = candidate.linear_regression_fit
    predict = candidate.linear_regression_predict

    # --------------------
    # Example 1 (perfect line with intercept)
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([1.0, 3.0, 5.0], dtype=np.float64)  # y = 2x + 1
    w, b = fit(X, y, l2_reg=0.0, fit_intercept=True)
    _assert_close(np.asarray(w), np.array([2.0], dtype=np.float64), atol=1e-12, rtol=0.0, msg="ex1 w")
    _assert_close(float(b), 1.0, atol=1e-12, rtol=0.0, msg="ex1 b")
    y_pred = predict(X, w, b)
    if not isinstance(y_pred, np.ndarray) or y_pred.shape != (3,):
        raise AssertionError(f"predict must return ndarray of shape (N,); got {type(y_pred)} {getattr(y_pred, 'shape', None)}")
    _assert_close(y_pred, y, atol=1e-12, rtol=0.0, msg="ex1 y_pred")

    # --------------------
    # Example 2 (2D features, no intercept)
    X2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    y2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # y = 1*x1 + 2*x2
    w2, b2 = fit(X2, y2, l2_reg=0.0, fit_intercept=False)
    _assert_close(np.asarray(w2), np.array([1.0, 2.0], dtype=np.float64), atol=1e-12, rtol=0.0, msg="ex2 w")
    _assert_close(float(b2), 0.0, atol=0.0, rtol=0.0, msg="ex2 b")
    y2_pred = predict(X2, w2, b2)
    _assert_close(y2_pred, y2, atol=1e-12, rtol=0.0, msg="ex2 y_pred")

    # --------------------
    # y can be (N, 1)
    y_col = y.reshape(-1, 1)
    w_col, b_col = fit(X, y_col, l2_reg=0.0, fit_intercept=True)
    _assert_close(np.asarray(w_col), np.array([2.0], dtype=np.float64), atol=1e-12, rtol=0.0, msg="y (N,1) w")
    _assert_close(float(b_col), 1.0, atol=1e-12, rtol=0.0, msg="y (N,1) b")

    # --------------------
    # Intercept should NOT be regularized
    # With X all zeros, the optimal w is zeros and b is mean(y), regardless of l2_reg.
    Xz = np.zeros((5, 3), dtype=np.float64)
    yz = np.full((5,), 3.0, dtype=np.float64)
    wz, bz = fit(Xz, yz, l2_reg=1e6, fit_intercept=True)
    _assert_close(np.asarray(wz), np.zeros((3,), dtype=np.float64), atol=1e-12, rtol=0.0, msg="no-reg intercept w")
    _assert_close(float(bz), 3.0, atol=1e-12, rtol=0.0, msg="no-reg intercept b")

    # --------------------
    # Random recovery (no noise): should recover exact weights/intercept (up to numerical tolerance)
    rng = np.random.default_rng(0)
    N, D = 80, 4
    Xr = rng.normal(size=(N, D)).astype(np.float64)
    w_true = rng.normal(size=(D,)).astype(np.float64)
    b_true = float(rng.normal())
    yr = Xr @ w_true + b_true
    w_hat, b_hat = fit(Xr, yr, l2_reg=0.0, fit_intercept=True)
    _assert_close(np.asarray(w_hat), w_true, atol=1e-9, rtol=1e-9, msg="random recovery w")
    _assert_close(float(b_hat), b_true, atol=1e-9, rtol=1e-9, msg="random recovery b")

    # --------------------
    # Ridge should shrink weights as regularization increases (typical property)
    Xs = rng.normal(size=(60, 6)).astype(np.float64)
    w_big = (10.0 * rng.normal(size=(6,))).astype(np.float64)
    y_noise = Xs @ w_big + 0.5 * rng.normal(size=(60,)).astype(np.float64)
    w_ols, _ = fit(Xs, y_noise, l2_reg=0.0, fit_intercept=True)
    w_ridge, _ = fit(Xs, y_noise, l2_reg=100.0, fit_intercept=True)
    if float(np.linalg.norm(w_ridge)) > float(np.linalg.norm(w_ols)) + 1e-12:
        raise AssertionError("expected ridge solution to have smaller (or equal) weight norm than unregularized solution")

    # --------------------
    # l2_reg must be non-negative
    try:
        fit(X, y, l2_reg=-1.0, fit_intercept=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative l2_reg")


