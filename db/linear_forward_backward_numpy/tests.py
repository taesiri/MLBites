from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_close(a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, msg: str) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{msg}: shape mismatch {a.shape} vs {b.shape}")
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}: not close (max_abs={max_abs:.3e})")


def _numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(x[idx])
        x[idx] = old + eps
        fxph = float(f(x))
        x[idx] = old - eps
        fxmh = float(f(x))
        x[idx] = old
        grad[idx] = (fxph - fxmh) / (2.0 * eps)
        it.iternext()
    return grad


def _loss(candidate: ModuleType, x: np.ndarray, W: np.ndarray, b: np.ndarray, dy: np.ndarray) -> float:
    y, _ = candidate.linear_forward(x, W, b)
    return float(np.sum(y * dy))


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module."""
    if not hasattr(candidate, "linear_forward"):
        raise AssertionError("candidate must define linear_forward")
    if not hasattr(candidate, "linear_backward"):
        raise AssertionError("candidate must define linear_backward")

    # --------------------
    # Forward: deterministic examples
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    W = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    b = np.array([1.0, -1.0], dtype=np.float64)
    y, cache = candidate.linear_forward(x, W, b)
    expected = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float64)
    _assert_close(y, expected, atol=0.0, rtol=0.0, msg="forward example 1")
    if cache is None:
        raise AssertionError("forward must return a non-None cache")

    x2 = np.array([[1.0, -1.0]], dtype=np.float64)
    W2 = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
    b2 = np.array([0.0, 1.0], dtype=np.float64)
    y2, _ = candidate.linear_forward(x2, W2, b2)
    expected2 = np.array([[1.5, -3.0]], dtype=np.float64)
    _assert_close(y2, expected2, atol=0.0, rtol=0.0, msg="forward example 2")

    # --------------------
    # Backward: shape checks + finite-difference gradient check
    rng = np.random.default_rng(0)
    N, D, M = 4, 3, 5
    xg = rng.normal(size=(N, D)).astype(np.float64)
    Wg = rng.normal(size=(D, M)).astype(np.float64)
    bg = rng.normal(size=(M,)).astype(np.float64)
    dy = rng.normal(size=(N, M)).astype(np.float64)

    y, cache = candidate.linear_forward(xg, Wg, bg)
    if not isinstance(y, np.ndarray) or y.shape != (N, M):
        raise AssertionError(f"y must be ndarray with shape {(N, M)}, got {type(y)} {getattr(y, 'shape', None)}")

    dx, dW, db = candidate.linear_backward(dy, cache)
    if not isinstance(dx, np.ndarray) or dx.shape != (N, D):
        raise AssertionError(f"dx must be ndarray with shape {(N, D)}, got {type(dx)} {getattr(dx, 'shape', None)}")
    if not isinstance(dW, np.ndarray) or dW.shape != (D, M):
        raise AssertionError(f"dW must be ndarray with shape {(D, M)}, got {type(dW)} {getattr(dW, 'shape', None)}")
    db_arr = np.asarray(db)
    if db_arr.shape != (M,):
        raise AssertionError(f"db must have shape {(M,)}, got {db_arr.shape}")

    # Numerical gradients against scalar loss L = sum(y * dy)
    eps = 1e-6

    def fx(z):
        return _loss(candidate, z, Wg, bg, dy)

    def fW(z):
        return _loss(candidate, xg, z, bg, dy)

    def fb(z):
        return _loss(candidate, xg, Wg, z, dy)

    num_dx = _numerical_grad(fx, xg.copy(), eps=eps)
    num_dW = _numerical_grad(fW, Wg.copy(), eps=eps)
    num_db = _numerical_grad(fb, bg.copy(), eps=eps)

    _assert_close(dx, num_dx, atol=2e-6, rtol=2e-6, msg="dx numerical grad")
    _assert_close(dW, num_dW, atol=2e-6, rtol=2e-6, msg="dW numerical grad")
    _assert_close(db_arr, num_db, atol=2e-6, rtol=2e-6, msg="db numerical grad")


