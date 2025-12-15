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


def _loss(candidate: ModuleType, x: np.ndarray, dy: np.ndarray) -> float:
    y, _ = candidate.relu_forward(x)
    return float(np.sum(y * dy))


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module."""
    if not hasattr(candidate, "relu_forward"):
        raise AssertionError("candidate must define relu_forward")
    if not hasattr(candidate, "relu_backward"):
        raise AssertionError("candidate must define relu_backward")

    # --------------------
    # Forward: deterministic examples
    x1 = np.array([-2.0, 0.0, 3.5], dtype=np.float64)
    y1, cache1 = candidate.relu_forward(x1)
    if cache1 is None:
        raise AssertionError("relu_forward must return a non-None cache")
    expected1 = np.array([0.0, 0.0, 3.5], dtype=np.float64)
    _assert_close(np.asarray(y1), expected1, atol=0.0, rtol=0.0, msg="forward example 1")

    x2 = np.array([[-1.0, 0.0, 2.0], [3.0, -4.0, 0.0]], dtype=np.float64)
    y2, _ = candidate.relu_forward(x2)
    expected2 = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    _assert_close(np.asarray(y2), expected2, atol=0.0, rtol=0.0, msg="forward example 2")

    # --------------------
    # Backward: deterministic example (derivative at x==0 is defined as 0)
    x3 = np.array([[-1.0, 0.0, 2.0]], dtype=np.float64)
    _, cache3 = candidate.relu_forward(x3)
    dy3 = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
    dx3 = candidate.relu_backward(dy3, cache3)
    expected_dx3 = np.array([[0.0, 0.0, 30.0]], dtype=np.float64)
    _assert_close(np.asarray(dx3), expected_dx3, atol=0.0, rtol=0.0, msg="backward example 1")

    # --------------------
    # Backward: finite-difference gradient check
    # Use scalar loss L = sum(relu(x) * dy). Avoid x values near 0 (ReLU not differentiable at 0).
    rng = np.random.default_rng(0)
    xg = rng.normal(size=(4, 5)).astype(np.float64)
    xg[np.abs(xg) < 0.2] += 0.3
    dy = rng.normal(size=xg.shape).astype(np.float64)

    y, cache = candidate.relu_forward(xg)
    if not isinstance(np.asarray(y), np.ndarray) or np.asarray(y).shape != xg.shape:
        raise AssertionError(f"y must be ndarray with shape {xg.shape}, got {type(y)} {getattr(y, 'shape', None)}")

    dx = candidate.relu_backward(dy, cache)
    dx_arr = np.asarray(dx)
    if dx_arr.shape != xg.shape:
        raise AssertionError(f"dx must have shape {xg.shape}, got {dx_arr.shape}")

    eps = 1e-6

    def fx(z):
        return _loss(candidate, z, dy)

    num_dx = _numerical_grad(fx, xg.copy(), eps=eps)
    _assert_close(dx_arr, num_dx, atol=2e-6, rtol=2e-6, msg="dx numerical grad")


