from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_close(
    a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, msg: str
) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{msg}: shape mismatch {a.shape} vs {b.shape}")
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}: not close (max_abs={max_abs:.3e})")


def _loss_from_forward(
    candidate: ModuleType,
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    dy: np.ndarray,
    *,
    eps: float,
) -> float:
    y, _ = candidate.layernorm_forward(x, gamma, beta, eps=eps)
    return float(np.sum(y * dy))


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


def _as_row(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v.reshape(1, -1)
    return v


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module."""
    if not hasattr(candidate, "layernorm_forward"):
        raise AssertionError("candidate must define layernorm_forward")
    if not hasattr(candidate, "layernorm_backward"):
        raise AssertionError("candidate must define layernorm_backward")

    # --------------------
    # Forward: exact example (eps=0)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    gamma = np.array([2.0, 1.0], dtype=np.float64)
    beta = np.array([0.5, -0.5], dtype=np.float64)
    y, cache = candidate.layernorm_forward(x, gamma, beta, eps=0.0)
    expected = np.array([[-1.5, 0.5], [-1.5, 0.5]], dtype=np.float64)
    _assert_close(y, expected, atol=0.0, rtol=0.0, msg="forward example 1")
    if cache is None:
        raise AssertionError("forward must return a non-None cache")

    # Forward: constant row -> zeros (default eps)
    x2 = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    g2 = np.ones(3, dtype=np.float64)
    b2 = np.zeros(3, dtype=np.float64)
    y2, _ = candidate.layernorm_forward(x2, g2, b2)
    _assert_close(y2, np.zeros_like(x2), atol=1e-12, rtol=0.0, msg="forward example 2")

    # --------------------
    # Property check: per-row mean ~ 0, var ~ 1 when gamma=1, beta=0
    rng = np.random.default_rng(0)
    xp = rng.normal(size=(4, 7)).astype(np.float64)
    gp = np.ones(7, dtype=np.float64)
    bp = np.zeros((1, 7), dtype=np.float64)
    # Use eps=0 so the normalized outputs have per-row variance ~ 1.
    yp, _ = candidate.layernorm_forward(xp, gp, bp, eps=0.0)
    row_mean = np.mean(yp, axis=1)
    row_var = np.var(yp, axis=1)
    _assert_close(
        row_mean, np.zeros_like(row_mean), atol=1e-10, rtol=0.0, msg="row mean"
    )
    _assert_close(row_var, np.ones_like(row_var), atol=1e-6, rtol=0.0, msg="row var")

    # --------------------
    # Backward: finite-difference gradient check
    N, D = 3, 5
    xg = rng.normal(size=(N, D)).astype(np.float64)
    gg = rng.normal(size=(D,)).astype(np.float64)
    bg = rng.normal(size=(1, D)).astype(np.float64)
    dy = rng.normal(size=(N, D)).astype(np.float64)

    y, cache = candidate.layernorm_forward(xg, gg, bg)
    dx, dgamma, dbeta = candidate.layernorm_backward(dy, cache)

    if not isinstance(dx, np.ndarray) or dx.shape != xg.shape:
        raise AssertionError(
            f"dx must be ndarray with shape {xg.shape}, got {type(dx)} {getattr(dx, 'shape', None)}"
        )
    dgamma_row = _as_row(np.asarray(dgamma))
    dbeta_row = _as_row(np.asarray(dbeta))
    if dgamma_row.shape != (1, D):
        raise AssertionError(
            f"dgamma must have shape (D,) or (1,D); got {np.asarray(dgamma).shape}"
        )
    if dbeta_row.shape != (1, D):
        raise AssertionError(
            f"dbeta must have shape (D,) or (1,D); got {np.asarray(dbeta).shape}"
        )

    eps = 1e-5

    def fx(z):
        return _loss_from_forward(candidate, z, gg, bg, dy, eps=eps)

    def fg(z):
        return _loss_from_forward(candidate, xg, z, bg, dy, eps=eps)

    def fb(z):
        return _loss_from_forward(candidate, xg, gg, z, dy, eps=eps)

    xg_copy = xg.copy()
    gg_copy = gg.copy()
    bg_copy = bg.copy()

    num_dx = _numerical_grad(fx, xg_copy)
    num_dg = _numerical_grad(fg, gg_copy)
    num_db = _numerical_grad(fb, bg_copy)

    _assert_close(dx, num_dx, atol=2e-6, rtol=2e-6, msg="dx numerical grad")
    _assert_close(
        dgamma_row,
        num_dg.reshape(1, D),
        atol=2e-6,
        rtol=2e-6,
        msg="dgamma numerical grad",
    )
    _assert_close(
        dbeta_row,
        num_db.reshape(1, D),
        atol=2e-6,
        rtol=2e-6,
        msg="dbeta numerical grad",
    )
