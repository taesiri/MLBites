from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_allclose(a: np.ndarray | float, b: np.ndarray | float, *, atol: float, rtol: float, msg: str) -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if not np.allclose(aa, bb, atol=atol, rtol=rtol):
            diff = np.max(np.abs(aa - bb))
            raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={aa}\nb={bb}")
    else:
        if not np.isclose(float(a), float(b), atol=atol, rtol=rtol):
            diff = abs(float(a) - float(b))
            raise AssertionError(f"{msg}\nabs_diff={diff}\na={a}\nb={b}")


def _fit_reference(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    num_steps: int,
    l2: float,
) -> tuple[np.ndarray, float]:
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n, d = X64.shape

    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(num_steps):
        logits = X64 @ w + b
        p = 1.0 / (1.0 + np.exp(-logits))

        grad_logits = (p - y64) / float(n)
        grad_w = X64.T @ grad_logits + l2 * w
        grad_b = float(grad_logits.sum())

        w -= lr * grad_w
        b -= lr * grad_b

    return w, float(b)


def _predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    logits = np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + float(b)
    return 1.0 / (1.0 + np.exp(-logits))


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "fit_logistic_regression"):
        raise AssertionError("Candidate must define function `fit_logistic_regression`.")

    fit = candidate.fit_logistic_regression

    # --- test 1: Example 1 (exact one-step update) ---
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y = np.array([1.0, 0.0], dtype=np.float64)
    w, b = fit(X, y, lr=1.0, num_steps=1, l2=0.0)
    _assert_allclose(w, np.array([0.25, -0.25]), atol=1e-12, rtol=0.0, msg="Example 1: w mismatch.")
    _assert_allclose(float(b), 0.0, atol=1e-12, rtol=0.0, msg="Example 1: b mismatch.")

    # --- test 2: Example 2 (exact one-step update) ---
    # NOTE: Use float32 here to ensure candidates are not implicitly forced into float64.
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    w, b = fit(X, y, lr=0.5, num_steps=1, l2=0.0)
    _assert_allclose(w, np.array([1.0 / 12.0]), atol=1e-6, rtol=1e-6, msg="Example 2: w mismatch.")
    _assert_allclose(float(b), -1.0 / 12.0, atol=1e-6, rtol=1e-6, msg="Example 2: b mismatch.")

    # --- test 3: Match reference implementation on a deterministic dataset (multi-step + L2) ---
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = -0.25
    p = _predict_proba(X, true_w, true_b)
    y = (p >= 0.5).astype(np.float64)

    lr = 0.2
    num_steps = 200
    l2 = 0.1
    w_ref, b_ref = _fit_reference(X, y, lr=lr, num_steps=num_steps, l2=l2)
    w_cand, b_cand = fit(X, y, lr=lr, num_steps=num_steps, l2=l2)

    # Be tolerant to dtype/implementation choices (float32 vs float64), while still
    # enforcing the correct update rule.
    _assert_allclose(w_cand, w_ref, atol=1e-5, rtol=1e-5, msg="Multi-step: w mismatch vs reference.")
    _assert_allclose(float(b_cand), b_ref, atol=1e-5, rtol=1e-5, msg="Multi-step: b mismatch vs reference.")

    # --- test 4: sanity check that training learns something (accuracy) ---
    # Use a clearly linearly separable labeling rule and no regularization.
    logits = X @ true_w + true_b
    y_sep = (logits > 0.0).astype(np.float64)
    w_s, b_s = fit(X, y_sep, lr=0.5, num_steps=500, l2=0.0)

    probs = _predict_proba(X, np.asarray(w_s), float(b_s))
    preds = (probs >= 0.5).astype(np.float64)
    acc = float((preds == y_sep).mean())
    if acc < 0.9:
        raise AssertionError(f"Expected training accuracy >= 0.9, got {acc:.3f}.")


