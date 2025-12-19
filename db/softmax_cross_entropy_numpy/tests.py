from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_allclose(
    a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, msg: str
) -> None:
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines softmax and cross_entropy_loss.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "softmax"):
        raise AssertionError("Candidate must define function `softmax`.")
    if not hasattr(candidate, "cross_entropy_loss"):
        raise AssertionError("Candidate must define function `cross_entropy_loss`.")

    softmax = candidate.softmax
    cross_entropy_loss = candidate.cross_entropy_loss

    rng = np.random.default_rng(42)
    atol = 1e-6
    rtol = 1e-6

    # --- Test 1: softmax basic correctness ---
    logits1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    probs1 = softmax(logits1)
    exp_vals = np.exp(logits1 - np.max(logits1))
    expected1 = exp_vals / np.sum(exp_vals)
    _assert_allclose(probs1, expected1, atol=atol, rtol=rtol, msg="Softmax basic test failed.")

    # Check rows sum to 1
    if not np.allclose(np.sum(probs1, axis=1), 1.0, atol=1e-9):
        raise AssertionError("Softmax output rows must sum to 1.")

    # --- Test 2: softmax numerical stability with large logits ---
    logits2 = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float64)
    probs2 = softmax(logits2)

    # Should not produce inf/nan
    if not np.all(np.isfinite(probs2)):
        raise AssertionError("Softmax must be numerically stable (no inf/nan with large logits).")

    # Should match the same distribution as [0, 1, 2]
    probs_ref = softmax(np.array([[0.0, 1.0, 2.0]], dtype=np.float64))
    _assert_allclose(probs2, probs_ref, atol=atol, rtol=rtol, msg="Softmax stability test failed.")

    # --- Test 3: softmax with negative logits ---
    logits3 = np.array([[-1000.0, -999.0, -998.0]], dtype=np.float64)
    probs3 = softmax(logits3)
    if not np.all(np.isfinite(probs3)):
        raise AssertionError("Softmax must handle large negative logits without underflow.")
    if not np.allclose(np.sum(probs3, axis=1), 1.0, atol=1e-9):
        raise AssertionError("Softmax output rows must sum to 1 for negative logits.")

    # --- Test 4: softmax batch processing ---
    logits4 = rng.normal(size=(10, 5)).astype(np.float64)
    probs4 = softmax(logits4)
    if probs4.shape != logits4.shape:
        raise AssertionError(f"Softmax output shape mismatch: {probs4.shape} vs {logits4.shape}")
    if not np.allclose(np.sum(probs4, axis=1), np.ones(10), atol=1e-9):
        raise AssertionError("Softmax rows must all sum to 1 in batch.")

    # --- Test 5: cross-entropy basic correctness ---
    logits5 = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float64)
    targets5 = np.array([0, 1], dtype=np.int64)
    loss5 = cross_entropy_loss(logits5, targets5)

    # Manual calculation using log-sum-exp
    def manual_ce(logits: np.ndarray, targets: np.ndarray) -> float:
        n = logits.shape[0]
        max_l = np.max(logits, axis=1, keepdims=True)
        lse = max_l.squeeze() + np.log(np.sum(np.exp(logits - max_l), axis=1))
        correct = logits[np.arange(n), targets]
        return float(np.mean(lse - correct))

    expected_loss5 = manual_ce(logits5, targets5)
    if not np.isclose(loss5, expected_loss5, atol=atol, rtol=rtol):
        raise AssertionError(
            f"Cross-entropy loss mismatch.\nExpected: {expected_loss5}\nActual: {loss5}"
        )

    # --- Test 6: cross-entropy stability with large logits ---
    logits6 = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float64)
    targets6 = np.array([2], dtype=np.int64)
    loss6 = cross_entropy_loss(logits6, targets6)
    if not np.isfinite(loss6):
        raise AssertionError("Cross-entropy must be stable with large logits (no inf/nan).")

    # Should match loss for [0, 1, 2] with target 2
    logits6_ref = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    loss6_ref = cross_entropy_loss(logits6_ref, targets6)
    if not np.isclose(loss6, loss6_ref, atol=atol, rtol=rtol):
        raise AssertionError(
            f"Cross-entropy stability test failed.\nExpected: {loss6_ref}\nActual: {loss6}"
        )

    # --- Test 7: cross-entropy with perfect prediction (low loss) ---
    logits7 = np.array([[100.0, 0.0, 0.0]], dtype=np.float64)
    targets7 = np.array([0], dtype=np.int64)
    loss7 = cross_entropy_loss(logits7, targets7)
    if not (loss7 < 1e-40):
        raise AssertionError(f"Loss should be near zero for perfect prediction, got {loss7}")

    # --- Test 8: cross-entropy batch with random data ---
    n_samples, n_classes = 100, 10
    logits8 = rng.normal(size=(n_samples, n_classes)).astype(np.float64)
    targets8 = rng.integers(0, n_classes, size=n_samples).astype(np.int64)
    loss8 = cross_entropy_loss(logits8, targets8)

    expected_loss8 = manual_ce(logits8, targets8)
    if not np.isclose(loss8, expected_loss8, atol=atol, rtol=rtol):
        raise AssertionError(
            f"Cross-entropy batch test failed.\nExpected: {expected_loss8}\nActual: {loss8}"
        )

    # --- Test 9: loss is always non-negative ---
    for _ in range(5):
        logits_rand = rng.normal(size=(20, 5)).astype(np.float64)
        targets_rand = rng.integers(0, 5, size=20).astype(np.int64)
        loss_rand = cross_entropy_loss(logits_rand, targets_rand)
        if loss_rand < 0:
            raise AssertionError(f"Cross-entropy loss must be non-negative, got {loss_rand}")


