from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_allclose(actual: np.ndarray, expected: np.ndarray, *, atol: float, rtol: float) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: got {actual.shape}, expected {expected.shape}")
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"values not close (max abs diff={max_abs})\nactual={actual}\nexpected={expected}")


def _assert_prob_distribution(y: np.ndarray, *, axis: int, atol: float = 1e-7) -> None:
    if not np.all(np.isfinite(y)):
        raise AssertionError("softmax output contains non-finite values (inf/nan)")
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise AssertionError("softmax output should be in [0, 1]")
    s = np.sum(y, axis=axis)
    if not np.allclose(s, 1.0, atol=atol, rtol=0.0):
        raise AssertionError(f"softmax outputs must sum to 1 along axis={axis}; got sums={s}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s)/class(es)
            for this question (same signature as in starting_point.py).

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "softmax"):
        raise AssertionError("candidate must define: softmax")

    softmax = candidate.softmax  # type: ignore[attr-defined]

    # Test 1: 1D known values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = softmax(x)
    expected = np.array([0.09003057, 0.24472847, 0.66524096], dtype=np.float64)
    _assert_allclose(y, expected, atol=1e-7, rtol=1e-7)
    _assert_prob_distribution(y, axis=-1)

    # Test 2: 2D row-wise (axis=1) including uniform row
    x2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y2 = softmax(x2, axis=1)
    expected2 = np.array([[0.5, 0.5], [0.26894142, 0.73105858]], dtype=np.float64)
    _assert_allclose(y2, expected2, atol=1e-7, rtol=1e-7)
    _assert_prob_distribution(y2, axis=1)

    # Test 3: axis handling on 3D input
    x3 = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    y3_last = softmax(x3, axis=-1)
    _assert_prob_distribution(y3_last, axis=-1)
    if y3_last.shape != x3.shape:
        raise AssertionError(f"softmax must preserve shape; got {y3_last.shape}, expected {x3.shape}")

    y3_mid = softmax(x3, axis=1)
    _assert_prob_distribution(y3_mid, axis=1)
    if y3_mid.shape != x3.shape:
        raise AssertionError(f"softmax must preserve shape; got {y3_mid.shape}, expected {x3.shape}")

    # Test 4: invariance to constant shifts along the normalization axis
    x_shift = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
    y_a = softmax(x_shift, axis=1)
    y_b = softmax(x_shift + 1000.0, axis=1)
    _assert_allclose(y_a, y_b, atol=1e-9, rtol=1e-9)

    # Test 5: numerical stability on very large logits (naive exp would overflow)
    huge = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float64)
    y_huge = softmax(huge, axis=1)
    _assert_prob_distribution(y_huge, axis=1)
    expected_huge = np.exp(np.array([[0.0, 1.0, -1.0]], dtype=np.float64))
    expected_huge = expected_huge / np.sum(expected_huge, axis=1, keepdims=True)
    _assert_allclose(y_huge, expected_huge, atol=1e-10, rtol=1e-10)


