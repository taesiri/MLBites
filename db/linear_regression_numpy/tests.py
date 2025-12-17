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
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "fit"):
        raise AssertionError("Candidate must define function `fit`.")
    if not hasattr(candidate, "predict"):
        raise AssertionError("Candidate must define function `predict`.")

    fit = candidate.fit
    predict = candidate.predict

    rng = np.random.default_rng(42)
    atol = 1e-6
    rtol = 1e-6

    # --- Test 1: Simple 1D regression (perfect fit) ---
    X1 = np.array([[1.0], [2.0], [3.0], [4.0]])
    y1 = np.array([2.0, 4.0, 6.0, 8.0])

    weights1 = fit(X1, y1)
    expected_weights1 = np.array([2.0])
    _assert_allclose(
        weights1, expected_weights1,
        atol=atol, rtol=rtol,
        msg="fit failed on simple 1D test."
    )

    predictions1 = predict(X1, weights1)
    _assert_allclose(
        predictions1, y1,
        atol=atol, rtol=rtol,
        msg="predict failed on simple 1D test."
    )

    # --- Test 2: 2D features (perfect fit) ---
    X2 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    y2 = np.array([1.0, 2.0, 3.0])

    weights2 = fit(X2, y2)
    expected_weights2 = np.array([1.0, 2.0])
    _assert_allclose(
        weights2, expected_weights2,
        atol=atol, rtol=rtol,
        msg="fit failed on 2D features test."
    )

    predictions2 = predict(X2, weights2)
    _assert_allclose(
        predictions2, y2,
        atol=atol, rtol=rtol,
        msg="predict failed on 2D features test."
    )

    # --- Test 3: With bias column ---
    X3 = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0]
    ])
    y3 = np.array([3.0, 5.0, 7.0])  # y = 1 + 2*x

    weights3 = fit(X3, y3)
    expected_weights3 = np.array([1.0, 2.0])
    _assert_allclose(
        weights3, expected_weights3,
        atol=atol, rtol=rtol,
        msg="fit failed on bias column test."
    )

    predictions3 = predict(X3, weights3)
    _assert_allclose(
        predictions3, y3,
        atol=atol, rtol=rtol,
        msg="predict failed on bias column test."
    )

    # --- Test 4: Random data (verify normal equation) ---
    n_samples = 50
    n_features = 5
    X4 = rng.normal(size=(n_samples, n_features))
    true_weights4 = rng.normal(size=(n_features,))
    y4 = X4 @ true_weights4

    weights4 = fit(X4, y4)
    _assert_allclose(
        weights4, true_weights4,
        atol=atol, rtol=rtol,
        msg="fit failed on random data test."
    )

    predictions4 = predict(X4, weights4)
    _assert_allclose(
        predictions4, y4,
        atol=atol, rtol=rtol,
        msg="predict failed on random data test."
    )

    # --- Test 5: Noisy data (least squares fit) ---
    n_samples5 = 100
    n_features5 = 3
    X5 = rng.normal(size=(n_samples5, n_features5))
    true_weights5 = np.array([1.0, -2.0, 0.5])
    noise5 = rng.normal(0, 0.1, size=(n_samples5,))
    y5 = X5 @ true_weights5 + noise5

    weights5 = fit(X5, y5)
    # Weights should be close to true weights
    _assert_allclose(
        weights5, true_weights5,
        atol=0.1, rtol=0.1,
        msg="fit failed on noisy data test (weights not close to true)."
    )

    # Predictions should match X @ weights
    predictions5 = predict(X5, weights5)
    expected_predictions5 = X5 @ weights5
    _assert_allclose(
        predictions5, expected_predictions5,
        atol=atol, rtol=rtol,
        msg="predict output does not match X @ weights."
    )

    # --- Test 6: Verify output shapes ---
    X6 = rng.normal(size=(30, 4))
    y6 = rng.normal(size=(30,))

    weights6 = fit(X6, y6)
    if weights6.shape != (4,):
        raise AssertionError(
            f"fit returned wrong shape: expected (4,), got {weights6.shape}"
        )

    predictions6 = predict(X6, weights6)
    if predictions6.shape != (30,):
        raise AssertionError(
            f"predict returned wrong shape: expected (30,), got {predictions6.shape}"
        )

    # --- Test 7: Single sample, single feature ---
    X7 = np.array([[2.0]])
    y7 = np.array([6.0])

    weights7 = fit(X7, y7)
    expected_weights7 = np.array([3.0])
    _assert_allclose(
        weights7, expected_weights7,
        atol=atol, rtol=rtol,
        msg="fit failed on single sample test."
    )

    predictions7 = predict(X7, weights7)
    _assert_allclose(
        predictions7, y7,
        atol=atol, rtol=rtol,
        msg="predict failed on single sample test."
    )

    # --- Test 8: Large number of features ---
    n_samples8 = 100
    n_features8 = 20
    X8 = rng.normal(size=(n_samples8, n_features8))
    true_weights8 = rng.normal(size=(n_features8,))
    y8 = X8 @ true_weights8

    weights8 = fit(X8, y8)
    _assert_allclose(
        weights8, true_weights8,
        atol=atol, rtol=rtol,
        msg="fit failed on large features test."
    )

    predictions8 = predict(X8, weights8)
    _assert_allclose(
        predictions8, y8,
        atol=atol, rtol=rtol,
        msg="predict failed on large features test."
    )
