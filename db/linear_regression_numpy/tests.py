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
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "LinearRegression"):
        raise AssertionError("Candidate must define class `LinearRegression`.")

    LinearRegression = candidate.LinearRegression  # type: ignore[attr-defined]

    rng = np.random.default_rng(0)
    atol = 1e-10
    rtol = 1e-10

    # --- test 1: exact recovery on noiseless data (with intercept) ---
    n, d = 50, 3
    X = rng.normal(size=(n, d)).astype(np.float64)
    w_true = np.array([1.5, -2.0, 0.25], dtype=np.float64)
    b_true = 0.7
    y = X @ w_true + b_true

    m = LinearRegression(fit_intercept=True).fit(X, y)
    if not hasattr(m, "coef_") or not hasattr(m, "intercept_"):
        raise AssertionError(
            "After fit(), model must have attributes `coef_` and `intercept_`."
        )

    _assert_allclose(
        np.asarray(m.coef_, dtype=np.float64),
        w_true,
        atol=atol,
        rtol=rtol,
        msg="Failed to recover coefficients on noiseless data (fit_intercept=True).",
    )

    if not np.isclose(float(m.intercept_), b_true, atol=atol, rtol=rtol):
        raise AssertionError(
            "Failed to recover intercept on noiseless data.\n"
            f"expected={b_true}\nactual={m.intercept_}"
        )

    X_test = rng.normal(size=(10, d)).astype(np.float64)
    y_pred = m.predict(X_test)
    _assert_allclose(
        np.asarray(y_pred, dtype=np.float64),
        X_test @ w_true + b_true,
        atol=atol,
        rtol=rtol,
        msg="Predictions mismatch (fit_intercept=True).",
    )

    # --- test 2: fit without intercept should match least-squares on X directly ---
    n2, d2 = 40, 2
    X2 = rng.normal(size=(n2, d2)).astype(np.float64)
    y2 = X2 @ np.array([-3.0, 0.5], dtype=np.float64)  # intercept = 0

    m2 = LinearRegression(fit_intercept=False).fit(X2, y2)
    beta_expected, *_ = np.linalg.lstsq(X2, y2, rcond=None)

    _assert_allclose(
        np.asarray(m2.coef_, dtype=np.float64),
        beta_expected,
        atol=atol,
        rtol=rtol,
        msg="Coefficients mismatch vs np.linalg.lstsq baseline (fit_intercept=False).",
    )

    if float(m2.intercept_) != 0.0:
        raise AssertionError("When fit_intercept=False, intercept_ must be exactly 0.0.")

    y2_pred = m2.predict(X2[:7])
    _assert_allclose(
        np.asarray(y2_pred, dtype=np.float64),
        X2[:7] @ beta_expected,
        atol=atol,
        rtol=rtol,
        msg="Predictions mismatch (fit_intercept=False).",
    )


