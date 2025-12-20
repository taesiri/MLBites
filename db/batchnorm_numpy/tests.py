from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_allclose(
    a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, msg: str
) -> None:
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def _numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient using central difference."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        fp = f(x)
        x[idx] = old_val - eps
        fm = f(x)
        x[idx] = old_val

        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines batchnorm_forward and batchnorm_backward.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "batchnorm_forward"):
        raise AssertionError("Candidate must define function `batchnorm_forward`.")
    if not hasattr(candidate, "batchnorm_backward"):
        raise AssertionError("Candidate must define function `batchnorm_backward`.")

    batchnorm_forward = candidate.batchnorm_forward
    batchnorm_backward = candidate.batchnorm_backward

    rng = np.random.default_rng(42)
    atol = 1e-5
    rtol = 1e-5

    # --- Test 1: Forward pass normalizes to zero mean and unit variance ---
    x1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    gamma1 = np.ones(2, dtype=np.float64)
    beta1 = np.zeros(2, dtype=np.float64)
    out1, cache1 = batchnorm_forward(x1, gamma1, beta1)

    # Check output shape
    if out1.shape != x1.shape:
        raise AssertionError(f"Output shape mismatch: {out1.shape} vs {x1.shape}")

    # Check normalized output has mean ≈ 0 and std ≈ 1
    out_mean = np.mean(out1, axis=0)
    out_std = np.std(out1, axis=0)
    _assert_allclose(
        out_mean,
        np.zeros(2),
        atol=atol,
        rtol=rtol,
        msg="Normalized output should have zero mean.",
    )
    _assert_allclose(
        out_std,
        np.ones(2),
        atol=atol,
        rtol=rtol,
        msg="Normalized output should have unit std.",
    )

    # --- Test 2: Forward pass with scale and shift ---
    gamma2 = np.array([2.0, 0.5], dtype=np.float64)
    beta2 = np.array([1.0, -1.0], dtype=np.float64)
    out2, _ = batchnorm_forward(x1, gamma2, beta2)

    # Check mean is beta and std is |gamma|
    out2_mean = np.mean(out2, axis=0)
    out2_std = np.std(out2, axis=0)
    _assert_allclose(
        out2_mean,
        beta2,
        atol=atol,
        rtol=rtol,
        msg="Output mean should equal beta.",
    )
    _assert_allclose(
        out2_std,
        np.abs(gamma2),
        atol=atol,
        rtol=rtol,
        msg="Output std should equal |gamma|.",
    )

    # --- Test 3: Backward pass - dbeta correctness ---
    x3 = rng.normal(size=(4, 3)).astype(np.float64)
    gamma3 = rng.normal(size=3).astype(np.float64)
    beta3 = rng.normal(size=3).astype(np.float64)
    out3, cache3 = batchnorm_forward(x3, gamma3, beta3)
    dout3 = rng.normal(size=out3.shape).astype(np.float64)
    dx3, dgamma3, dbeta3 = batchnorm_backward(dout3, cache3)

    # dbeta should be sum of dout over batch
    expected_dbeta = np.sum(dout3, axis=0)
    _assert_allclose(
        dbeta3,
        expected_dbeta,
        atol=atol,
        rtol=rtol,
        msg="dbeta should be sum of dout over batch dimension.",
    )

    # --- Test 4: Backward pass - dgamma correctness ---
    # dgamma = sum(dout * x_norm)
    x_norm3 = cache3["x_norm"]
    expected_dgamma = np.sum(dout3 * x_norm3, axis=0)
    _assert_allclose(
        dgamma3,
        expected_dgamma,
        atol=atol,
        rtol=rtol,
        msg="dgamma should be sum of (dout * x_norm) over batch.",
    )

    # --- Test 5: Numerical gradient check for dx ---
    x5 = rng.normal(size=(5, 4)).astype(np.float64)
    gamma5 = rng.normal(size=4).astype(np.float64)
    beta5 = rng.normal(size=4).astype(np.float64)
    dout5 = rng.normal(size=(5, 4)).astype(np.float64)

    def f_x(x_):
        out_, _ = batchnorm_forward(x_, gamma5, beta5)
        return np.sum(out_ * dout5)

    out5, cache5 = batchnorm_forward(x5.copy(), gamma5, beta5)
    dx5, _, _ = batchnorm_backward(dout5, cache5)
    dx5_num = _numerical_gradient(f_x, x5.copy())

    _assert_allclose(
        dx5,
        dx5_num,
        atol=1e-4,
        rtol=1e-4,
        msg="dx failed numerical gradient check.",
    )

    # --- Test 6: Numerical gradient check for dgamma ---
    def f_gamma(gamma_):
        out_, _ = batchnorm_forward(x5, gamma_, beta5)
        return np.sum(out_ * dout5)

    out6, cache6 = batchnorm_forward(x5, gamma5.copy(), beta5)
    _, dgamma6, _ = batchnorm_backward(dout5, cache6)
    dgamma6_num = _numerical_gradient(f_gamma, gamma5.copy())

    _assert_allclose(
        dgamma6,
        dgamma6_num,
        atol=1e-4,
        rtol=1e-4,
        msg="dgamma failed numerical gradient check.",
    )

    # --- Test 7: Numerical gradient check for dbeta ---
    def f_beta(beta_):
        out_, _ = batchnorm_forward(x5, gamma5, beta_)
        return np.sum(out_ * dout5)

    out7, cache7 = batchnorm_forward(x5, gamma5, beta5.copy())
    _, _, dbeta7 = batchnorm_backward(dout5, cache7)
    dbeta7_num = _numerical_gradient(f_beta, beta5.copy())

    _assert_allclose(
        dbeta7,
        dbeta7_num,
        atol=1e-4,
        rtol=1e-4,
        msg="dbeta failed numerical gradient check.",
    )

    # --- Test 8: Larger batch size ---
    x8 = rng.normal(size=(32, 64)).astype(np.float64)
    gamma8 = rng.normal(size=64).astype(np.float64)
    beta8 = rng.normal(size=64).astype(np.float64)
    out8, cache8 = batchnorm_forward(x8, gamma8, beta8)

    if out8.shape != x8.shape:
        raise AssertionError(f"Output shape mismatch for larger batch: {out8.shape}")

    dout8 = rng.normal(size=out8.shape).astype(np.float64)
    dx8, dgamma8, dbeta8 = batchnorm_backward(dout8, cache8)

    if dx8.shape != x8.shape:
        raise AssertionError(f"dx shape mismatch: {dx8.shape} vs {x8.shape}")
    if dgamma8.shape != gamma8.shape:
        raise AssertionError(f"dgamma shape mismatch: {dgamma8.shape} vs {gamma8.shape}")
    if dbeta8.shape != beta8.shape:
        raise AssertionError(f"dbeta shape mismatch: {dbeta8.shape} vs {beta8.shape}")

    # --- Test 9: Output should be finite ---
    if not np.all(np.isfinite(out8)):
        raise AssertionError("Forward pass output contains inf or nan.")
    if not np.all(np.isfinite(dx8)):
        raise AssertionError("dx contains inf or nan.")
    if not np.all(np.isfinite(dgamma8)):
        raise AssertionError("dgamma contains inf or nan.")
    if not np.all(np.isfinite(dbeta8)):
        raise AssertionError("dbeta contains inf or nan.")

    # --- Test 10: dx columns should sum to approximately zero ---
    # This is a property of batchnorm: gradients are centered
    dx_col_sums = np.sum(dx5, axis=0)
    _assert_allclose(
        dx_col_sums,
        np.zeros(dx5.shape[1]),
        atol=1e-4,
        rtol=1e-4,
        msg="dx columns should sum to approximately zero.",
    )



