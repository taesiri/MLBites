from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_close(a: float, b: float, *, atol: float, msg: str) -> None:
    if not np.isclose(a, b, atol=atol, rtol=1e-6):
        raise AssertionError(f"{msg}\nExpected: {b}\nActual: {a}")


def _assert_allclose(
    a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, msg: str
) -> None:
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines sigmoid, compute_loss,
            compute_gradients, and train functions.

    Raises:
        AssertionError: if any test fails.
    """
    for name in ["sigmoid", "compute_loss", "compute_gradients", "train"]:
        if not hasattr(candidate, name):
            raise AssertionError(f"Candidate must define function `{name}`.")

    sigmoid = candidate.sigmoid
    compute_loss = candidate.compute_loss
    compute_gradients = candidate.compute_gradients
    train = candidate.train

    rng = np.random.default_rng(42)
    atol = 1e-6
    rtol = 1e-6

    # --- Test 1: sigmoid basic values ---
    z1 = np.array([0.0, 2.0, -2.0])
    s1 = sigmoid(z1)
    expected_s1 = np.array([0.5, 1 / (1 + np.exp(-2)), 1 / (1 + np.exp(2))])
    _assert_allclose(s1, expected_s1, atol=atol, rtol=rtol, msg="Sigmoid basic test failed.")

    # --- Test 2: sigmoid at extreme values ---
    z2 = np.array([100.0, -100.0])
    s2 = sigmoid(z2)
    if not (s2[0] > 0.99999 and s2[1] < 0.00001):
        raise AssertionError(f"Sigmoid should saturate at extremes. Got: {s2}")

    # --- Test 3: sigmoid shape preservation ---
    z3 = rng.normal(size=(3, 4, 5))
    s3 = sigmoid(z3)
    if s3.shape != z3.shape:
        raise AssertionError(f"Sigmoid shape mismatch: {s3.shape} vs {z3.shape}")

    # --- Test 4: sigmoid output range ---
    z4 = rng.normal(size=(100,)) * 10
    s4 = sigmoid(z4)
    if not (np.all(s4 > 0) and np.all(s4 < 1)):
        raise AssertionError("Sigmoid outputs must be in (0, 1).")

    # --- Test 5: compute_loss basic ---
    X5 = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    y5 = np.array([0.0, 0.0, 1.0])
    w5 = np.array([0.1, 0.2])
    b5 = -0.5

    # Manual calculation
    z5 = X5 @ w5 + b5
    p5 = 1 / (1 + np.exp(-z5))
    eps = 1e-15
    p5_clip = np.clip(p5, eps, 1 - eps)
    expected_loss5 = -np.mean(y5 * np.log(p5_clip) + (1 - y5) * np.log(1 - p5_clip))

    loss5 = compute_loss(X5, y5, w5, b5)
    _assert_close(loss5, expected_loss5, atol=atol, msg="compute_loss basic test failed.")

    # --- Test 6: compute_loss with perfect predictions ---
    X6 = np.array([[10.0], [-10.0]])
    y6 = np.array([1.0, 0.0])
    w6 = np.array([1.0])
    b6 = 0.0
    loss6 = compute_loss(X6, y6, w6, b6)
    if not (loss6 < 0.001):
        raise AssertionError(f"Loss should be near zero for perfect predictions, got {loss6}")

    # --- Test 7: compute_gradients basic ---
    X7 = np.array([[1.0, 2.0], [2.0, 1.0]])
    y7 = np.array([0.0, 1.0])
    w7 = np.array([0.0, 0.0])
    b7 = 0.0

    dw7, db7 = compute_gradients(X7, y7, w7, b7)

    # At w=0, b=0: p = 0.5 for all samples
    # error = p - y = [0.5, -0.5]
    # dw = X.T @ error / n = [[1, 2], [2, 1]].T @ [0.5, -0.5] / 2
    #    = [[1*0.5 + 2*(-0.5)], [2*0.5 + 1*(-0.5)]] / 2 = [-0.5, 0.5] / 2 = [-0.25, 0.25]
    # db = mean([0.5, -0.5]) = 0
    expected_dw7 = np.array([-0.25, 0.25])
    expected_db7 = 0.0

    _assert_allclose(dw7, expected_dw7, atol=atol, rtol=rtol, msg="compute_gradients dw test failed.")
    _assert_close(db7, expected_db7, atol=atol, msg="compute_gradients db test failed.")

    # --- Test 8: compute_gradients shape ---
    n_samples, n_features = 50, 10
    X8 = rng.normal(size=(n_samples, n_features))
    y8 = rng.integers(0, 2, size=n_samples).astype(float)
    w8 = rng.normal(size=n_features)
    b8 = 0.5

    dw8, db8 = compute_gradients(X8, y8, w8, b8)
    if dw8.shape != (n_features,):
        raise AssertionError(f"dw shape mismatch: {dw8.shape} vs ({n_features},)")
    if not isinstance(db8, (float, np.floating)):
        raise AssertionError(f"db should be a scalar, got type {type(db8)}")

    # --- Test 9: train on linearly separable data ---
    # Create linearly separable 2D data
    rng_train = np.random.default_rng(123)
    X9_class0 = rng_train.normal(loc=[-2, -2], scale=0.5, size=(50, 2))
    X9_class1 = rng_train.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    X9 = np.vstack([X9_class0, X9_class1])
    y9 = np.array([0.0] * 50 + [1.0] * 50)

    w9, b9 = train(X9, y9, lr=0.5, n_iters=500)

    # Check predictions
    probs9 = sigmoid(X9 @ w9 + b9)
    preds9 = (probs9 > 0.5).astype(float)
    accuracy = np.mean(preds9 == y9)
    if accuracy < 0.95:
        raise AssertionError(f"Training accuracy too low: {accuracy:.2%}")

    # --- Test 10: train output shapes ---
    if w9.shape != (2,):
        raise AssertionError(f"Trained w shape mismatch: {w9.shape} vs (2,)")
    if not isinstance(b9, (float, np.floating)):
        raise AssertionError(f"Trained b should be a scalar, got type {type(b9)}")

    # --- Test 11: train reduces loss ---
    X11 = rng.normal(size=(30, 5))
    y11 = rng.integers(0, 2, size=30).astype(float)
    w11_init = np.zeros(5)
    b11_init = 0.0

    initial_loss = compute_loss(X11, y11, w11_init, b11_init)
    w11, b11 = train(X11, y11, lr=0.1, n_iters=100)
    final_loss = compute_loss(X11, y11, w11, b11)

    if final_loss >= initial_loss:
        raise AssertionError(
            f"Training should reduce loss. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
        )

    # --- Test 12: gradient descent direction check ---
    X12 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y12 = np.array([1.0, 1.0, 1.0, 0.0])
    w12 = np.zeros(2)
    b12 = 0.0

    # One step of gradient descent
    dw12, db12 = compute_gradients(X12, y12, w12, b12)
    w12_new = w12 - 0.1 * dw12
    b12_new = b12 - 0.1 * db12

    loss_before = compute_loss(X12, y12, w12, b12)
    loss_after = compute_loss(X12, y12, w12_new, b12_new)

    if loss_after >= loss_before:
        raise AssertionError(
            f"One gradient step should reduce loss. Before: {loss_before:.4f}, After: {loss_after:.4f}"
        )
