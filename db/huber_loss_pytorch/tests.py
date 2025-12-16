from __future__ import annotations

from types import ModuleType

import torch


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "huber_loss"):
        raise AssertionError("Candidate must define function `huber_loss`.")

    huber_loss = candidate.huber_loss
    dtype = torch.float64

    # --- test 1: small residuals (all within delta) ---
    predictions = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    targets = torch.tensor([1.2, 1.8, 3.1], dtype=dtype)
    delta = 1.0
    # residuals = [-0.2, 0.2, -0.1]
    # huber = 0.5 * r^2 = [0.02, 0.02, 0.005]
    # mean = 0.015
    expected = torch.tensor(0.015, dtype=dtype)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 1 failed: small residuals")

    # --- test 2: large residuals (all outside delta) ---
    predictions = torch.tensor([0.0, 5.0], dtype=dtype)
    targets = torch.tensor([3.0, 0.0], dtype=dtype)
    delta = 1.0
    # residuals = [-3.0, 5.0]
    # huber = delta * (|r| - 0.5 * delta) = [1.0 * 2.5, 1.0 * 4.5] = [2.5, 4.5]
    # mean = 3.5
    expected = torch.tensor(3.5, dtype=dtype)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 2 failed: large residuals")

    # --- test 3: mixed residuals ---
    predictions = torch.tensor([1.0, 0.0], dtype=dtype)
    targets = torch.tensor([1.5, 3.0], dtype=dtype)
    delta = 1.0
    # residual[0] = -0.5, |r| <= delta -> 0.5 * 0.25 = 0.125
    # residual[1] = -3.0, |r| > delta -> 1.0 * (3.0 - 0.5) = 2.5
    # mean = 1.3125
    expected = torch.tensor(1.3125, dtype=dtype)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 3 failed: mixed residuals")

    # --- test 4: exact boundary (|residual| == delta) ---
    predictions = torch.tensor([0.0], dtype=dtype)
    targets = torch.tensor([1.0], dtype=dtype)
    delta = 1.0
    # residual = -1.0, |r| == delta
    # quadratic: 0.5 * 1.0 = 0.5
    # linear: 1.0 * (1.0 - 0.5) = 0.5
    # Both should give 0.5 (continuity)
    expected = torch.tensor(0.5, dtype=dtype)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 4 failed: boundary case")

    # --- test 5: custom delta ---
    predictions = torch.tensor([0.0, 0.0], dtype=dtype)
    targets = torch.tensor([0.5, 2.0], dtype=dtype)
    delta = 0.5
    # residual[0] = -0.5, |r| == delta -> quadratic: 0.5 * 0.25 = 0.125
    # residual[1] = -2.0, |r| > delta -> linear: 0.5 * (2.0 - 0.25) = 0.875
    # mean = 0.5
    expected = torch.tensor(0.5, dtype=dtype)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 5 failed: custom delta")

    # --- test 6: match PyTorch's HuberLoss ---
    torch.manual_seed(42)
    predictions = torch.randn(10, 5, dtype=dtype)
    targets = torch.randn(10, 5, dtype=dtype)
    delta = 1.0

    # PyTorch's HuberLoss with delta and reduction='mean' is equivalent
    torch_loss = torch.nn.HuberLoss(reduction="mean", delta=delta)
    expected = torch_loss(predictions, targets)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 6 failed: mismatch with PyTorch HuberLoss")

    # --- test 7: different delta with PyTorch comparison ---
    delta = 2.0
    torch_loss = torch.nn.HuberLoss(reduction="mean", delta=delta)
    expected = torch_loss(predictions, targets)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 7 failed: mismatch with delta=2.0")

    # --- test 8: zero residual ---
    predictions = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    targets = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    delta = 1.0
    expected = torch.tensor(0.0, dtype=dtype)
    result = huber_loss(predictions, targets, delta)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 8 failed: zero residual")

