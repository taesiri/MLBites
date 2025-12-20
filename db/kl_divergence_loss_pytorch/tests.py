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
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "KLDivLoss"):
        raise AssertionError("Candidate must define class `KLDivLoss`.")

    KLDivLoss = candidate.KLDivLoss
    dtype = torch.float64

    # --- test 1: identical distributions (KL should be 0) ---
    p = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype)
    input_log = torch.log(p)
    target = p.clone()
    loss_fn = KLDivLoss(reduction="batchmean")
    result = loss_fn(input_log, target)
    expected = torch.tensor(0.0, dtype=dtype)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 1 failed: identical distributions")

    # --- test 2: simple KL divergence with batchmean ---
    input_log = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=dtype))
    target = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=dtype)
    loss_fn = KLDivLoss(reduction="batchmean")
    # Compare with PyTorch's implementation
    torch_loss = torch.nn.KLDivLoss(reduction="batchmean")
    expected = torch_loss(input_log, target)
    result = loss_fn(input_log, target)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 2 failed: batchmean reduction")

    # --- test 3: sum reduction ---
    input_log = torch.log(torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=dtype))
    target = torch.tensor([[0.4, 0.6], [0.5, 0.5]], dtype=dtype)
    loss_fn = KLDivLoss(reduction="sum")
    torch_loss = torch.nn.KLDivLoss(reduction="sum")
    expected = torch_loss(input_log, target)
    result = loss_fn(input_log, target)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 3 failed: sum reduction")

    # --- test 4: mean reduction ---
    input_log = torch.log(torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=dtype))
    target = torch.tensor([[0.4, 0.6], [0.5, 0.5]], dtype=dtype)
    loss_fn = KLDivLoss(reduction="mean")
    torch_loss = torch.nn.KLDivLoss(reduction="mean")
    expected = torch_loss(input_log, target)
    result = loss_fn(input_log, target)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 4 failed: mean reduction")

    # --- test 5: no reduction ---
    input_log = torch.log(torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=dtype))
    target = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype)
    loss_fn = KLDivLoss(reduction="none")
    torch_loss = torch.nn.KLDivLoss(reduction="none")
    expected = torch_loss(input_log, target)
    result = loss_fn(input_log, target)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 5 failed: no reduction")

    # --- test 6: target with zeros (0 * log(0) should be 0) ---
    input_log = torch.log(torch.tensor([[0.5, 0.5]], dtype=dtype))
    target = torch.tensor([[1.0, 0.0]], dtype=dtype)  # second element is 0
    loss_fn = KLDivLoss(reduction="sum")
    torch_loss = torch.nn.KLDivLoss(reduction="sum")
    expected = torch_loss(input_log, target)
    result = loss_fn(input_log, target)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 6 failed: target with zeros")

    # --- test 7: larger batch ---
    torch.manual_seed(42)
    logits = torch.randn(8, 10, dtype=dtype)
    input_log = torch.log_softmax(logits, dim=-1)
    target = torch.softmax(torch.randn(8, 10, dtype=dtype), dim=-1)
    
    for reduction in ["none", "sum", "mean", "batchmean"]:
        loss_fn = KLDivLoss(reduction=reduction)
        torch_loss = torch.nn.KLDivLoss(reduction=reduction)
        expected = torch_loss(input_log, target)
        result = loss_fn(input_log, target)
        _assert_allclose(result, expected, atol=1e-9, rtol=1e-9, 
                        msg=f"Test 7 failed: larger batch with reduction={reduction}")

    # --- test 8: 3D input (e.g., sequence data) ---
    torch.manual_seed(123)
    logits = torch.randn(4, 5, 6, dtype=dtype)
    input_log = torch.log_softmax(logits, dim=-1)
    target = torch.softmax(torch.randn(4, 5, 6, dtype=dtype), dim=-1)
    
    loss_fn = KLDivLoss(reduction="batchmean")
    torch_loss = torch.nn.KLDivLoss(reduction="batchmean")
    expected = torch_loss(input_log, target)
    result = loss_fn(input_log, target)
    _assert_allclose(result, expected, atol=1e-9, rtol=1e-9, msg="Test 8 failed: 3D input")

    # --- test 9: verify it's an nn.Module ---
    loss_fn = KLDivLoss(reduction="mean")
    if not isinstance(loss_fn, torch.nn.Module):
        raise AssertionError("Test 9 failed: KLDivLoss must inherit from nn.Module")

    # --- test 10: verify forward method exists ---
    if not hasattr(loss_fn, "forward"):
        raise AssertionError("Test 10 failed: KLDivLoss must have a forward method")




