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
    for fn_name in ("softmax", "log_softmax", "cross_entropy"):
        if not hasattr(candidate, fn_name):
            raise AssertionError(f"Candidate must define function `{fn_name}`.")

    softmax = candidate.softmax
    log_softmax = candidate.log_softmax
    cross_entropy = candidate.cross_entropy
    dtype = torch.float64

    # ==================== SOFTMAX TESTS ====================

    # --- test softmax 1: basic 2D ---
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=dtype)
    result = softmax(x, dim=-1)
    expected = torch.nn.functional.softmax(x, dim=-1)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Softmax test 1 failed: basic 2D")

    # Check rows sum to 1
    row_sums = result.sum(dim=-1)
    expected_sums = torch.ones(2, dtype=dtype)
    _assert_allclose(row_sums, expected_sums, atol=1e-10, rtol=1e-10, msg="Softmax test 1 failed: rows don't sum to 1")

    # --- test softmax 2: numerical stability with large values ---
    x = torch.tensor([[1000.0, 1001.0, 1002.0]], dtype=dtype)
    result = softmax(x, dim=-1)
    if torch.isnan(result).any() or torch.isinf(result).any():
        raise AssertionError("Softmax test 2 failed: produced nan/inf with large logits")
    expected = torch.nn.functional.softmax(x, dim=-1)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Softmax test 2 failed: large values")

    # --- test softmax 3: dim=0 ---
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype)
    result = softmax(x, dim=0)
    expected = torch.nn.functional.softmax(x, dim=0)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Softmax test 3 failed: dim=0")

    # --- test softmax 4: 3D tensor ---
    torch.manual_seed(42)
    x = torch.randn(2, 3, 4, dtype=dtype)
    result = softmax(x, dim=-1)
    expected = torch.nn.functional.softmax(x, dim=-1)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Softmax test 4 failed: 3D tensor")

    # --- test softmax 5: 1D tensor ---
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    result = softmax(x, dim=0)
    expected = torch.nn.functional.softmax(x, dim=0)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Softmax test 5 failed: 1D tensor")

    # --- test softmax 6: negative values ---
    x = torch.tensor([[-1000.0, -999.0, -998.0]], dtype=dtype)
    result = softmax(x, dim=-1)
    if torch.isnan(result).any() or torch.isinf(result).any():
        raise AssertionError("Softmax test 6 failed: produced nan/inf with large negative logits")
    expected = torch.nn.functional.softmax(x, dim=-1)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Softmax test 6 failed: negative values")

    # ==================== LOG_SOFTMAX TESTS ====================

    # --- test log_softmax 1: basic 2D ---
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    result = log_softmax(x, dim=-1)
    expected = torch.nn.functional.log_softmax(x, dim=-1)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="LogSoftmax test 1 failed: basic 2D")

    # --- test log_softmax 2: numerical stability ---
    x = torch.tensor([[1000.0, 1001.0, 1002.0]], dtype=dtype)
    result = log_softmax(x, dim=-1)
    if torch.isnan(result).any() or torch.isinf(result).any():
        raise AssertionError("LogSoftmax test 2 failed: produced nan/inf with large logits")
    expected = torch.nn.functional.log_softmax(x, dim=-1)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="LogSoftmax test 2 failed: large values")

    # --- test log_softmax 3: exp(log_softmax) equals softmax ---
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
    log_result = log_softmax(x, dim=-1)
    softmax_result = softmax(x, dim=-1)
    exp_log_result = torch.exp(log_result)
    _assert_allclose(exp_log_result, softmax_result, atol=1e-10, rtol=1e-10, 
                     msg="LogSoftmax test 3 failed: exp(log_softmax) != softmax")

    # --- test log_softmax 4: dim=0 ---
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    result = log_softmax(x, dim=0)
    expected = torch.nn.functional.log_softmax(x, dim=0)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="LogSoftmax test 4 failed: dim=0")

    # --- test log_softmax 5: 3D tensor ---
    torch.manual_seed(123)
    x = torch.randn(2, 3, 4, dtype=dtype)
    result = log_softmax(x, dim=-1)
    expected = torch.nn.functional.log_softmax(x, dim=-1)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="LogSoftmax test 5 failed: 3D tensor")

    # --- test log_softmax 6: all values are negative (log-probs) ---
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    result = log_softmax(x, dim=-1)
    if (result > 0).any():
        raise AssertionError("LogSoftmax test 6 failed: log-softmax should be <= 0")

    # ==================== CROSS_ENTROPY TESTS ====================

    # --- test cross_entropy 1: basic case ---
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], dtype=dtype)
    targets = torch.tensor([0, 1])
    result = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="CrossEntropy test 1 failed: basic case")

    # --- test cross_entropy 2: perfect prediction ---
    logits = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]], dtype=dtype)
    targets = torch.tensor([0, 1])
    result = cross_entropy(logits, targets)
    # Loss should be very close to 0
    if result > 1e-5:
        raise AssertionError(f"CrossEntropy test 2 failed: perfect prediction should have near-zero loss, got {result}")

    # --- test cross_entropy 3: wrong prediction ---
    logits = torch.tensor([[100.0, 0.0, 0.0]], dtype=dtype)
    targets = torch.tensor([1])  # Model confident in class 0, but truth is class 1
    result = cross_entropy(logits, targets)
    # Loss should be high (around 100)
    if result < 50.0:
        raise AssertionError(f"CrossEntropy test 3 failed: wrong prediction should have high loss, got {result}")

    # --- test cross_entropy 4: numerical stability ---
    logits = torch.tensor([[1000.0, 1001.0, 1002.0]], dtype=dtype)
    targets = torch.tensor([2])
    result = cross_entropy(logits, targets)
    if torch.isnan(result) or torch.isinf(result):
        raise AssertionError("CrossEntropy test 4 failed: produced nan/inf with large logits")
    expected = torch.nn.functional.cross_entropy(logits, targets)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="CrossEntropy test 4 failed: large values")

    # --- test cross_entropy 5: batch of 1 ---
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    targets = torch.tensor([0])
    result = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="CrossEntropy test 5 failed: batch of 1")

    # --- test cross_entropy 6: larger batch ---
    torch.manual_seed(456)
    logits = torch.randn(32, 10, dtype=dtype)
    targets = torch.randint(0, 10, (32,))
    result = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="CrossEntropy test 6 failed: larger batch")

    # --- test cross_entropy 7: two classes (binary) ---
    logits = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    targets = torch.tensor([0, 0, 1])
    result = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="CrossEntropy test 7 failed: binary classification")

    # --- test cross_entropy 8: all same class ---
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=dtype)
    targets = torch.tensor([2, 2, 2])
    result = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="CrossEntropy test 8 failed: all same class")


