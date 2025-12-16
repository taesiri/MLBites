from __future__ import annotations

import math
from types import ModuleType

import torch
import torch.nn as nn


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def _make_reference_torch_mha(
    *, embed_dim: int, num_heads: int, bias: bool, candidate_mha: nn.Module
) -> nn.MultiheadAttention:
    ref = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=bias,
        batch_first=True,
    )

    # Copy projections so outputs should match exactly (dropout=0.0).
    with torch.no_grad():
        ref.in_proj_weight.copy_(candidate_mha.qkv_proj.weight)
        if bias:
            ref.in_proj_bias.copy_(candidate_mha.qkv_proj.bias)
        ref.out_proj.weight.copy_(candidate_mha.out_proj.weight)
        if bias:
            ref.out_proj.bias.copy_(candidate_mha.out_proj.bias)

    return ref


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "SimpleMultiheadAttention"):
        raise AssertionError("Candidate must define class `SimpleMultiheadAttention`.")

    torch.manual_seed(0)
    dtype = torch.float64

    # --- test 1: match torch.nn.MultiheadAttention (no mask) ---
    embed_dim = 8
    num_heads = 2

    m = candidate.SimpleMultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout_p=0.0, bias=True
    ).to(dtype=dtype)
    m.eval()

    ref = _make_reference_torch_mha(
        embed_dim=embed_dim, num_heads=num_heads, bias=True, candidate_mha=m
    ).to(dtype=dtype)
    ref.eval()

    x = torch.randn(3, 5, embed_dim, dtype=dtype)
    y = m(x)
    y_ref, _ = ref(x, x, x, need_weights=False)

    _assert_allclose(y, y_ref, atol=1e-12, rtol=1e-12, msg="Output mismatch vs torch MHA.")

    # --- test 2: match torch.nn.MultiheadAttention with key_padding_mask ---
    key_padding_mask = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, True, True, True],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    y2 = m(x, key_padding_mask=key_padding_mask)
    y2_ref, _ = ref(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
    _assert_allclose(
        y2, y2_ref, atol=1e-12, rtol=1e-12, msg="Mask behavior mismatch vs torch MHA."
    )

    # --- test 3: deterministic identity-projection example (1 head) ---
    m2 = candidate.SimpleMultiheadAttention(
        embed_dim=2, num_heads=1, dropout_p=0.0, bias=False
    ).to(dtype=dtype)
    m2.eval()

    with torch.no_grad():
        m2.qkv_proj.weight.zero_()
        # Q
        m2.qkv_proj.weight[0, 0] = 1.0
        m2.qkv_proj.weight[1, 1] = 1.0
        # K
        m2.qkv_proj.weight[2, 0] = 1.0
        m2.qkv_proj.weight[3, 1] = 1.0
        # V
        m2.qkv_proj.weight[4, 0] = 1.0
        m2.qkv_proj.weight[5, 1] = 1.0

        m2.out_proj.weight.copy_(torch.eye(2, dtype=dtype))

    x2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=dtype)
    y3 = m2(x2)

    a = 1.0 / math.sqrt(2.0)
    w = math.exp(a) / (math.exp(a) + 1.0)
    expected = torch.tensor([[[w, 1.0 - w], [1.0 - w, w]]], dtype=dtype)
    _assert_allclose(
        y3,
        expected,
        atol=1e-12,
        rtol=1e-12,
        msg="Identity-projection example output mismatch.",
    )

    # --- test 4: mask ignores a key/value position (padding the 2nd token) ---
    mask2 = torch.tensor([[False, True]], dtype=torch.bool)
    y4 = m2(x2, key_padding_mask=mask2)
    expected2 = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=dtype)
    _assert_allclose(
        y4, expected2, atol=1e-12, rtol=1e-12, msg="Padding mask example mismatch."
    )

    # --- test 5: gradients flow back to input ---
    xg = torch.randn(2, 4, embed_dim, dtype=dtype, requires_grad=True)
    yg = m(xg).sum()
    yg.backward()
    if xg.grad is None:
        raise AssertionError("Expected non-None gradient for input.")
    if not torch.isfinite(xg.grad).all():
        raise AssertionError("Found non-finite values in input gradient.")


