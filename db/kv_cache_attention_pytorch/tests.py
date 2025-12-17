from __future__ import annotations

from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def _reference_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Reference causal attention for validation.

    Args:
        q: (B, H, T_q, D)
        k: (B, H, T_kv, D)
        v: (B, H, T_kv, D)
        scale: scaling factor

    Returns:
        (B, H, T_q, D)
    """
    T_q = q.size(2)
    T_kv = k.size(2)
    scores = q @ k.transpose(-2, -1) * scale  # (B, H, T_q, T_kv)

    # Causal mask: query at position i can attend to keys 0..(T_kv - T_q + i)
    # This is for the case where q corresponds to the last T_q positions
    offset = T_kv - T_q
    query_pos = torch.arange(T_q, device=q.device).unsqueeze(1) + offset
    key_pos = torch.arange(T_kv, device=q.device).unsqueeze(0)
    mask = key_pos > query_pos
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = F.softmax(scores, dim=-1)
    return attn @ v


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "CachedMultiheadAttention"):
        raise AssertionError("Candidate must define class `CachedMultiheadAttention`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- Test 1: Basic shape check ---
    embed_dim = 8
    num_heads = 2
    head_dim = embed_dim // num_heads

    m = candidate.CachedMultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, bias=True
    ).to(dtype=dtype)
    m.eval()

    x1 = torch.randn(2, 5, embed_dim, dtype=dtype)
    out1, cache1 = m(x1, kv_cache=None)

    assert out1.shape == (2, 5, embed_dim), f"Expected output shape (2, 5, {embed_dim}), got {out1.shape}"
    assert cache1[0].shape == (2, num_heads, 5, head_dim), f"Expected cache K shape (2, {num_heads}, 5, {head_dim}), got {cache1[0].shape}"
    assert cache1[1].shape == (2, num_heads, 5, head_dim), f"Expected cache V shape (2, {num_heads}, 5, {head_dim}), got {cache1[1].shape}"

    # --- Test 2: Cache grows correctly ---
    x2 = torch.randn(2, 3, embed_dim, dtype=dtype)
    out2, cache2 = m(x2, kv_cache=cache1)

    assert out2.shape == (2, 3, embed_dim), f"Expected output shape (2, 3, {embed_dim}), got {out2.shape}"
    assert cache2[0].shape == (2, num_heads, 8, head_dim), f"Expected cache K shape (2, {num_heads}, 8, {head_dim}), got {cache2[0].shape}"
    assert cache2[1].shape == (2, num_heads, 8, head_dim), f"Expected cache V shape (2, {num_heads}, 8, {head_dim}), got {cache2[1].shape}"

    # --- Test 3: Cached K/V values are preserved ---
    # The first 5 positions in cache2 should match cache1
    _assert_allclose(
        cache2[0][:, :, :5, :], cache1[0], atol=1e-12, rtol=1e-12,
        msg="Cached K values were modified when they should be preserved."
    )
    _assert_allclose(
        cache2[1][:, :, :5, :], cache1[1], atol=1e-12, rtol=1e-12,
        msg="Cached V values were modified when they should be preserved."
    )

    # --- Test 4: Single-token generation matches incremental computation ---
    torch.manual_seed(123)
    m2 = candidate.CachedMultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, bias=True
    ).to(dtype=dtype)
    m2.eval()

    # Process 4 tokens one by one with caching
    x_full = torch.randn(1, 4, embed_dim, dtype=dtype)
    
    # Token by token with cache
    out_t0, cache = m2(x_full[:, 0:1, :], kv_cache=None)
    out_t1, cache = m2(x_full[:, 1:2, :], kv_cache=cache)
    out_t2, cache = m2(x_full[:, 2:3, :], kv_cache=cache)
    out_t3, cache = m2(x_full[:, 3:4, :], kv_cache=cache)
    out_incremental = torch.cat([out_t0, out_t1, out_t2, out_t3], dim=1)

    # Full sequence at once
    out_full, _ = m2(x_full, kv_cache=None)

    # Outputs should match (causal attention means each position sees the same context)
    _assert_allclose(
        out_full, out_incremental, atol=1e-10, rtol=1e-10,
        msg="Incremental cached outputs don't match full sequence output."
    )

    # --- Test 5: Verify causal masking is applied ---
    torch.manual_seed(456)
    m3 = candidate.CachedMultiheadAttention(
        embed_dim=4, num_heads=1, bias=False
    ).to(dtype=dtype)
    m3.eval()

    # Create input where we can verify causal masking
    x = torch.randn(1, 3, 4, dtype=dtype)
    out, cache = m3(x, kv_cache=None)

    # Manually compute what the output should be with causal masking
    with torch.no_grad():
        qkv = m3.qkv_proj(x)
        qkv = qkv.reshape(1, 3, 3, 1, 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = 4 ** -0.5
        expected = _reference_causal_attention(q, k, v, scale)
        expected = expected.transpose(1, 2).reshape(1, 3, 4)
        expected = m3.out_proj(expected)

    _assert_allclose(
        out, expected, atol=1e-10, rtol=1e-10,
        msg="Causal masking not applied correctly."
    )

    # --- Test 6: Gradients flow correctly ---
    xg = torch.randn(2, 3, embed_dim, dtype=dtype, requires_grad=True)
    out_g, _ = m(xg, kv_cache=None)
    loss = out_g.sum()
    loss.backward()

    if xg.grad is None:
        raise AssertionError("Expected non-None gradient for input.")
    if not torch.isfinite(xg.grad).all():
        raise AssertionError("Found non-finite values in input gradient.")

    # --- Test 7: Works with different batch sizes ---
    for batch_size in [1, 4, 8]:
        x_batch = torch.randn(batch_size, 5, embed_dim, dtype=dtype)
        out_batch, cache_batch = m(x_batch, kv_cache=None)
        assert out_batch.shape == (batch_size, 5, embed_dim)
        assert cache_batch[0].shape == (batch_size, num_heads, 5, head_dim)

