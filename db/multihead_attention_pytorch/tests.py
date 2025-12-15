from __future__ import annotations

from types import ModuleType

import torch


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{msg}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = float((a - b).abs().max().item())
        raise AssertionError(f"{msg}: not close (max_abs={max_abs:.3e})")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module."""
    if not hasattr(candidate, "multihead_attention"):
        raise AssertionError("candidate must define multihead_attention")

    torch.manual_seed(0)

    # --------------------
    # Example 1 (exact): uniform attention when all scores are zero
    b, tq, tk, d, h = 1, 2, 2, 4, 2
    q = torch.zeros(b, tq, d)
    k = torch.zeros(b, tk, d)
    v = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    w_q = torch.eye(d)
    w_k = torch.eye(d)
    w_v = torch.eye(d)
    w_o = torch.eye(d)

    y = candidate.multihead_attention(
        q, k, v, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, num_heads=h
    )
    expected = torch.tensor([[[3.0, 4.0, 5.0, 6.0], [3.0, 4.0, 5.0, 6.0]]])
    _assert_close(y, expected, atol=0.0, rtol=0.0, msg="example 1")

    # --------------------
    # Example 2: causal mask changes allowed keys
    b, t, d, h = 1, 3, 2, 1
    q = torch.zeros(b, t, d)
    k = torch.zeros(b, t, d)
    v = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    w_q = torch.eye(d)
    w_k = torch.eye(d)
    w_v = torch.eye(d)
    w_o = torch.eye(d)
    attn_mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)

    y = candidate.multihead_attention(
        q,
        k,
        v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=h,
        attn_mask=attn_mask,
    )
    expected = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [2.0 / 3.0, 2.0 / 3.0]]])
    _assert_close(y, expected, atol=1e-6, rtol=0.0, msg="example 2 (causal mask)")

    # --------------------
    # Validation: D must be divisible by num_heads
    try:
        candidate.multihead_attention(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
            w_q=torch.zeros(3, 3),
            w_k=torch.zeros(3, 3),
            w_v=torch.zeros(3, 3),
            w_o=torch.zeros(3, 3),
            num_heads=2,
        )
    except Exception:
        pass
    else:
        raise AssertionError("expected an error when D is not divisible by num_heads")

    # --------------------
    # Numerical check: compare to torch.nn.MultiheadAttention (bias=False)
    b, tq, tk, d, h = 2, 4, 5, 8, 2
    q = torch.randn(b, tq, d)
    k = torch.randn(b, tk, d)
    v = torch.randn(b, tk, d)
    w_q = torch.randn(d, d) * 0.1
    w_k = torch.randn(d, d) * 0.1
    w_v = torch.randn(d, d) * 0.1
    w_o = torch.randn(d, d) * 0.1

    # Some keys are padding (True means masked out)
    key_padding_mask = torch.zeros(b, tk, dtype=torch.bool)
    key_padding_mask[0, -1] = True
    key_padding_mask[1, -2:] = True

    # Random boolean attention mask (shape Tq x Tk)
    attn_mask = torch.zeros(tq, tk, dtype=torch.bool)
    attn_mask[0, 0] = True
    attn_mask[2, 1] = True

    y = candidate.multihead_attention(
        q,
        k,
        v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=h,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )

    ref = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([w_q, w_k, w_v], dim=0))
        ref.out_proj.weight.copy_(w_o)

    y_ref, _ = ref(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
    _assert_close(y, y_ref, atol=1e-5, rtol=1e-4, msg="compare to torch.nn.MultiheadAttention")


