from __future__ import annotations

import math

import torch


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    attn_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Basic multi-head attention forward pass (PyTorch).

    This is a minimal, interview-friendly reference implementation:
    - No bias terms
    - Supports boolean or additive attention masks
    - Supports boolean key padding masks (True means masked out)

    Args:
        q: (B, Tq, D)
        k: (B, Tk, D)
        v: (B, Tk, D)
        w_q, w_k, w_v, w_o: (D, D) projection matrices
        num_heads: H where D % H == 0
        attn_mask: optional (Tq, Tk) mask (bool or float)
        key_padding_mask: optional (B, Tk) bool mask where True means "padding"

    Returns:
        y: (B, Tq, D)
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must all have shape (B, T, D)")

    bq, tq, d = q.shape
    bk, tk, dk = k.shape
    bv, tv, dv = v.shape
    if bk != bq or bv != bq:
        raise ValueError("q, k, v must have the same batch size B")
    if tk != tv:
        raise ValueError("k and v must have the same sequence length Tk")
    if dk != d or dv != d:
        raise ValueError("q, k, v must have the same embedding dimension D")

    if num_heads <= 0:
        raise ValueError("num_heads must be >= 1")
    if d % num_heads != 0:
        raise ValueError(f"embedding dim D={d} must be divisible by num_heads={num_heads}")
    head_dim = d // num_heads

    for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v), ("w_o", w_o)]:
        if w.ndim != 2 or w.shape != (d, d):
            raise ValueError(f"{name} must have shape (D, D)=({d}, {d}), got {tuple(w.shape)}")

    # Project inputs: (B, T, D) -> (B, T, D)
    q_proj = q @ w_q.T
    k_proj = k @ w_k.T
    v_proj = v @ w_v.T

    # Split into heads: (B, T, D) -> (B, H, T, Dh)
    qh = q_proj.reshape(bq, tq, num_heads, head_dim).transpose(1, 2)
    kh = k_proj.reshape(bq, tk, num_heads, head_dim).transpose(1, 2)
    vh = v_proj.reshape(bq, tk, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention scores: (B, H, Tq, Tk)
    scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(head_dim)

    if attn_mask is not None:
        if attn_mask.shape != (tq, tk):
            raise ValueError(f"attn_mask must have shape (Tq, Tk)=({tq}, {tk})")
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype)[None, None, :, :]

    if key_padding_mask is not None:
        if key_padding_mask.shape != (bq, tk):
            raise ValueError(f"key_padding_mask must have shape (B, Tk)=({bq}, {tk})")
        if key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be a bool tensor")
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh  # (B, H, Tq, Dh)

    # Combine heads: (B, H, Tq, Dh) -> (B, Tq, D)
    out = out.transpose(1, 2).contiguous().reshape(bq, tq, d)

    # Final projection
    y = out @ w_o.T
    return y


