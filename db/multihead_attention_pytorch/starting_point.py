from __future__ import annotations

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

    Args:
        q: (B, Tq, D) queries
        k: (B, Tk, D) keys
        v: (B, Tk, D) values
        w_q: (D, D) query projection weight
        w_k: (D, D) key projection weight
        w_v: (D, D) value projection weight
        w_o: (D, D) output projection weight
        num_heads: number of attention heads H (must divide D)
        attn_mask: optional (Tq, Tk) mask
            - bool: True means "masked out"
            - float: additive mask (0 for allowed, -inf for disallowed)
        key_padding_mask: optional (B, Tk) bool mask where True means "padding" (masked out)

    Returns:
        y: (B, Tq, D) attention output
    """
    # TODO:
    # - Project q/k/v into head dimensions
    # - Compute scaled dot-product attention scores
    # - Apply attn_mask and key_padding_mask if provided
    # - Softmax over keys, compute weighted sum of values
    # - Concatenate heads and apply output projection
    raise NotImplementedError


