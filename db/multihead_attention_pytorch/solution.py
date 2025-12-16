from __future__ import annotations

import math

import torch
import torch.nn as nn


class SimpleMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        """A minimal multi-head self-attention module (batch-first).

        Args:
            embed_dim: Model dimension E. Assumed divisible by num_heads.
            num_heads: Number of attention heads H.
            dropout_p: Dropout probability applied to attention weights.
            bias: Whether to use bias in the linear projections.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run multi-head self-attention.

        Args:
            x: Input tensor of shape (B, T, E).
            key_padding_mask: Optional bool tensor of shape (B, T).
                True means "padding" (ignore that position as a key/value).

        Returns:
            Tensor of shape (B, T, E).
        """
        bsz, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3E)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, E) -> (B, H, T, D)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(bsz, seq_len, self.num_heads, self.head_dim)
            return t.transpose(1, 2)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q * scale) @ k.transpose(-2, -1)  # (B, H, T, T)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(out)


