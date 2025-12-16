from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            embed_dim: Model dimension D. Assumed divisible by num_heads.
            num_heads: Number of attention heads.
            dropout_p: Dropout probability applied to attention weights.
            bias: Whether to use bias in the linear projections.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

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
            x: Input tensor of shape (B, L, D).
            key_padding_mask: Optional bool tensor of shape (B, L).
                True means "padding" (ignore that position as a key/value).

        Returns:
            Tensor of shape (B, L, D).
        """
        B, L, D = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # [B, L, 3, n_heads, d_k]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, L, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, n_heads, L, d_k]

        # Attention scores
        scores = q @ k.transpose(-2, -1)  # [B, n_heads, L, L]
        scores = scores * self.scale

        # Apply mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, n_heads, L, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_proj(out)
        return out
