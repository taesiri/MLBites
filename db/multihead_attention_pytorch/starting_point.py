from __future__ import annotations

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
            embed_dim: Model dimension E. Must be divisible by num_heads.
            num_heads: Number of attention heads H.
            dropout_p: Dropout probability applied to attention weights.
            bias: Whether to use bias in the linear projections.

        Notes:
            - This is intentionally minimal and interview-friendly.
            - The module should implement scaled dot-product self-attention:
              Q = x Wq, K = x Wk, V = x Wv, then softmax(QK^T / sqrt(d)) V.
        """
        super().__init__()
        # TODO: set self.embed_dim, self.num_heads, self.head_dim (E // H)
        # TODO: create one projection for QKV: nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # TODO: create output projection: nn.Linear(embed_dim, embed_dim, bias=bias)
        # TODO: create dropout module for attention weights: nn.Dropout(dropout_p)
        raise NotImplementedError

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
        # TODO: project to q, k, v (shape: B, T, E each)
        # TODO: reshape into heads: (B, H, T, D) where D = E // H
        # TODO: compute scores = (q @ k^T) / sqrt(D) with shape (B, H, T, T)
        # TODO: apply key_padding_mask by setting masked key positions to -inf
        # TODO: softmax over last dim to get attention weights
        # TODO: apply dropout to attention weights
        # TODO: compute output = attn @ v, then merge heads back to (B, T, E)
        # TODO: apply out_proj and return
        raise NotImplementedError


