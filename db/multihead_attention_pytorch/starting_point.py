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
            embed_dim: Model dimension D. Must be divisible by num_heads.
            num_heads: Number of attention heads.
            dropout_p: Dropout probability applied to attention weights.
            bias: Whether to use bias in the linear projections.
        """
        super().__init__()
        # TODO: store embed_dim, num_heads, head_dim (embed_dim // num_heads)
        # TODO: compute and store scale = head_dim ** -0.5
        # TODO: create fused QKV projection: nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        #       name it self.qkv_proj
        # TODO: create output projection: nn.Linear(embed_dim, embed_dim, bias=bias)
        #       name it self.out_proj
        # TODO: create dropout: nn.Dropout(dropout_p), name it self.attn_dropout
        raise NotImplementedError

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
        # TODO: get B, L, D from x.shape
        # TODO: fused QKV projection -> [B, L, 3*D]
        # TODO: reshape to [B, L, 3, n_heads, d_k]
        # TODO: permute to [3, B, n_heads, L, d_k]
        # TODO: unpack q, k, v (each [B, n_heads, L, d_k])
        # TODO: compute attention scores: q @ k.transpose(-2, -1) -> [B, n_heads, L, L]
        # TODO: scale scores by self.scale
        # TODO: apply key_padding_mask (set masked positions to -inf)
        # TODO: softmax over last dim
        # TODO: apply dropout
        # TODO: apply attention to values: attn @ v -> [B, n_heads, L, d_k]
        # TODO: merge heads: transpose + view -> [B, L, D]
        # TODO: final projection and return
        raise NotImplementedError
