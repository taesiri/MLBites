from __future__ import annotations

import torch
import torch.nn as nn


class CachedMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        """Multi-head self-attention with KV caching for autoregressive generation.

        Args:
            embed_dim: Model dimension E. Must be divisible by num_heads.
            num_heads: Number of attention heads.
            bias: Whether to use bias in the linear projections.
        """
        super().__init__()
        # TODO: store embed_dim, num_heads, head_dim (embed_dim // num_heads)
        # TODO: compute and store scale = head_dim ** -0.5
        # TODO: create fused QKV projection: nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        #       name it self.qkv_proj
        # TODO: create output projection: nn.Linear(embed_dim, embed_dim, bias=bias)
        #       name it self.out_proj
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run multi-head self-attention with KV caching.

        Args:
            x: Input tensor of shape (B, T, E) where T is the number of new tokens.
            kv_cache: Optional tuple (cached_k, cached_v), each of shape
                (B, num_heads, S, head_dim) where S is the cached sequence length.

        Returns:
            output: Tensor of shape (B, T, E) — attention output for new tokens.
            new_cache: Tuple (new_k, new_v), each of shape (B, num_heads, S+T, head_dim).
        """
        # TODO: get B, T, E from x.shape
        # TODO: compute Q, K, V for new tokens via qkv_proj
        #       reshape to (B, T, 3, num_heads, head_dim)
        #       permute to (3, B, num_heads, T, head_dim)
        #       unpack q, k, v

        # TODO: if kv_cache is not None:
        #       concatenate cached_k with k along sequence dimension
        #       concatenate cached_v with v along sequence dimension

        # TODO: create new_cache = (k, v) to return

        # TODO: compute attention scores: q @ k.transpose(-2, -1)
        # TODO: scale by self.scale

        # TODO: apply causal mask:
        #       - compute cache_len = total_len - T
        #       - for each query position i (0 to T-1), its absolute position is cache_len + i
        #       - it can only attend to key positions 0 to cache_len + i
        #       - mask positions where key_pos > query_abs_pos with -inf

        # TODO: softmax over last dimension
        # TODO: compute output: attn @ v
        # TODO: merge heads: transpose and reshape to (B, T, E)
        # TODO: apply output projection
        # TODO: return (output, new_cache)
        raise NotImplementedError


