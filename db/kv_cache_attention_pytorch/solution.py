from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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
        B, T, E = x.shape

        # Compute Q, K, V for new tokens
        qkv = self.qkv_proj(x)  # (B, T, 3*E)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, T, head_dim)

        # Concatenate with cached K, V if available
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # (B, num_heads, S+T, head_dim)
            v = torch.cat([cached_v, v], dim=2)

        # Update cache with new K, V
        new_cache = (k, v)

        # Compute attention scores: Q @ K^T
        # q: (B, num_heads, T, head_dim)
        # k: (B, num_heads, S+T, head_dim)
        scores = q @ k.transpose(-2, -1)  # (B, num_heads, T, S+T)
        scores = scores * self.scale

        # Apply causal mask: each query can only attend to positions <= its position
        # For cached setting: query at position i (0-indexed within T new tokens)
        # corresponds to absolute position S+i, and can attend to positions 0..S+i
        total_len = k.size(2)  # S + T
        cache_len = total_len - T  # S
        # Build causal mask for the T new queries attending to S+T keys
        # Query i (abs pos cache_len + i) can see keys 0..(cache_len + i)
        query_pos = torch.arange(T, device=x.device).unsqueeze(1) + cache_len  # (T, 1)
        key_pos = torch.arange(total_len, device=x.device).unsqueeze(0)  # (1, S+T)
        causal_mask = key_pos > query_pos  # (T, S+T), True where should mask
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax and attention
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, num_heads, T, head_dim)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.out_proj(out)

        return out, new_cache

