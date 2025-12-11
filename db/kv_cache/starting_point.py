"""
KV Cache for LLM Inference - Starting Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Key-Value cache for efficient autoregressive generation."""
    
    def __init__(self, max_batch_size: int, max_seq_len: int, 
                 num_layers: int, num_heads: int, head_dim: int, device: str = 'cpu'):
        """
        Pre-allocate cache tensors.
        """
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # TODO: Pre-allocate cache tensors for each layer
        # cache_k: (num_layers, batch, num_heads, max_seq_len, head_dim)
        # cache_v: (num_layers, batch, num_heads, max_seq_len, head_dim)
        
        # TODO: Track current sequence position
        pass
    
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> tuple:
        """
        Update cache with new key/value and return full cached K,V.
        
        Args:
            layer_idx: Which layer's cache to update
            key: New keys (batch, num_heads, seq_len, head_dim)
            value: New values (batch, num_heads, seq_len, head_dim)
        
        Returns:
            (cached_keys, cached_values) including the new ones
        """
        # TODO: Store new K,V at current position
        # TODO: Return full cached K,V up to current position
        pass
    
    def get(self, layer_idx: int) -> tuple:
        """Get cached K,V for a layer."""
        pass
    
    def clear(self):
        """Reset cache for new sequence."""
        pass


class CausalAttentionWithCache(nn.Module):
    """Causal self-attention with KV cache support."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # TODO: Q, K, V projections
        # TODO: Output projection
        pass
    
    def forward(self, x: torch.Tensor, cache: KVCache = None, 
                layer_idx: int = 0, start_pos: int = 0) -> torch.Tensor:
        """
        Forward with optional caching.
        
        Args:
            x: Input (batch, seq_len, embed_dim)
            cache: Optional KV cache
            layer_idx: Layer index for cache
            start_pos: Position offset for cached sequences
        """
        # TODO: Compute Q, K, V
        
        # TODO: If cache exists, update it and use cached K, V
        
        # TODO: Compute attention (handle causal masking correctly)
        
        pass


class TransformerBlockWithCache(nn.Module):
    """Transformer block with KV cache."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # TODO: Attention, MLP, LayerNorms
        pass
    
    def forward(self, x, cache=None, layer_idx=0, start_pos=0):
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch, num_heads, head_dim = 2, 8, 64
    embed_dim = num_heads * head_dim
    num_layers = 4
    
    # Create cache
    cache = KVCache(
        max_batch_size=batch,
        max_seq_len=128,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim
    )
    
    # Create attention layer
    attn = CausalAttentionWithCache(embed_dim, num_heads)
    
    # Prefill (first forward with prompt)
    prompt = torch.randn(batch, 10, embed_dim)
    out = attn(prompt, cache=cache, layer_idx=0, start_pos=0)
    print(f"Prefill output: {out.shape}")
    
    # Decode (generate one token at a time)
    for i in range(5):
        new_token = torch.randn(batch, 1, embed_dim)
        out = attn(new_token, cache=cache, layer_idx=0, start_pos=10 + i)
        print(f"Decode step {i}: {out.shape}")
