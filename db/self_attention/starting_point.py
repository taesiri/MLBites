"""
Self-Attention from Scratch - Starting Point

Implement scaled dot-product self-attention from scratch.
Fill in the TODO sections to complete the implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: Query tensor of shape (batch, seq_len, d_k)
        key: Key tensor of shape (batch, seq_len, d_k)
        value: Value tensor of shape (batch, seq_len, d_v)
        mask: Optional mask tensor
        
    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = query.size(-1)
    
    # TODO: Compute attention scores
    # scores = Q @ K^T
    
    # TODO: Scale by sqrt(d_k)
    
    # TODO: Apply mask if provided
    # Set masked positions to -inf
    
    # TODO: Apply softmax to get attention weights
    
    # TODO: Apply attention weights to values
    
    pass


class SelfAttention(nn.Module):
    """Single-head self-attention layer."""
    
    def __init__(self, embed_dim: int):
        """
        Initialize self-attention layer.
        
        Args:
            embed_dim: Dimension of embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # TODO: Create linear projections for Q, K, V
        
        # TODO: Create output projection
        
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # TODO: Project input to Q, K, V
        
        # TODO: Compute attention
        
        # TODO: Apply output projection
        
        pass


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Mask tensor of shape (seq_len, seq_len)
        True values indicate positions to attend to
    """
    # TODO: Create lower triangular mask
    # Position i can only attend to positions <= i
    
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch_size, seq_len, d_k = 2, 5, 64
    
    # Test scaled dot-product attention
    print("Testing scaled dot-product attention...")
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention row sum (should be 1.0): {attn_weights[0, 0].sum():.4f}")
    
    # Test self-attention module
    print("\nTesting SelfAttention module...")
    self_attn = SelfAttention(embed_dim=64)
    x = torch.randn(batch_size, seq_len, 64)
    output, attn_weights = self_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
