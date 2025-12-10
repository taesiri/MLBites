"""
Multi-Head Attention - Starting Point

Implement multi-head attention from scratch using PyTorch.
Fill in the TODO sections to complete the implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # TODO: Create linear projections for Q, K, V
        # Hint: Use nn.Linear to project input to query, key, value
        
        # TODO: Create output projection
        
        # TODO: Create dropout layer
        pass
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor of shape (batch, seq_len, embed_dim)
            value: Value tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # TODO: Project inputs to Q, K, V
        
        # TODO: Reshape to (batch, num_heads, seq_len, head_dim)
        
        # TODO: Compute scaled dot-product attention
        # scores = (Q @ K^T) / sqrt(d_k)
        
        # TODO: Apply mask if provided (set masked positions to -inf before softmax)
        
        # TODO: Apply softmax to get attention weights
        
        # TODO: Apply dropout to attention weights
        
        # TODO: Apply attention weights to values
        
        # TODO: Reshape back to (batch, seq_len, embed_dim)
        
        # TODO: Apply output projection
        
        pass  # Return (output, attention_weights)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create multi-head attention layer
    mha = MultiHeadAttention(embed_dim=512, num_heads=8)
    
    # Generate random input (self-attention)
    batch_size, seq_len, embed_dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Compute self-attention
    output, attn_weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {attn_weights[0, 0, 0].sum():.4f}")
