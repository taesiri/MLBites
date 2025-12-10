"""
Group Query Attention (GQA) - Starting Point

Implement Group Query Attention from scratch using PyTorch.
Fill in the TODO sections to complete the implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupQueryAttention(nn.Module):
    """Group Query Attention mechanism used in LLaMA 2 and other modern LLMs."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        num_kv_heads: int,
        dropout: float = 0.0
    ):
        """
        Initialize Group Query Attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of query heads
            num_kv_heads: Number of key-value heads (must divide num_heads evenly)
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # TODO: Create query projection (projects to num_heads * head_dim)
        
        # TODO: Create key projection (projects to num_kv_heads * head_dim)
        
        # TODO: Create value projection (projects to num_kv_heads * head_dim)
        
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
        Compute group query attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor of shape (batch, seq_len, embed_dim)
            value: Value tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # TODO: Project inputs to Q, K, V
        # Q: (batch, seq_len, num_heads * head_dim)
        # K, V: (batch, seq_len, num_kv_heads * head_dim)
        
        # TODO: Reshape Q to (batch, num_heads, seq_len, head_dim)
        
        # TODO: Reshape K, V to (batch, num_kv_heads, seq_len, head_dim)
        
        # TODO: Expand K, V to match the number of query heads
        # Hint: Use repeat_interleave to repeat each KV head for its query group
        # K, V should become (batch, num_heads, seq_len, head_dim)
        
        # TODO: Compute scaled dot-product attention
        # scores = (Q @ K^T) / sqrt(d_k)
        
        # TODO: Apply mask if provided
        
        # TODO: Apply softmax and dropout
        
        # TODO: Apply attention weights to values
        
        # TODO: Reshape and apply output projection
        
        pass  # Return (output, attention_weights)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create GQA layer with 8 query heads and 2 KV heads
    gqa = GroupQueryAttention(
        embed_dim=512, 
        num_heads=8, 
        num_kv_heads=2
    )
    
    # Generate random input
    batch_size, seq_len, embed_dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Compute attention
    output, attn_weights = gqa(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Number of query heads: 8, Number of KV heads: 2")
    print(f"Each KV head serves {8 // 2} query heads")
