"""
Group Query Attention (GQA) - Solution

Complete implementation of Group Query Attention from scratch using PyTorch.
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
        
        # Query projection (projects to num_heads * head_dim)
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim)
        
        # Key projection (projects to num_kv_heads * head_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        
        # Value projection (projects to num_kv_heads * head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
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
        
        # Project inputs to Q, K, V
        Q = self.q_proj(query)  # (batch, seq_len, num_heads * head_dim)
        K = self.k_proj(key)    # (batch, seq_len, num_kv_heads * head_dim)
        V = self.v_proj(value)  # (batch, seq_len, num_kv_heads * head_dim)
        
        # Reshape Q to (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape K, V to (batch, num_kv_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand K, V to match the number of query heads
        # Repeat each KV head for its query group
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)  # (batch, num_heads, seq_len, head_dim)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)  # (batch, num_heads, seq_len, head_dim)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights


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
