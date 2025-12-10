# Multi-Head Attention

## Problem Statement

Implement **Multi-Head Attention** from scratch using PyTorch. Multi-Head Attention is a key component of the Transformer architecture that allows the model to jointly attend to information from different representation subspaces at different positions.

Your task is to:

1. Implement scaled dot-product attention
2. Create a multi-head attention module with projections for Q, K, V
3. Combine the heads and apply the output projection

## Requirements

- Implement without using `nn.MultiheadAttention`
- Support configurable number of heads and embedding dimensions
- Apply proper scaling (divide by sqrt(d_k))
- Return both the attention output and attention weights

## Function Signature

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """Initialize multi-head attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        pass
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor of shape (batch, seq_len, embed_dim)
            value: Value tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        pass
```

## Example

```python
import torch

# Create multi-head attention layer
mha = MultiHeadAttention(embed_dim=512, num_heads=8)

# Generate random input (self-attention)
batch_size, seq_len, embed_dim = 2, 10, 512
x = torch.randn(batch_size, seq_len, embed_dim)

# Compute self-attention
output, attn_weights = mha(x, x, x)

print(f"Output shape: {output.shape}")        # (2, 10, 512)
print(f"Attention shape: {attn_weights.shape}") # (2, 8, 10, 10)
```

## Hints

- Split `embed_dim` across `num_heads`, so each head operates on `embed_dim // num_heads` dimensions
- Reshape tensors to (batch, num_heads, seq_len, head_dim) for parallel computation
- Remember to apply softmax along the key dimension (last dimension)
- Use `torch.matmul` for batched matrix multiplication
