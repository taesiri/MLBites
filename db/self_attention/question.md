# Self-Attention from Scratch

## Problem Statement

Implement **Scaled Dot-Product Self-Attention** from scratch using PyTorch. Self-attention is the core mechanism behind Transformers, allowing each position in a sequence to attend to all other positions.

This is a simpler version of multi-head attention with just a single head, which is great for understanding the fundamentals.

Your task is to:

1. Compute Query, Key, and Value projections
2. Calculate attention scores with proper scaling
3. Apply softmax to get attention weights
4. Compute the weighted sum of values

## Requirements

- Do **NOT** use `nn.MultiheadAttention`
- Apply proper scaling (divide by sqrt(d_k))
- Support optional attention mask
- Return both the output and attention weights

## Function Signature

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (batch, seq_len, d_k)
        key: Key tensor of shape (batch, seq_len, d_k)
        value: Value tensor of shape (batch, seq_len, d_v)
        mask: Optional mask tensor
        
    Returns:
        Tuple of (output, attention_weights)
        - output: (batch, seq_len, d_v)
        - attention_weights: (batch, seq_len, seq_len)
    """
    pass


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        """Initialize self-attention layer."""
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        pass
```

## Example

```python
import torch

# Basic attention computation
batch_size, seq_len, d_k = 2, 5, 64
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"Output shape: {output.shape}")           # (2, 5, 64)
print(f"Attention shape: {attn_weights.shape}")  # (2, 5, 5)
print(f"Attention row sum: {attn_weights[0, 0].sum()}")  # ~1.0
```

## Hints

- Attention formula: `Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`
- Use `torch.matmul` for batched matrix multiplication
- `K.transpose(-2, -1)` transposes the last two dimensions
- For masking, set masked positions to -inf before softmax
