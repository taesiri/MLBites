# Group Query Attention (GQA)

## Problem Statement

Implement **Group Query Attention (GQA)** from scratch using PyTorch. GQA is an efficient attention mechanism used in modern LLMs like LLaMA 2 that interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).

In GQA:
- Query heads are grouped together
- Each group shares a single key-value head
- This reduces memory bandwidth during inference while maintaining quality

Your task is to:

1. Implement GQA where multiple query heads share the same key-value pair
2. Handle the case where `num_heads` is divisible by `num_kv_heads`
3. Properly broadcast K and V across query head groups

## Requirements

- Implement without using `nn.MultiheadAttention`
- Support configurable number of query heads and key-value heads
- When `num_kv_heads == num_heads`, this should behave like standard MHA
- When `num_kv_heads == 1`, this should behave like Multi-Query Attention

## Function Signature

```python
class GroupQueryAttention(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        num_kv_heads: int,
        dropout: float = 0.0
    ):
        """Initialize Group Query Attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of query heads
            num_kv_heads: Number of key-value heads (must divide num_heads)
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
        """Compute group query attention.
        
        Returns:
            Tuple of (output, attention_weights)
        """
        pass
```

## Example

```python
import torch

# Create GQA layer with 8 query heads and 2 KV heads
# Each KV head serves 4 query heads
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

print(f"Output shape: {output.shape}")        # (2, 10, 512)
print(f"Attention shape: {attn_weights.shape}") # (2, 8, 10, 10)
```

## Hints

- `num_heads` must be divisible by `num_kv_heads`
- K and V projections output `num_kv_heads * head_dim` dimensions
- Use `repeat_interleave` or `expand` to broadcast K and V to match query heads
- The attention computation is the same as MHA once K and V are expanded
