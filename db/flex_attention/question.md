# FlexAttention Patterns

## Problem Statement

Learn to use **FlexAttention** (PyTorch 2.5+) to implement various attention patterns efficiently. FlexAttention allows defining custom attention masks via score modification functions, enabling patterns like sliding window, causal, and more.

Your task is to:

1. Understand the FlexAttention API
2. Implement common attention patterns
3. Create custom score modification functions
4. Combine patterns with AND/OR operations

## FlexAttention API

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def score_mod(score, batch, head, q_idx, kv_idx):
    """Modify attention scores. Return modified score or -inf to mask."""
    return score

output = flex_attention(query, key, value, score_mod=score_mod)
```

## Common Attention Patterns

```python
# Causal (autoregressive)
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Sliding window
def sliding_window(window_size):
    def mask(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= window_size
    return mask

# Prefix LM (attend to prefix, then causal)
def prefix_lm(prefix_length):
    def mask(b, h, q_idx, kv_idx):
        return (kv_idx < prefix_length) | (q_idx >= kv_idx)
    return mask

# Document masking (same document only)
def document_mask(doc_ids):
    def mask(b, h, q_idx, kv_idx):
        return doc_ids[q_idx] == doc_ids[kv_idx]
    return mask
```

## Function Signature

```python
def create_causal_mask() -> Callable:
    """Return causal attention mask function."""
    pass

def create_sliding_window_mask(window_size: int) -> Callable:
    """Return sliding window mask function."""
    pass

def create_alibi_bias(num_heads: int) -> Callable:
    """Return ALiBi positional bias function."""
    pass
```

## Example

```python
from torch.nn.attention.flex_attention import flex_attention

# Sliding window causal attention
def sliding_causal(b, h, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    window = (q_idx - kv_idx) <= 128
    return causal & window

output = flex_attention(q, k, v, score_mod=sliding_causal)
```

## Hints

- `create_block_mask` can precompute masks for efficiency
- Score mod can add biases (ALiBi) or mask (-inf for 0)
- Combine masks with `and_masks`, `or_masks` utilities
- Use `torch.compile` for best performance
