# KV Cache for LLM Inference

## Problem Statement

Implement **KV Cache** to accelerate autoregressive LLM inference. During generation, we cache key-value pairs from previous tokens to avoid recomputing them.

Your task is to:

1. Implement a KV cache data structure
2. Modify attention to use cached keys/values
3. Support incremental updates during generation
4. Handle cache management (sizing, clearing)

## Why KV Cache?

Without cache: Each new token requires recomputing attention for ALL previous tokens → O(n²) per token

With cache: Only compute attention for the NEW token using cached K,V → O(n) per token

## Function Signature

```python
class KVCache:
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int):
        pass
    
    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int) -> tuple:
        """Add new K,V and return full cached K,V."""
        pass
    
    def get(self, layer_idx: int) -> tuple:
        """Get cached K,V for a layer."""
        pass

class CausalAttentionWithCache(nn.Module):
    def forward(self, x: torch.Tensor, cache: KVCache = None, start_pos: int = 0):
        """Attention with optional KV cache for efficient inference."""
        pass
```

## Cache Structure

```
cache_k[layer][batch, num_heads, seq_len, head_dim]
cache_v[layer][batch, num_heads, seq_len, head_dim]

During generation:
- Input: (batch, 1, dim)  # Just the new token
- Cache: (batch, heads, prev_len, head_dim)
- Output: (batch, 1, dim)
```

## Example

```python
model = GPTWithCache(...)
cache = KVCache(batch_size=1, max_len=2048, num_heads=32, head_dim=128)

# First forward (prefill)
prompt = tokenize("Hello, how are")
logits = model(prompt, cache=cache, start_pos=0)

# Subsequent forwards (decode)
for i in range(max_new_tokens):
    next_token = sample(logits[:, -1])
    logits = model(next_token.unsqueeze(1), cache=cache, start_pos=len(prompt) + i)
```

## Hints

- Pre-allocate cache tensors for max sequence length
- Track current position in cache
- Only compute Q for new tokens, but attend to all cached K,V
- Use `start_pos` to correctly slice position embeddings/masks
