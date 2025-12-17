# Implement KV Cache in Multi-Head Attention

## Problem
During autoregressive text generation (e.g., GPT-style models), tokens are generated one at a time. Naively recomputing attention over the entire sequence for each new token is inefficient because the keys and values for previous tokens don't change. KV caching stores these previously computed key/value tensors so they can be reused, reducing the per-token computation from O(T²) to O(T).

## Task
Implement a PyTorch module that performs **multi-head self-attention with KV caching** for efficient autoregressive generation. The module should:
1. Accept an optional KV cache from previous steps
2. Compute attention using cached keys/values plus new ones
3. Return the updated cache for the next step

## Function Signature

```python
class CachedMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: `torch.Tensor` of shape `(B, T, E)` where:
    - `B` = batch size
    - `T` = number of new tokens (1 during generation, or full sequence length on first pass)
    - `E` = `embed_dim`
  - `kv_cache` (optional): tuple of two tensors `(cached_k, cached_v)`, each of shape `(B, num_heads, S, head_dim)` where:
    - `S` = number of previously cached tokens
    - If `None`, this is the first forward pass (no cache yet)
- **outputs**:
  - `output`: `torch.Tensor` of shape `(B, T, E)` — the attention output for the new tokens
  - `new_cache`: tuple `(new_k, new_v)`, each of shape `(B, num_heads, S+T, head_dim)` — updated cache including new keys/values

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- You may assume:
  - `embed_dim % num_heads == 0`
  - During generation, queries only attend to past and current positions (causal)
  - No dropout required (inference mode)

## Examples

### Example 1 (Full sequence prefill, then single-token generation)

```python
import torch

torch.manual_seed(42)
m = CachedMultiheadAttention(embed_dim=4, num_heads=2, bias=False)

# First pass: process full prompt (T=3)
x1 = torch.randn(1, 3, 4)  # (B=1, T=3, E=4)
out1, cache = m(x1, kv_cache=None)
print(out1.shape)  # torch.Size([1, 3, 4])
print(cache[0].shape)  # torch.Size([1, 2, 3, 2]) - cached K
print(cache[1].shape)  # torch.Size([1, 2, 3, 2]) - cached V

# Second pass: generate next token (T=1), reusing cache
x2 = torch.randn(1, 1, 4)  # (B=1, T=1, E=4)
out2, cache = m(x2, kv_cache=cache)
print(out2.shape)  # torch.Size([1, 1, 4])
print(cache[0].shape)  # torch.Size([1, 2, 4, 2]) - cache now has 4 tokens
```

### Example 2 (Verify cache produces same output as full recompute)

```python
import torch

torch.manual_seed(0)
m = CachedMultiheadAttention(embed_dim=4, num_heads=2, bias=False)

# Full sequence in one pass
x_full = torch.randn(1, 4, 4)
out_full, _ = m(x_full, kv_cache=None)

# Same sequence, but processed incrementally with caching
out1, cache = m(x_full[:, :2, :], kv_cache=None)  # First 2 tokens
out2, cache = m(x_full[:, 2:3, :], kv_cache=cache)  # 3rd token
out3, cache = m(x_full[:, 3:4, :], kv_cache=cache)  # 4th token

# Outputs for positions 2,3,4 should match (position 0,1 differ due to causal masking)
out_incremental = torch.cat([out1, out2, out3], dim=1)
print(torch.allclose(out_full[:, -1:, :], out_incremental[:, -1:, :], atol=1e-5))  # True
```

