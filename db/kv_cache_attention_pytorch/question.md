# KV Cache in Multi-Head Attention

## Problem
Implement multi-head self-attention with KV caching for efficient autoregressive generation.

## Task
Implement a PyTorch module that:
1. Accepts an optional KV cache from previous steps
2. Computes attention using cached keys/values plus new ones
3. Returns the updated cache for the next step

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
- `x`: Input tensor of shape `(B, T, E)` where `E = embed_dim`
- `kv_cache`: Optional tuple `(cached_k, cached_v)` from previous steps, or `None` for first pass
- Returns: `(output, new_cache)` where `output` has shape `(B, T, E)` and `new_cache` is `(new_k, new_v)`

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- Assume `embed_dim % num_heads == 0` and causal masking is required.




