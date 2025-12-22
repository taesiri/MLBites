# Motivation

During autoregressive text generation (e.g., GPT-style models), tokens are generated one at a time. Naively recomputing attention over the entire sequence for each new token is inefficient because the keys and values for previous tokens don't change. KV caching stores these previously computed key/value tensors so they can be reused, reducing the per-token computation from O(T²) to O(T).

# Detailed Inputs and Outputs

- **inputs**:
  - `x`: `torch.Tensor` of shape `(B, T, E)` where:
    - `B` = batch size
    - `T` = number of new tokens (1 during generation, or full sequence length on first pass)
    - `E` = `embed_dim`
  - `kv_cache` (optional): tuple of two tensors `(cached_k, cached_v)`, each of shape `(B, num_heads, S, head_dim)` where:
    - `S` = number of previously cached tokens
    - `head_dim` = `embed_dim // num_heads`
    - If `None`, this is the first forward pass (no cache yet)
- **outputs**:
  - `output`: `torch.Tensor` of shape `(B, T, E)` — the attention output for the new tokens
  - `new_cache`: tuple `(new_k, new_v)`, each of shape `(B, num_heads, S+T, head_dim)` — updated cache including new keys/values

# Approach

- Compute Q, K, V for the new input tokens using a fused linear projection, then reshape into multi-head format `(B, num_heads, T, head_dim)`.
- If a KV cache exists from previous steps, concatenate the cached K and V with the newly computed K and V along the sequence dimension.
- Store the updated K and V tensors as the new cache to return.
- Compute scaled dot-product attention: `scores = (Q @ K^T) * scale`.
- Apply a causal mask: for each new query at position `i`, mask out all key positions `> (cache_len + i)` to prevent attending to future tokens.
- Apply softmax to get attention weights and compute the weighted sum of values.
- Merge heads back to `(B, T, E)` and apply the output projection.

# Implementation Details

## Initialization
- Store `embed_dim`, `num_heads`, and compute `head_dim = embed_dim // num_heads`
- Compute and store `scale = head_dim ** -0.5` for attention scaling
- Create fused QKV projection: `nn.Linear(embed_dim, 3 * embed_dim, bias=bias)` named `self.qkv_proj`
- Create output projection: `nn.Linear(embed_dim, embed_dim, bias=bias)` named `self.out_proj`

## Forward Pass
1. Get `B, T, E` from `x.shape`
2. Compute Q, K, V for new tokens via `qkv_proj`:
   - Reshape to `(B, T, 3, num_heads, head_dim)`
   - Permute to `(3, B, num_heads, T, head_dim)`
   - Unpack `q, k, v`
3. If `kv_cache` is not None:
   - Concatenate `cached_k` with `k` along sequence dimension (dim=2)
   - Concatenate `cached_v` with `v` along sequence dimension (dim=2)
4. Create `new_cache = (k, v)` to return
5. Compute attention scores: `q @ k.transpose(-2, -1)`
6. Scale by `self.scale`
7. Apply causal mask:
   - Compute `cache_len = total_len - T` where `total_len = k.size(2)`
   - For each query position `i` (0 to T-1), its absolute position is `cache_len + i`
   - It can only attend to key positions 0 to `cache_len + i`
   - Mask positions where `key_pos > query_abs_pos` with `-inf`
8. Apply softmax over last dimension
9. Compute output: `attn @ v`
10. Merge heads: transpose and reshape to `(B, T, E)`
11. Apply output projection
12. Return `(output, new_cache)`

# Examples

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

# Correctness

- Concatenating cached K/V with new K/V ensures queries have access to the full history without recomputing old projections.
- The causal mask ensures autoregressive behavior: each query only attends to positions at or before its absolute position in the sequence.
- When processing one token at a time, the output matches processing the full sequence at once because each position sees exactly the same causal context.
- The cache grows by T tokens each forward pass, correctly tracking the full key/value history.

# Complexity

- **Time**: O(B · H · T · S) for computing attention scores where T is new tokens and S is total sequence length (cached + new). This is O(T) per token instead of O(T²) naive recomputation.
- **Space**: O(B · H · S · D) for storing the KV cache, where S grows linearly with sequence length.

# Common Pitfalls

- Forgetting to concatenate cached K/V before computing attention scores.
- Incorrect causal mask indexing: the query at position `i` within the T new tokens has absolute position `cache_len + i`, not just `i`.
- Returning the wrong cache shape (should include both cached and new K/V).
- Applying the causal mask in the wrong dimension (should mask keys, not queries).
- Not handling the `kv_cache=None` case for the first forward pass.
- Modifying the cached tensors in-place instead of creating new concatenated tensors.




