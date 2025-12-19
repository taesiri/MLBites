# Approach

- Compute Q, K, V for the new input tokens using a fused linear projection, then reshape into multi-head format `(B, num_heads, T, head_dim)`.
- If a KV cache exists from previous steps, concatenate the cached K and V with the newly computed K and V along the sequence dimension.
- Store the updated K and V tensors as the new cache to return.
- Compute scaled dot-product attention: `scores = (Q @ K^T) * scale`.
- Apply a causal mask: for each new query at position `i`, mask out all key positions `> (cache_len + i)` to prevent attending to future tokens.
- Apply softmax to get attention weights and compute the weighted sum of values.
- Merge heads back to `(B, T, E)` and apply the output projection.

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


