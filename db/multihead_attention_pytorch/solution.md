Approach
- Use a single linear layer to compute Q, K, V in one pass: `qkv = W(x)` then split into three tensors.
- Reshape from `(B, T, E)` into heads `(B, H, T, D)` where `D = E / H`.
- Compute scaled dot-product attention scores: `scores = (Q K^T) / sqrt(D)`.
- If `key_padding_mask` is provided, set masked key positions to `-inf` in `scores`.
- Apply softmax over the key dimension to get attention weights, optionally apply dropout.
- Compute the weighted sum of values: `out = attn @ V`, then merge heads back to `(B, T, E)` and apply an output projection.

Correctness
- Splitting into heads preserves the original embedding dimension while allowing independent attention distributions per head.
- Scaling by `sqrt(D)` keeps the dot products in a reasonable range, which stabilizes softmax.
- Masking with `-inf` forces softmax to assign zero probability to masked key/value positions.
- The final output projection mixes information across heads back into the model dimension.

Complexity
- Time: \(O(B \cdot H \cdot T^2 \cdot D)\) for attention, i.e. \(O(B \cdot T^2 \cdot E)\).
- Space: \(O(B \cdot H \cdot T^2)\) for attention scores/weights (dominant term).

Common Pitfalls
- Forgetting to transpose dimensions when splitting/merging heads.
- Not scaling by `sqrt(head_dim)` (softmax becomes too peaky for larger dimensions).
- Applying `key_padding_mask` to the wrong axis (it should mask keys/values, i.e. the last dimension of scores).
- Forgetting `.contiguous()` before `.view()` after a transpose.





