## Approach
- Project `q`, `k`, `v` with learned matrices `w_q`, `w_k`, `w_v` to get `q_proj`, `k_proj`, `v_proj`.
- Reshape to split the embedding dimension into `num_heads` heads: `(B, T, D) -> (B, H, T, Dh)`.
- Compute scaled dot-product attention scores: `scores = (Q @ K^T) / sqrt(Dh)`.
- Apply masks:
  - `attn_mask` (shape `(Tq, Tk)`): boolean `True` positions are set to `-inf`, or float masks are added.
  - `key_padding_mask` (shape `(B, Tk)`): boolean `True` positions are set to `-inf` for all heads and queries.
- Softmax over the key dimension to get attention probabilities.
- Weighted sum of values: `attn @ V`.
- Concatenate heads back to `(B, Tq, D)` and apply output projection `w_o`.

## Correctness
- The projection + reshape step ensures each head attends in its own `Dh = D / H` subspace.
- Scaling by `sqrt(Dh)` keeps score magnitudes in a reasonable range for softmax.
- Masking by `-inf` guarantees masked key positions get probability ~0 after softmax.
- The final concatenation and output projection matches the standard multi-head attention formulation.

## Complexity
- Let `B` be batch size, `Tq` query length, `Tk` key/value length, `D` embedding dim, `H` heads.
- Time: \(O(B \cdot H \cdot Tq \cdot Tk \cdot (D/H)) = O(B \cdot Tq \cdot Tk \cdot D)\)
- Space: \(O(B \cdot H \cdot Tq \cdot Tk)\) for attention scores/probabilities (dominant term).

## Common Pitfalls
- Forgetting to scale by `sqrt(head_dim)` (can make softmax too peaky).
- Mask semantics: in PyTorch masks typically use `True` to mean "masked out".
- Shape mistakes when reshaping/transposing heads.
- Applying masks after softmax (too late).
- Not ensuring `D % num_heads == 0`.


