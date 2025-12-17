# GPT-2 Implementation

## Approach
- **Embeddings**: Combine learned token embeddings with learned positional embeddings (both of dimension `embed_dim`). Add them together and apply dropout.
- **Causal Self-Attention**: Use a single linear layer for fused Q/K/V projection. Reshape into multiple heads, compute scaled dot-product attention, and apply a causal mask (lower triangular) to prevent attending to future tokens.
- **MLP**: Two linear layers with GELU activation in between. Hidden dimension is 4× the embedding dimension (standard GPT-2 design).
- **Pre-Normalization**: Apply LayerNorm *before* attention and MLP (not after). This improves training stability compared to post-norm.
- **Residual Connections**: Add the output of attention/MLP back to the input (skip connections).
- **Stacking**: Chain `num_layers` Transformer blocks sequentially.
- **Output**: Final LayerNorm followed by a linear projection to vocabulary size to produce logits.

## Correctness
- The causal mask (lower triangular matrix) ensures position `i` can only attend to positions `0..i`, preventing information leakage from future tokens.
- Pre-normalization matches the GPT-2 architecture and provides stable gradients.
- The scale factor `1/sqrt(head_dim)` keeps dot products in a stable range for softmax.
- Position embeddings are learned (not sinusoidal) matching the original GPT-2.

## Complexity
- **Time**: \(O(B \cdot T^2 \cdot C)\) per layer for attention (quadratic in sequence length).
- **Space**: \(O(B \cdot H \cdot T^2)\) for attention weights per layer; \(O(\text{num\_layers} \cdot C^2)\) for parameters.

## Common Pitfalls
- Forgetting the causal mask, allowing the model to "see" future tokens.
- Using post-normalization instead of pre-normalization (different from GPT-2).
- Not applying the scale factor `1/sqrt(head_dim)` to attention scores.
- Forgetting `.contiguous()` before `.view()` when reshaping after transpose.
- Mixing up the order of dimensions when splitting/merging attention heads.
- Not handling the MLP expansion factor (should be 4×, not 2×).

