## Approach

- Split the input image into non-overlapping patches using a convolution with `kernel_size=patch_size` and `stride=patch_size`.
- Flatten and transpose the patches to get a sequence of shape `(B, num_patches, embed_dim)`.
- Create a learnable CLS token and prepend it to the patch sequence, resulting in `(B, 1 + num_patches, embed_dim)`.
- Add learnable position embeddings to encode spatial information.
- Pass the sequence through `num_layers` Transformer encoder blocks, each consisting of:
  - Layer normalization (pre-norm style)
  - Multi-head self-attention with residual connection
  - Layer normalization
  - MLP (Linear → GELU → Linear) with residual connection
- Apply final layer normalization.
- Use the CLS token output (first token) as the image representation for classification.
- Project through a linear head to get class logits.

## Math

The patch embedding uses a convolution to project each patch:

\[x_p = \text{Conv2d}(x) \in \mathbb{R}^{B \times N \times D}\]

where \(N = (H/P)^2\) is the number of patches and \(D\) is the embedding dimension.

The CLS token and position embeddings are added:

\[z_0 = [x_{\text{cls}}; x_p^1; x_p^2; \ldots; x_p^N] + E_{\text{pos}}\]

Each Transformer block applies (pre-norm style):

\[z'_l = z_{l-1} + \text{MSA}(\text{LN}(z_{l-1}))\]
\[z_l = z'_l + \text{MLP}(\text{LN}(z'_l))\]

where MSA is multi-head self-attention and LN is layer normalization.

The final classification uses the CLS token:

\[y = \text{Linear}(\text{LN}(z_L^0))\]

## Correctness

- The patch embedding via convolution is equivalent to splitting and linearly projecting patches.
- Position embeddings provide spatial information since self-attention is permutation-equivariant.
- The CLS token aggregates global information through attention layers.
- Pre-norm (LayerNorm before attention/MLP) is more stable for deep networks.
- The residual connections ensure gradient flow through deep layers.

## Complexity

- Time: \(O(B \cdot N^2 \cdot D)\) for self-attention per layer, where \(N = (H/P)^2 + 1\).
- Space: \(O(B \cdot N^2)\) for attention weights per layer.
- For ViT-Base (224×224, patch 16): \(N = 197\), much smaller than pixel-level attention.

## Common Pitfalls

- Forgetting to prepend the CLS token before adding position embeddings.
- Using post-norm instead of pre-norm (ViT uses pre-norm for stability).
- Incorrect patch embedding dimensions (flatten then transpose).
- Not initializing CLS token and position embeddings properly.
- Forgetting the final layer norm before the classification head.
- Using the wrong token for classification (should be the CLS token at index 0).


