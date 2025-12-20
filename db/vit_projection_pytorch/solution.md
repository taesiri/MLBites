## Approach

- The goal is to convert an image of shape `(B, C, H, W)` into a sequence of patch embeddings of shape `(B, num_patches, embed_dim)`.
- Instead of manually splitting the image into patches and applying a linear layer, use a 2D convolution with `kernel_size=patch_size` and `stride=patch_size`.
- This convolution efficiently extracts non-overlapping patches and projects them in one operation.
- After the convolution, the output has shape `(B, embed_dim, H/P, W/P)`.
- Flatten the spatial dimensions to get `(B, embed_dim, num_patches)`.
- Transpose to get the final sequence format `(B, num_patches, embed_dim)`.

## Math

Each patch \(p_i \in \mathbb{R}^{P \times P \times C}\) is extracted from the image and flattened to a vector of size \(P^2 \cdot C\).

The projection is a linear transformation:

\[z_i = W \cdot \text{flatten}(p_i) + b\]

where \(W \in \mathbb{R}^{D \times (P^2 \cdot C)}\) and \(b \in \mathbb{R}^D\).

Using a Conv2d with `kernel_size=P` and `stride=P` is mathematically equivalent:

\[\text{Conv2d}(x) \in \mathbb{R}^{B \times D \times (H/P) \times (W/P)}\]

The number of patches is:

\[N = \frac{H}{P} \times \frac{W}{P} = \left(\frac{H}{P}\right)^2\]

## Correctness

- The convolution with `kernel_size=patch_size` and `stride=patch_size` naturally extracts non-overlapping patches.
- Each output channel of the convolution corresponds to one dimension of the embedding.
- The `flatten(2)` operation merges the spatial dimensions `(H/P, W/P)` into a single dimension.
- The `transpose(1, 2)` reorders from `(B, D, N)` to `(B, N, D)`, matching the sequence format expected by Transformer layers.

## Complexity

- Time: \(O(B \cdot N \cdot P^2 \cdot C \cdot D)\) where \(N = (H/P)^2\) is the number of patches.
- Space: \(O(B \cdot N \cdot D)\) for the output tensor.
- The convolution is highly optimized on GPUs, making this efficient in practice.

## Common Pitfalls

- Forgetting to transpose after flattening—Transformers expect `(B, seq_len, embed_dim)`, not `(B, embed_dim, seq_len)`.
- Using wrong kernel/stride size—both must equal `patch_size` for non-overlapping patches.
- Incorrect flatten dimension—should use `flatten(2)` to merge only spatial dimensions, not batch.
- Not storing `num_patches` as an attribute—this is often needed for position embeddings in the full ViT.



