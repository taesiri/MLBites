## Background

CLIP learns a joint embedding space where images and their corresponding text descriptions are close together. Visualizing this space in 2D reveals:
- How well different modalities cluster together
- Whether semantically similar concepts (across modalities) are neighbors
- The overall structure of the learned representation

t-SNE and UMAP are the two most popular methods for embedding visualization:
- **t-SNE**: Preserves local structure, good for revealing clusters
- **UMAP**: Faster, preserves more global structure, scales better

---

## Approach

- Validate the `method` parameter in `__init__` (must be "tsne" or "umap").
- Store hyperparameters: `perplexity` for t-SNE, `n_neighbors` for UMAP.
- In `fit_transform()`:
  - Convert PyTorch tensors to NumPy arrays using `.detach().cpu().numpy()`.
  - L2-normalize embeddings (CLIP embeddings should be normalized for cosine similarity).
  - Concatenate image and text embeddings, with images first.
  - Create modality labels: 0 for images, 1 for texts.
  - Apply the chosen dimensionality reduction algorithm.
  - Return 2D coordinates and modality labels.

## Correctness

- L2 normalization ensures embeddings are on the unit hypersphere, making distances comparable.
- Using `.detach().cpu()` handles GPU tensors and tensors with gradients.
- The small epsilon (1e-8) in normalization prevents division by zero.
- Importing UMAP lazily allows the code to work without `umap-learn` for t-SNE mode.
- Random state ensures reproducible results for debugging and testing.

## Complexity

- **t-SNE Time:** \(O(N^2)\) for exact t-SNE, or \(O(N \log N)\) for Barnes-Hut approximation (sklearn default for large N).
- **UMAP Time:** \(O(N^{1.14})\) approximately, more scalable than exact t-SNE.
- **Space:** \(O(N^2)\) for distance matrices in both methods, \(O(N \cdot D)\) for embeddings.

## Common Pitfalls

- Forgetting to normalize embeddings (CLIP uses cosine similarity, so L2 norm matters).
- Not handling GPU tensors (need `.cpu()` before `.numpy()`).
- Not handling tensors with gradients (need `.detach()`).
- Setting perplexity too high for small datasets (must be < N/3).
- Forgetting to set random_state (non-reproducible results).
- Mixing up the order of concatenation (images vs. text first affects label interpretation).

