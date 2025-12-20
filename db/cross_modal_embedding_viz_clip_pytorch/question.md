# Cross-Modal Embedding Visualization (CLIP + t-SNE/UMAP)

## Problem

CLIP (Contrastive Language-Image Pretraining) maps both images and text into a shared embedding space where semantically similar concepts are close together. Visualizing these embeddings in 2D helps understand how well the model aligns different modalities.

Your task is to implement a `CrossModalVisualizer` class that takes pre-extracted image and text embeddings from CLIP and projects them to 2D using either t-SNE or UMAP for visualization.

## Task

Implement a `CrossModalVisualizer` class that:
- Accepts image embeddings and text embeddings (both in CLIP's shared space)
- Projects all embeddings to 2D using t-SNE (default) or UMAP
- Returns the 2D coordinates along with modality labels

## Function Signature

```python
class CrossModalVisualizer:
    def __init__(
        self,
        method: str = "tsne",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42,
    ) -> None: ...

    def fit_transform(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]: ...
```

## Inputs and Outputs

- **`__init__`**:
  - `method`: str, either `"tsne"` or `"umap"` (dimensionality reduction algorithm)
  - `perplexity`: float, perplexity parameter for t-SNE (ignored if using UMAP)
  - `n_neighbors`: int, number of neighbors for UMAP (ignored if using t-SNE)
  - `random_state`: int, random seed for reproducibility

- **`fit_transform`**:
  - `image_embeddings`: `torch.Tensor` of shape `(N, D)`, CLIP image embeddings
  - `text_embeddings`: `torch.Tensor` of shape `(M, D)`, CLIP text embeddings
  - **Returns**: tuple of:
    - `coords`: `np.ndarray` of shape `(N+M, 2)`, 2D coordinates (images first, then text)
    - `modalities`: `np.ndarray` of shape `(N+M,)`, integer labels (0 for image, 1 for text)

## Constraints

- Must be solvable in 20–30 minutes.
- Interview-friendly: no need to load actual CLIP models or images.
- Assume inputs satisfy the documented contract (embeddings are pre-extracted, normalized, same dimensionality).
- Allowed libs: PyTorch (`torch`), NumPy (`numpy`), scikit-learn (`sklearn.manifold.TSNE`), and optionally `umap-learn` (`umap.UMAP`).
- For UMAP: if `umap-learn` is not installed, the method should raise `ImportError` with a helpful message.
- Embeddings should be L2-normalized before projection (CLIP embeddings are typically normalized).

## Examples

### Example 1 (small embeddings with t-SNE)

```python
import torch
import numpy as np

# 3 image embeddings, 2 text embeddings, 4D features
image_emb = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.8, 0.2, 0.0, 0.0],
])
text_emb = torch.tensor([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.9, 0.1],
])

viz = CrossModalVisualizer(method="tsne", perplexity=2.0, random_state=42)
coords, modalities = viz.fit_transform(image_emb, text_emb)

assert coords.shape == (5, 2)
assert modalities.shape == (5,)
assert list(modalities) == [0, 0, 0, 1, 1]  # 3 images, 2 texts
```

### Example 2 (larger embeddings)

```python
torch.manual_seed(42)
image_emb = torch.randn(50, 512)  # 50 CLIP image embeddings
text_emb = torch.randn(30, 512)   # 30 CLIP text embeddings

viz = CrossModalVisualizer(method="tsne", perplexity=10.0, random_state=0)
coords, modalities = viz.fit_transform(image_emb, text_emb)

assert coords.shape == (80, 2)
assert (modalities[:50] == 0).all()  # First 50 are images
assert (modalities[50:] == 1).all()  # Last 30 are texts
```




