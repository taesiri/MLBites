from __future__ import annotations

import numpy as np
import torch
from sklearn.manifold import TSNE


class CrossModalVisualizer:
    def __init__(
        self,
        method: str = "tsne",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42,
    ) -> None:
        """Initialize the cross-modal embedding visualizer.

        Args:
            method: Dimensionality reduction method, either "tsne" or "umap".
            perplexity: Perplexity parameter for t-SNE (ignored for UMAP).
            n_neighbors: Number of neighbors for UMAP (ignored for t-SNE).
            random_state: Random seed for reproducibility.
        """
        if method not in ("tsne", "umap"):
            raise ValueError(f"method must be 'tsne' or 'umap', got '{method}'")

        self.method = method
        self.perplexity = perplexity
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_transform(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project image and text embeddings to 2D for visualization.

        Args:
            image_embeddings: Tensor of shape (N, D), CLIP image embeddings.
            text_embeddings: Tensor of shape (M, D), CLIP text embeddings.

        Returns:
            coords: ndarray of shape (N+M, 2), 2D coordinates.
            modalities: ndarray of shape (N+M,), 0 for image, 1 for text.
        """
        # Convert to numpy
        img_np = image_embeddings.detach().cpu().numpy()
        txt_np = text_embeddings.detach().cpu().numpy()

        # L2-normalize embeddings
        img_np = img_np / (np.linalg.norm(img_np, axis=1, keepdims=True) + 1e-8)
        txt_np = txt_np / (np.linalg.norm(txt_np, axis=1, keepdims=True) + 1e-8)

        # Concatenate: images first, then text
        combined = np.concatenate([img_np, txt_np], axis=0)

        # Create modality labels: 0 for images, 1 for text
        n_images = img_np.shape[0]
        n_texts = txt_np.shape[0]
        modalities = np.array([0] * n_images + [1] * n_texts)

        # Apply dimensionality reduction
        if self.method == "tsne":
            reducer = TSNE(
                n_components=2,
                perplexity=self.perplexity,
                random_state=self.random_state,
            )
            coords = reducer.fit_transform(combined)
        else:  # umap
            try:
                import umap
            except ImportError:
                raise ImportError(
                    "UMAP requires 'umap-learn' package. Install with: pip install umap-learn"
                )

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
            coords = reducer.fit_transform(combined)

        return coords, modalities


