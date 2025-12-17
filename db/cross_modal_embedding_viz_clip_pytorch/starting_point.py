from __future__ import annotations

import numpy as np
import torch


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

        Notes:
            - Store parameters for later use in fit_transform().
            - Validate that method is either "tsne" or "umap".
        """
        # TODO: Store method and relevant hyperparameters
        # TODO: Validate method is "tsne" or "umap"
        raise NotImplementedError

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

        Notes:
            - Convert tensors to numpy arrays.
            - L2-normalize embeddings before projection.
            - Concatenate image and text embeddings (images first).
            - Apply t-SNE or UMAP based on self.method.
            - Create modality labels: 0 for images, 1 for texts.
        """
        # TODO: Convert to numpy and L2-normalize
        # TODO: Concatenate embeddings (images first, then text)
        # TODO: Create modality labels array
        # TODO: Apply dimensionality reduction (t-SNE or UMAP)
        # TODO: Return 2D coords and modality labels
        raise NotImplementedError

