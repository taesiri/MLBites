from __future__ import annotations

import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Vision Transformer for image classification.

        Args:
            image_size: Input image size (assumes square images).
            patch_size: Size of each patch (assumes square patches).
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            mlp_ratio: Ratio of MLP hidden dim to embed_dim.
            dropout: Dropout probability.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        raise NotImplementedError


