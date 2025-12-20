from __future__ import annotations

import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        """Patch projection layer for Vision Transformer.

        Args:
            image_size: Size of the input image (assumes square images).
            patch_size: Size of each patch (assumes square patches).
            in_channels: Number of input channels.
            embed_dim: Dimension of the output patch embeddings.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches into embeddings.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        raise NotImplementedError



