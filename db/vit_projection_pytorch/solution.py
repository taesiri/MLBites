from __future__ import annotations

import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    """Patch projection layer for Vision Transformer.

    Splits an image into non-overlapping patches and projects each patch
    into an embedding space using a single convolution operation.
    """

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

        # Calculate number of patches: (H/P) * (W/P) = (image_size/patch_size)^2
        self.num_patches = (image_size // patch_size) ** 2

        # Use Conv2d to simultaneously extract and project patches
        # kernel_size=patch_size ensures each kernel covers exactly one patch
        # stride=patch_size ensures non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches into embeddings.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        # Apply convolution: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)

        # Flatten spatial dimensions: (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)

        # Transpose to sequence format: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x



