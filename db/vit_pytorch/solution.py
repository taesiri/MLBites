from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(
        self, image_size: int, patch_size: int, in_channels: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        # Use conv2d as an efficient way to split and embed patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = self.proj(x)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MLP(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture: norm before attention/mlp
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification."""

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

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token prepended to patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable position embeddings for CLS + patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize position embeddings and cls token with small values
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        B = x.shape[0]

        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, num_patches, D) -> (B, 1 + num_patches, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer encoder
        x = self.blocks(x)

        # Final layer norm
        x = self.norm(x)

        # Extract CLS token output and classify
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits




