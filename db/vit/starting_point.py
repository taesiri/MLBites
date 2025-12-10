"""
Vision Transformer (ViT) - Starting Point

Implement ViT from scratch.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # TODO: Create patch projection
        # Use Conv2d with kernel_size=patch_size, stride=patch_size
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            (batch, n_patches, embed_dim)
        """
        # TODO: Project patches and reshape
        pass


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # TODO: Implement multi-head attention
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # TODO: Layer norm, attention, MLP
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x = x + attention(norm(x))
        # TODO: x = x + mlp(norm(x))
        pass


class ViT(nn.Module):
    """Vision Transformer."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # TODO: Patch embedding
        
        # TODO: [CLS] token (learnable)
        
        # TODO: Position embeddings (learnable)
        
        # TODO: Transformer encoder blocks
        
        # TODO: Classification head
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            (batch, num_classes)
        """
        # TODO: Patch embed
        # TODO: Prepend [CLS] token
        # TODO: Add position embeddings
        # TODO: Pass through transformer blocks
        # TODO: Take [CLS] token and classify
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create ViT 
    vit = ViT(
        img_size=224, patch_size=16, in_channels=3,
        num_classes=10, embed_dim=768, num_heads=12, num_layers=6
    )
    
    x = torch.randn(2, 3, 224, 224)
    logits = vit(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in vit.parameters()):,}")
