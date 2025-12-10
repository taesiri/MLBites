# Vision Transformer (ViT)

## Problem Statement

Implement a **Vision Transformer (ViT)** from scratch. ViT applies the Transformer architecture directly to images by treating image patches as tokens.

Your task is to:

1. Implement patch embedding (split image into patches)
2. Add positional embeddings
3. Implement transformer encoder blocks
4. Add classification head with [CLS] token

## Requirements

- Split images into fixed-size patches
- Linear projection of flattened patches
- Learnable position embeddings
- Standard Transformer encoder
- Use [CLS] token for classification

## Function Signature

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        pass

class ViT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, 
                 num_classes: int, embed_dim: int, num_heads: int, num_layers: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images."""
        pass
```

## ViT Architecture

```
Image (3, 224, 224)
    ↓ Split into patches (14x14 patches of 16x16)
Patch Tokens (196, 768)
    ↓ Add [CLS] token  
Tokens (197, 768)
    ↓ Add position embeddings
    ↓ Transformer Encoder × N
    ↓ Take [CLS] token output
Classification Head → num_classes
```

## Example

```python
vit = ViT(
    img_size=224, patch_size=16, in_channels=3,
    num_classes=1000, embed_dim=768, num_heads=12, num_layers=12
)

x = torch.randn(1, 3, 224, 224)
logits = vit(x)  # (1, 1000)
```

## Hints

- Patch embedding: Conv2d with kernel_size=patch_size, stride=patch_size
- Number of patches = (img_size / patch_size)²
- [CLS] token is prepended to patch tokens
- Position embeddings are learned, not sinusoidal
