# Vision Transformer (ViT) from Scratch

## Problem
The Vision Transformer (ViT) applies the Transformer architecture to image classification by treating an image as a sequence of patches. Unlike CNNs that use convolutions, ViT splits an image into fixed-size patches, linearly embeds them, adds position embeddings, and processes the sequence through standard Transformer encoder blocks.

## Task
Implement a minimal Vision Transformer (ViT) model for image classification in PyTorch. Your implementation should include:
1. Patch embedding layer (split image into patches and project to embedding dimension)
2. Learnable CLS token prepended to the patch sequence
3. Learnable position embeddings
4. Transformer encoder blocks (multi-head self-attention + MLP with residual connections)
5. Classification head using the CLS token output

## Function Signature

```python
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
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: `torch.Tensor` of shape `(B, C, H, W)` where:
    - `B` = batch size
    - `C` = number of channels (default 3)
    - `H` = image height (default 224)
    - `W` = image width (default 224)
- **outputs**:
  - returns `torch.Tensor` of shape `(B, num_classes)` — logits for each class

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- You may assume:
  - `image_size % patch_size == 0`
  - `embed_dim % num_heads == 0`
  - Input images have the expected size

## Examples

### Example 1 (basic forward pass)

```python
import torch

model = ViT(
    image_size=32,
    patch_size=8,
    in_channels=3,
    num_classes=10,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    mlp_ratio=4.0,
    dropout=0.0,
)
model.eval()

x = torch.randn(2, 3, 32, 32)  # batch of 2 images
logits = model(x)
print(logits.shape)  # torch.Size([2, 10])
```

### Example 2 (verifying number of patches)

```python
import torch

# 224x224 image with 16x16 patches -> 14*14 = 196 patches
model = ViT(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    num_heads=12,
    num_layers=1,
    mlp_ratio=4.0,
    dropout=0.0,
)
model.eval()

x = torch.randn(1, 3, 224, 224)
logits = model(x)
print(logits.shape)  # torch.Size([1, 1000])

# The internal sequence length should be 196 patches + 1 CLS token = 197
```




