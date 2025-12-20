# Vision Transformer Patch Projection Layer

## Problem
The Vision Transformer (ViT) processes images by first splitting them into fixed-size non-overlapping patches and then projecting each patch into an embedding space. This patch projection layer is the critical first step that converts a 2D image into a sequence of patch embeddings that can be processed by Transformer layers.

## Task
Implement the `PatchProjection` class that takes an input image and:
1. Splits it into non-overlapping square patches of size `patch_size × patch_size`
2. Projects each flattened patch into an `embed_dim`-dimensional embedding
3. Returns the sequence of patch embeddings

The efficient way to implement this is using a 2D convolution with `kernel_size=patch_size` and `stride=patch_size`, which simultaneously extracts and projects all patches.

## Function Signature

```python
class PatchProjection(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs
- **Inputs**:
  - `__init__`:
    - `image_size`: size of the input image (assumes square images)
    - `patch_size`: size of each patch (assumes square patches)
    - `in_channels`: number of input channels (e.g., 3 for RGB)
    - `embed_dim`: dimension of the output patch embeddings
  - `forward`:
    - `x`: `torch.Tensor` of shape `(B, C, H, W)` where:
      - `B` = batch size
      - `C` = number of channels (equals `in_channels`)
      - `H` = image height (equals `image_size`)
      - `W` = image width (equals `image_size`)
- **Outputs**:
  - `forward` returns `torch.Tensor` of shape `(B, num_patches, embed_dim)` where:
    - `num_patches = (image_size // patch_size) ** 2`

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- You may assume:
  - `image_size % patch_size == 0`
  - Input images have the expected size

## Examples

### Example 1 (basic forward pass)

```python
import torch

proj = PatchProjection(
    image_size=32,
    patch_size=8,
    in_channels=3,
    embed_dim=64,
)
proj.eval()

x = torch.randn(2, 3, 32, 32)  # batch of 2 images
patches = proj(x)
print(patches.shape)  # torch.Size([2, 16, 64])
# 32/8 = 4, so 4*4 = 16 patches per image
```

### Example 2 (standard ViT configuration)

```python
import torch

proj = PatchProjection(
    image_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
)
proj.eval()

x = torch.randn(1, 3, 224, 224)
patches = proj(x)
print(patches.shape)  # torch.Size([1, 196, 768])
# 224/16 = 14, so 14*14 = 196 patches
```

### Example 3 (single channel input)

```python
import torch

proj = PatchProjection(
    image_size=28,
    patch_size=7,
    in_channels=1,
    embed_dim=128,
)
proj.eval()

x = torch.randn(4, 1, 28, 28)  # e.g., grayscale MNIST
patches = proj(x)
print(patches.shape)  # torch.Size([4, 16, 128])
# 28/7 = 4, so 4*4 = 16 patches
```



