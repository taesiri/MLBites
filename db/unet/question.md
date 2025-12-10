# U-Net for Segmentation

## Problem Statement

Implement **U-Net** from scratch for image segmentation. U-Net's encoder-decoder architecture with skip connections is widely used for medical image segmentation and other pixel-wise prediction tasks.

Your task is to:

1. Build the contracting path (encoder)
2. Build the expansive path (decoder)
3. Implement skip connections between encoder and decoder
4. Output a segmentation mask

## U-Net Architecture

```
Input Image
    ↓
Encoder (contracting):
    Conv → Conv → Pool (save for skip)
    Conv → Conv → Pool (save for skip)
    Conv → Conv → Pool (save for skip)
    Conv → Conv → Pool (save for skip)
    ↓
Bottleneck: Conv → Conv
    ↓
Decoder (expansive):
    UpConv → Concat(skip) → Conv → Conv
    UpConv → Concat(skip) → Conv → Conv
    UpConv → Concat(skip) → Conv → Conv
    UpConv → Concat(skip) → Conv → Conv
    ↓
Output Conv (1x1) → Segmentation Mask
```

## Function Signature

```python
class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""
    def __init__(self, in_channels: int, out_channels: int):
        pass

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, features: list = [64, 128, 256, 512]):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return segmentation logits (B, out_channels, H, W)."""
        pass
```

## Example

```python
unet = UNet(in_channels=1, out_channels=2)  # Binary segmentation

x = torch.randn(1, 1, 256, 256)
mask = unet(x)  # (1, 2, 256, 256)

prediction = mask.argmax(dim=1)  # (1, 256, 256)
```

## Hints

- Use `nn.ConvTranspose2d` or `nn.Upsample` for upsampling
- Skip connections: concatenate encoder features with decoder features
- Crop encoder features if sizes don't match (or use padding='same')
- Common loss: CrossEntropyLoss or DiceLoss for segmentation
