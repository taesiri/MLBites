# ResNet from Scratch

## Problem Statement

Implement **ResNet (Residual Network)** from scratch. ResNet introduced skip connections that enable training of very deep networks (100+ layers).

Your task is to:

1. Implement the Basic Block (for ResNet-18/34)
2. Implement the Bottleneck Block (for ResNet-50/101/152)
3. Build configurable ResNet architectures
4. Understand why skip connections help training

## ResNet Architecture

```
Conv 7x7, stride 2 → BatchNorm → ReLU → MaxPool 3x3 stride 2
    ↓
Layer1: N × Block (64 filters)
Layer2: N × Block (128 filters, stride 2)
Layer3: N × Block (256 filters, stride 2)
Layer4: N × Block (512 filters, stride 2)
    ↓
AdaptiveAvgPool → FC(num_classes)
```

## Block Types

**Basic Block (ResNet-18/34):**
```
x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+x) → ReLU
```

**Bottleneck Block (ResNet-50+):**
```
x → Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN → (+x) → ReLU
```

## Function Signature

```python
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        pass

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        pass

class ResNet(nn.Module):
    def __init__(self, block, layers: list[int], num_classes: int = 1000):
        pass
```

## Example

```python
# ResNet-18: BasicBlock, [2, 2, 2, 2]
resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)

# ResNet-50: Bottleneck, [3, 4, 6, 3]
resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)

x = torch.randn(1, 3, 224, 224)
out = resnet50(x)  # (1, 1000)
```

## Hints

- Downsample shortcut when stride > 1 or channels change
- Use 1x1 conv for downsampling shortcut
- Bottleneck expansion = 4 (output channels = 4 × base channels)
- Pre-activation variant: BN→ReLU→Conv (optional)
