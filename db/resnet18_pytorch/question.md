# Implement ResNet-18

## Problem
ResNet (Residual Network) revolutionized deep learning by introducing skip connections that allow gradients to flow directly through the network, enabling training of much deeper networks. ResNet-18 is the smallest variant, consisting of 18 layers with residual blocks. Understanding residual connections is fundamental for modern deep learning architectures.

## Task
Implement ResNet-18 as a PyTorch `nn.Module`. The network should accept 224×224 RGB images and output class logits for a configurable number of classes (default 1000 for ImageNet).

You need to implement:
1. **BasicBlock**: A residual block with two 3×3 convolutions and a skip connection
2. **ResNet18**: The full network using BasicBlocks

### BasicBlock Structure
- Conv(3×3) → BatchNorm → ReLU → Conv(3×3) → BatchNorm
- Add skip connection (with optional downsample for dimension matching)
- Final ReLU

### ResNet-18 Architecture
1. **Initial**: Conv(7×7, stride=2, 64 filters) → BatchNorm → ReLU → MaxPool(3×3, stride=2)
2. **Layer 1**: 2 BasicBlocks with 64 filters
3. **Layer 2**: 2 BasicBlocks with 128 filters (first block stride=2)
4. **Layer 3**: 2 BasicBlocks with 256 filters (first block stride=2)
5. **Layer 4**: 2 BasicBlocks with 512 filters (first block stride=2)
6. **Output**: AdaptiveAvgPool → Flatten → FC(512 → num_classes)

## Function Signature

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: a `torch.Tensor` of shape `(batch_size, 3, 224, 224)` — batch of RGB images
- **outputs**:
  - `torch.Tensor` of shape `(batch_size, num_classes)` — raw logits (no softmax)

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: use standard PyTorch layers (`nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`).
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`, `torch.nn`) and Python standard library.

## Examples

### Example 1 (single image, default classes)
```python
model = ResNet18()
x = torch.randn(1, 3, 224, 224)
out = model(x)
# out.shape == torch.Size([1, 1000])
```

### Example 2 (batch, custom classes)
```python
model = ResNet18(num_classes=10)
x = torch.randn(8, 3, 224, 224)
out = model(x)
# out.shape == torch.Size([8, 10])
```

### Example 3 (BasicBlock usage)
```python
block = BasicBlock(64, 128, stride=2)
x = torch.randn(1, 64, 56, 56)
out = block(x)
# out.shape == torch.Size([1, 128, 28, 28])
```

