## Approach
- Subclass `nn.Module` and define all layers in `__init__`.
- Use two convolutional blocks, each consisting of `Conv2d` → `ReLU` → `MaxPool2d`.
- First conv: 1 input channel → 32 output channels, 3×3 kernel with padding=1 (preserves spatial size). After pooling: 14×14×32.
- Second conv: 32 → 64 channels, 3×3 kernel with padding=1. After pooling: 7×7×64.
- Flatten the feature maps to a vector of size 3136 (64×7×7).
- Apply two fully connected layers: 3136→128→10.
- Use ReLU activation after conv layers and first FC layer.
- Output raw logits (no softmax) for compatibility with `nn.CrossEntropyLoss`.

## Correctness
- The 3×3 kernels with padding=1 preserve spatial dimensions before pooling, making the math straightforward: 28→14→7.
- Shape progression: 28×28×1 → 28×28×32 → 14×14×32 → 14×14×64 → 7×7×64 → 3136 → 128 → 10.
- No activation after the final layer ensures the output represents logits.
- Works with any batch size due to proper use of `x.view(x.size(0), -1)`.

## Complexity
- **Time**: O(B × C × H × W × K²) per conv layer, where B=batch, C=channels, H/W=spatial, K=kernel size. FC layers are O(in × out).
- **Space**: O(num_parameters) ≈ 420K parameters total:
  - conv1: 1×32×3×3 + 32 = 320
  - conv2: 32×64×3×3 + 64 = 18,496
  - fc1: 3136×128 + 128 = 401,536
  - fc2: 128×10 + 10 = 1,290

## Common Pitfalls
- Forgetting to call `super().__init__()` in the constructor.
- Wrong input size for fc1: must account for padding (7×7×64=3136, not 6×6×64).
- Forgetting `padding=1` in conv layers, which changes the output spatial dimensions.
- Applying softmax in forward (CrossEntropyLoss expects raw logits).
- Hardcoding batch size in reshape instead of using `x.size(0)` or `-1`.
- Forgetting to apply ReLU between layers.

