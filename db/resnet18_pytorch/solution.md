## Approach
- Implement `BasicBlock` as the core building block with two 3×3 convolutions and a skip connection.
- Each conv is followed by BatchNorm; ReLU is applied after bn1 and after adding the skip.
- Skip connection uses 1×1 conv + BN when dimensions change (stride > 1 or channel mismatch).
- ResNet-18 starts with 7×7 conv (stride 2) → BN → ReLU → MaxPool (stride 2), reducing 224→56.
- Four residual layers with [64, 128, 256, 512] channels, each containing 2 BasicBlocks.
- Layers 2-4 downsample with stride=2 in their first block.
- Use `AdaptiveAvgPool2d((1, 1))` to handle any final spatial size, then flatten and apply FC.
- Use `bias=False` in convs since BatchNorm handles the bias.
- Helper method `_make_layer` keeps the code DRY.

## Correctness
- Skip connections ensure gradient flow even in deep networks (solves vanishing gradient).
- Spatial dimension progression: 224 → 112 → 56 → 56 → 28 → 14 → 7 → 1.
- Channel progression: 3 → 64 → 64 → 128 → 256 → 512.
- BatchNorm normalizes activations, stabilizing training.
- No softmax in output—raw logits work with `CrossEntropyLoss`.
- Works with any batch size due to proper use of `torch.flatten(x, 1)`.

## Complexity
- **Time**: O(B × H × W × C² × K²) per conv layer. Total ≈ 1.8 billion FLOPs for 224×224 input.
- **Space**: ~11.7M parameters total (vs. 60M+ for VGG-16).
- Memory during training dominated by activations stored for backprop.

## Common Pitfalls
- Forgetting the skip connection—this makes it NOT a ResNet.
- Wrong padding for 3×3 conv (must be padding=1 to preserve spatial dims).
- Applying downsample to `out` instead of original `x` (identity).
- Forgetting `bias=False` when using BatchNorm (works but wasteful).
- Using `nn.ReLU()` without storing it, or calling it multiple times (fine, but inconsistent).
- Hardcoding spatial sizes instead of using `AdaptiveAvgPool2d`.
- Forgetting to call `super().__init__()` in constructors.


