## Approach
- Subclass `nn.Module` and define all layers in `__init__`.
- Use two convolutional blocks, each consisting of `Conv2d` → `ReLU` → `MaxPool2d`.
- First conv: 1 input channel → 6 output channels, 5×5 kernel. After pooling: 14×14×6.
- Second conv: 6 → 16 channels, 5×5 kernel. After pooling: 5×5×16.
- Flatten the feature maps to a vector of size 400 (16×5×5).
- Apply three fully connected layers: 400→120→84→10.
- Use ReLU activations after conv and first two FC layers.
- Output raw logits (no softmax) for compatibility with `nn.CrossEntropyLoss`.

## Correctness
- The architecture matches the classic LeNet-5 design with modern modifications (ReLU instead of sigmoid/tanh, max pooling instead of average pooling).
- Shape progression is verified: 32→28→14→10→5 spatially, then flattened.
- No activation after the final layer ensures the output represents logits.
- Works with any batch size due to proper use of `x.view(x.size(0), -1)`.

## Complexity
- **Time**: O(B × C × H × W × K²) per conv layer, where B=batch, C=channels, H/W=spatial, K=kernel size. FC layers are O(in × out).
- **Space**: O(num_parameters) ≈ 62K parameters total (conv1: 156, conv2: 2416, fc1: 48120, fc2: 10164, fc3: 850).

## Common Pitfalls
- Forgetting to call `super().__init__()` in the constructor.
- Using wrong input size for fc1 (must be 16×5×5=400, not 16×4×4 or similar).
- Applying softmax in forward (CrossEntropyLoss expects raw logits).
- Hardcoding batch size in reshape instead of using `x.size(0)` or `-1`.
- Forgetting to apply ReLU between layers.
- Confusing kernel_size with other conv parameters (padding, stride).

