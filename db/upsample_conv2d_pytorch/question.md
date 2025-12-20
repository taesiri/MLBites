# UpsampleConv2D Layer

## Problem

In generative models, autoencoders, and image segmentation networks, we often need to increase the spatial resolution of feature maps. A common approach is the **Upsample-Convolution** pattern: first upsample the input to a higher resolution using interpolation, then apply a standard convolution to refine the features.

This approach is preferred over transposed convolution (`ConvTranspose2d`) because it avoids the "checkerboard artifacts" that transposed convolutions can produce due to overlapping kernels.

## Task

Implement an `UpsampleConv2d` class in PyTorch that:
1. Upsamples the input by a given scale factor using a specified interpolation mode
2. Applies a 2D convolution to the upsampled output

Your implementation should combine `F.interpolate` for upsampling with a standard `nn.Conv2d` for the convolution step.

## Function Signature

```python
import torch
import torch.nn as nn

class UpsampleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: int = 2,
        mode: str = "nearest",
        padding: int = 0,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## Inputs and Outputs

### `__init__`
- **Inputs**:
  - `in_channels`: Number of channels in the input image (integer)
  - `out_channels`: Number of channels produced by the convolution (integer)
  - `kernel_size`: Size of the square convolving kernel (integer)
  - `scale_factor`: Upsampling scale factor (default 2)
  - `mode`: Interpolation mode for upsampling, one of `"nearest"` or `"bilinear"` (default `"nearest"`)
  - `padding`: Zero-padding added to both sides of the input after upsampling (default 0)
- The internal `Conv2d` weight should have shape `(out_channels, in_channels, kernel_size, kernel_size)`
- The internal `Conv2d` bias should have shape `(out_channels,)`

### `forward`
- **Inputs**:
  - `x`: Input tensor of shape `(batch_size, in_channels, height, width)`
- **Outputs**:
  - Output tensor of shape `(batch_size, out_channels, out_height, out_width)`
  - Where `out_height = height * scale_factor + 2*padding - kernel_size + 1`
  - And `out_width = width * scale_factor + 2*padding - kernel_size + 1`

## Constraints

- Must be solvable in 20–30 minutes.
- Interview-friendly: focus on correctness, not edge cases.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- You MAY use `torch.nn.Conv2d` and `torch.nn.functional.interpolate` (the goal is understanding the pattern, not re-implementing convolution).
- Only square kernels are required (kernel_size is a single int, not a tuple).
- Only support `mode="nearest"` and `mode="bilinear"` for upsampling.

## Examples

### Example 1 (2x upsample with 3x3 conv)
```python
import torch

layer = UpsampleConv2d(in_channels=3, out_channels=16, kernel_size=3, scale_factor=2)
x = torch.randn(1, 3, 8, 8)  # batch=1, channels=3, height=8, width=8

out = layer(x)
# After upsampling: (1, 3, 16, 16)
# After conv (kernel=3, no padding): (1, 16, 14, 14)
```

### Example 2 (with padding to preserve size)
```python
import torch

layer = UpsampleConv2d(in_channels=8, out_channels=4, kernel_size=3, scale_factor=2, padding=1)
x = torch.randn(2, 8, 16, 16)

out = layer(x)
# After upsampling: (2, 8, 32, 32)
# After conv (kernel=3, padding=1): (2, 4, 32, 32)
```

### Example 3 (bilinear upsampling)
```python
import torch

layer = UpsampleConv2d(
    in_channels=1, out_channels=8, kernel_size=5,
    scale_factor=4, mode="bilinear", padding=2
)
x = torch.randn(1, 1, 7, 7)

out = layer(x)
# After upsampling: (1, 1, 28, 28)
# After conv (kernel=5, padding=2): (1, 8, 28, 28)
```



