# Conv2D Layer Forward Pass

## Problem
The 2D convolution operation is the fundamental building block of Convolutional Neural Networks (CNNs). It slides a set of learnable filters (kernels) across the input image, computing element-wise products and summing them to produce feature maps. Each filter detects a specific pattern (edges, textures, shapes) in the input.

## Task
Implement a function `conv2d_forward` that performs the forward pass of a 2D convolution layer in NumPy. The function should support:
- Batch processing (multiple images at once)
- Multiple input and output channels
- Configurable stride and padding

## Function Signature

```python
import numpy as np

def conv2d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    ...
```

## Inputs and Outputs

### Inputs
- `x`: Input tensor of shape `(N, C_in, H, W)` where:
  - `N` is batch size
  - `C_in` is number of input channels
  - `H` is input height
  - `W` is input width
- `weight`: Convolution kernels of shape `(C_out, C_in, kH, kW)` where:
  - `C_out` is number of output channels (number of filters)
  - `C_in` is number of input channels (must match input)
  - `kH` is kernel height
  - `kW` is kernel width
- `bias`: Optional bias of shape `(C_out,)`, added to each output channel
- `stride`: Stride of the convolution (default 1)
- `padding`: Zero-padding added to both sides of the input (default 0)

### Outputs
- Output tensor of shape `(N, C_out, H_out, W_out)` where:
  - `H_out = (H + 2*padding - kH) // stride + 1`
  - `W_out = (W + 2*padding - kW) // stride + 1`

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: focus on a clear, straightforward implementation.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: NumPy (`numpy`) and Python standard library.
- You may use nested loops for clarity; no need for highly optimized im2col.

## Examples

### Example 1 (single channel, no padding, stride=1)
```python
import numpy as np

# Single 4x4 image, 1 channel
x = np.array([[[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]]]).astype(np.float64)  # shape (1, 1, 4, 4)

# Single 2x2 filter
weight = np.array([[[[1, 0],
                     [0, 1]]]]).astype(np.float64)  # shape (1, 1, 2, 2)

out = conv2d_forward(x, weight)
# out shape: (1, 1, 3, 3)
# out[0, 0] = [[1+6, 2+7, 3+8],
#              [5+10, 6+11, 7+12],
#              [9+14, 10+15, 11+16]]
#           = [[7, 9, 11],
#              [15, 17, 19],
#              [23, 25, 27]]
```

### Example 2 (with padding and stride)
```python
import numpy as np

x = np.ones((1, 1, 4, 4), dtype=np.float64)  # shape (1, 1, 4, 4)
weight = np.ones((1, 1, 3, 3), dtype=np.float64)  # shape (1, 1, 3, 3)

out = conv2d_forward(x, weight, stride=2, padding=1)
# With padding=1, padded input is 6x6
# With stride=2 and kernel 3x3: H_out = (6 - 3) // 2 + 1 = 2
# out shape: (1, 1, 2, 2)
# Each output is sum of 3x3 = 9 for interior positions
# Corner positions have some zeros from padding
```

### Example 3 (with bias and multiple channels)
```python
import numpy as np

x = np.ones((2, 3, 4, 4), dtype=np.float64)  # batch=2, 3 channels, 4x4
weight = np.ones((2, 3, 2, 2), dtype=np.float64)  # 2 filters, 3 in-channels, 2x2
bias = np.array([1.0, -1.0])  # bias for each output channel

out = conv2d_forward(x, weight, bias=bias)
# out shape: (2, 2, 3, 3)
# Each position sums 3 channels * 2*2 kernel = 12, plus bias
# out[:, 0, :, :] = 12 + 1 = 13
# out[:, 1, :, :] = 12 - 1 = 11
```



