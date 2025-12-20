# Batch Normalization Forward and Backward Pass

## Problem
Batch Normalization is a technique to normalize activations in neural networks, making training faster and more stable. During training, it normalizes inputs using batch statistics (mean and variance), then applies a learned scale (gamma) and shift (beta). The backward pass computes gradients for the input, gamma, and beta—this is trickier because the mean and variance depend on all inputs in the batch.

## Task
Implement two functions in NumPy:
1. `batchnorm_forward(x, gamma, beta, eps)` — compute the forward pass of batch normalization
2. `batchnorm_backward(dout, cache)` — compute the backward pass given upstream gradients and cached values

The forward pass should return the normalized output and a cache for use in the backward pass. The backward pass should return gradients with respect to the input, gamma, and beta.

## Function Signature

```python
import numpy as np
from typing import Any

def batchnorm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    ...

def batchnorm_backward(
    dout: np.ndarray,
    cache: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

## Inputs and Outputs

### `batchnorm_forward`
- **Inputs**:
  - `x`: Input tensor of shape `(N, D)` where N is batch size and D is feature dimension
  - `gamma`: Scale parameter of shape `(D,)`
  - `beta`: Shift parameter of shape `(D,)`
  - `eps`: Small constant for numerical stability (default 1e-5)
- **Outputs**: Tuple of:
  - `out`: Normalized output of shape `(N, D)`
  - `cache`: Dictionary containing values needed for backward pass

### `batchnorm_backward`
- **Inputs**:
  - `dout`: Upstream gradient of shape `(N, D)`
  - `cache`: Dictionary from forward pass
- **Outputs**: Tuple of:
  - `dx`: Gradient with respect to input x, shape `(N, D)`
  - `dgamma`: Gradient with respect to gamma, shape `(D,)`
  - `dbeta`: Gradient with respect to beta, shape `(D,)`

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: focus on correctness, not edge cases.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: NumPy (`numpy`) and Python standard library.
- This is training mode batch normalization (use batch statistics, not running averages).

## Examples

### Example 1 (forward pass)
```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
gamma = np.array([1.0, 1.0])
beta = np.array([0.0, 0.0])

out, cache = batchnorm_forward(x, gamma, beta)
# out is approximately:
# [[-1.2247, -1.2247],
#  [ 0.0,     0.0   ],
#  [ 1.2247,  1.2247]]
# (each column has mean≈0 and std≈1)
```

### Example 2 (forward with scale and shift)
```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
gamma = np.array([2.0, 0.5])
beta = np.array([1.0, -1.0])

out, cache = batchnorm_forward(x, gamma, beta)
# out is approximately:
# [[-1.4494, -1.6124],
#  [ 1.0,    -1.0   ],
#  [ 3.4494, -0.3876]]
```

### Example 3 (backward pass)
```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
gamma = np.array([1.0, 1.0])
beta = np.array([0.0, 0.0])

out, cache = batchnorm_forward(x, gamma, beta)
dout = np.ones_like(out)

dx, dgamma, dbeta = batchnorm_backward(dout, cache)
# dbeta ≈ [2.0, 2.0] (sum of dout over batch)
# dgamma ≈ [0.0, 0.0] (sum of dout * x_normalized)
# dx should have each column sum to ≈ 0
```



