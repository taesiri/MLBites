# Linear (Affine) Layer Forward + Backward (NumPy)

## Problem
Implement the forward and backward passes of a fully-connected (linear/affine) layer using NumPy.

## Task
Implement:
- `linear_forward(x, W, b)` to compute \(y = xW + b\) and return `(y, cache)`
- `linear_backward(dy, cache)` to compute gradients `(dx, dW, db)` given upstream gradient `dy`

## Function Signature
```python
import numpy as np

def linear_forward(x, W, b):
    """x:(N,D), W:(D,M), b:(M,) -> y, cache"""
    ...

def linear_backward(dy, cache):
    """dy:(N,M) -> dx, dW, db"""
    ...
```

## Inputs and Outputs
- **Inputs (`linear_forward`)**
  - **x**: `np.ndarray` of shape `(N, D)` (batch of inputs)
  - **W**: `np.ndarray` of shape `(D, M)` (weights)
  - **b**: `np.ndarray` of shape `(M,)` (bias)
- **Outputs (`linear_forward`)**
  - **y**: `np.ndarray` of shape `(N, M)` (outputs)
  - **cache**: any python object you need for backward (e.g., tensors from forward)

- **Inputs (`linear_backward`)**
  - **dy**: `np.ndarray` of shape `(N, M)` (upstream gradient)
  - **cache**: the cache returned by `linear_forward`
- **Outputs (`linear_backward`)**
  - **dx**: `np.ndarray` of shape `(N, D)` (gradient w.r.t. `x`)
  - **dW**: `np.ndarray` of shape `(D, M)` (gradient w.r.t. `W`)
  - **db**: `np.ndarray` of shape `(M,)` (gradient w.r.t. `b`)

## Constraints
- Use **NumPy only**.
- Keep it short and interview-friendly (20–30 minutes).
- You may assume inputs are 2D (`x` is `(N, D)`).

## Examples
### Example 1
```python
x = np.array([[1., 2.],
              [3., 4.]])
W = np.array([[1., 0.],
              [0., 1.]])
b = np.array([1., -1.])

y, _ = linear_forward(x, W, b)
# Expected:
# array([[2., 1.],
#        [4., 3.]])
```

### Example 2
```python
x = np.array([[ 1., -1.]])
W = np.array([[ 2.0, -1.0],
              [ 0.5,  3.0]])
b = np.array([0.0, 1.0])

y, _ = linear_forward(x, W, b)
# Expected:
# array([[ 1.5, -3. ]])
```


