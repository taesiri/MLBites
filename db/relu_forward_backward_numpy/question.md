# ReLU (NumPy) Forward + Backward

## Problem
Implement the ReLU (Rectified Linear Unit) activation function and its backward pass using NumPy.

## Task
Implement:
- `relu_forward(x)` to compute \(y = \max(0, x)\) and return `(y, cache)`
- `relu_backward(dy, cache)` to compute `dx` given upstream gradient `dy`

For this question, define the derivative at exactly `x == 0` as **0** (i.e., use the mask `x > 0`).

## Function Signature
```python
import numpy as np

def relu_forward(x):
    """x: np.ndarray -> y, cache"""
    ...

def relu_backward(dy, cache):
    """dy: np.ndarray, cache -> dx"""
    ...
```

## Inputs and Outputs
- **Inputs (`relu_forward`)**
  - **x**: `np.ndarray` of any shape (e.g. `(N, D)`), real-valued
- **Outputs (`relu_forward`)**
  - **y**: `np.ndarray` of same shape as `x`
  - **cache**: any python object you need for backward (typically `x` or a boolean mask)

- **Inputs (`relu_backward`)**
  - **dy**: `np.ndarray` of same shape as `x` (upstream gradient)
  - **cache**: the cache returned by `relu_forward`
- **Outputs (`relu_backward`)**
  - **dx**: `np.ndarray` of same shape as `x` (gradient w.r.t. `x`)

## Constraints
- Use **NumPy only**.
- Keep it short and interview-friendly (vectorized; no unnecessary boilerplate).

## Examples
### Example 1 (forward)
```python
x = np.array([-2.0, 0.0, 3.5])
y, _ = relu_forward(x)
# Expected:
# array([0.0, 0.0, 3.5])
```

### Example 2 (backward)
```python
x = np.array([[-1.0, 0.0, 2.0]])
y, cache = relu_forward(x)
dy = np.array([[10.0, 20.0, 30.0]])
dx = relu_backward(dy, cache)
# Expected (note: derivative at x==0 is defined as 0 here):
# array([[ 0.0,  0.0, 30.0]])
```


