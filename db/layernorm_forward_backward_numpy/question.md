# LayerNorm (NumPy) Forward + Backward

## Problem
Implement Layer Normalization (LayerNorm) for a batch of inputs using NumPy.

## Task
Implement the forward pass `layernorm_forward` and the backward pass `layernorm_backward`.

## Function Signature
```python
import numpy as np

def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x:(N,D), gamma:(D,) or (1,D), beta:(D,) or (1,D) -> y, cache"""
    ...

def layernorm_backward(dy, cache):
    """dy:(N,D) -> dx, dgamma, dbeta"""
    ...
```

## Inputs and Outputs
- **Inputs**:
  - `x`: NumPy array of shape `(N, D)`
  - `gamma`: NumPy array of shape `(D,)` or `(1, D)` (scale)
  - `beta`: NumPy array of shape `(D,)` or `(1, D)` (shift)
  - `eps`: small float for numerical stability
  - `dy`: upstream gradient of shape `(N, D)`
- **Outputs**:
  - `y`: normalized output of shape `(N, D)`
  - `cache`: any objects needed for backward pass
  - `dx`: gradient w.r.t. `x`, shape `(N, D)`
  - `dgamma`: gradient w.r.t. `gamma`, shape `(D,)` or `(1, D)`
  - `dbeta`: gradient w.r.t. `beta`, shape `(D,)` or `(1, D)`

## Constraints
- Use **NumPy only**.
- Normalize over the **feature dimension** `D` independently for each sample (each row).
- Keep it short and interview-friendly (20–30 minutes).

## Examples
Example 1 (using `eps=0` for clean numbers):

```python
x = np.array([[1.0, 2.0],
              [3.0, 4.0]])
gamma = np.array([2.0, 1.0])
beta = np.array([0.5, -0.5])
y, _ = layernorm_forward(x, gamma, beta, eps=0.0)
# Expected:
# array([[-1.5,  0.5],
#        [-1.5,  0.5]])
```

Example 2:

```python
x = np.array([[1.0, 1.0, 1.0]])
gamma = np.ones(3)
beta = np.zeros(3)
y, _ = layernorm_forward(x, gamma, beta)  # eps=1e-5
# Expected:
# array([[0.0, 0.0, 0.0]])
```
# LayerNorm forward/backward (NumPy)

## Problem
Implement Layer Normalization (LayerNorm) forward and backward passes in NumPy for 2D inputs.

## Task
Implement:
- `layernorm_forward(x, gamma, beta, eps=1e-5)` to return `(y, cache)`
- `layernorm_backward(dy, cache)` to return `(dx, dgamma, dbeta)`

LayerNorm is computed **per example** (per row): normalize across the feature dimension \(D\).

## Function Signature

```python
import numpy as np

def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x:(N,D), gamma:(D,) or (1,D), beta:(D,) or (1,D) -> y, cache"""
    raise NotImplementedError

def layernorm_backward(dy, cache):
    """dy:(N,D) -> dx, dgamma, dbeta"""
    raise NotImplementedError
```

## Inputs and Outputs
- **Inputs (`layernorm_forward`)**
  - **x**: `np.ndarray` of shape `(N, D)`
  - **gamma**: `np.ndarray` of shape `(D,)` or `(1, D)` (scale)
  - **beta**: `np.ndarray` of shape `(D,)` or `(1, D)` (shift)
  - **eps**: float, small constant for numerical stability
- **Outputs (`layernorm_forward`)**
  - **y**: `np.ndarray` of shape `(N, D)`
  - **cache**: any python object you need for backward

- **Inputs (`layernorm_backward`)**
  - **dy**: `np.ndarray` of shape `(N, D)` (upstream gradient)
  - **cache**: the cache from `layernorm_forward`
- **Outputs (`layernorm_backward`)**
  - **dx**: `np.ndarray` of shape `(N, D)`
  - **dgamma**: `np.ndarray` of shape `(D,)`
  - **dbeta**: `np.ndarray` of shape `(D,)`

## Constraints
- Use **NumPy only**.
- Implement in a clean, interview-friendly way (vectorized; no unnecessary abstractions).
- Normalize across axis `1` (the feature dimension `D`) for each row.

## Examples
Use `eps=0.0` here to make the numbers cleaner.

### Example 1 (forward, identity scale/shift)

```python
x = np.array([[1., 2., 3.],
              [2., 4., 6.]])
gamma = np.ones(3)
beta = np.zeros(3)
y, cache = layernorm_forward(x, gamma, beta, eps=0.0)
```

Expected `y` (rounded to 6 decimals):

```python
np.array([[-1.224745,  0.      ,  1.224745],
          [-1.224745,  0.      ,  1.224745]])
```

### Example 2 (forward, non-trivial gamma/beta)

```python
gamma = np.array([1., 2., 3.])
beta = np.array([0.5, -1., 2.])
y, cache = layernorm_forward(x, gamma, beta, eps=0.0)
```

Expected `y` (rounded to 6 decimals):

```python
np.array([[-0.724745, -1.      ,  5.674235],
          [-0.724745, -1.      ,  5.674235]])
```

### Example 3 (backward, simple upstream gradient)

```python
y, cache = layernorm_forward(x, np.ones(3), np.zeros(3), eps=0.0)
dy = np.ones_like(x)
dx, dgamma, dbeta = layernorm_backward(dy, cache)
```

Expected (rounded to 6 decimals):
- `dbeta = [2., 2., 2.]`
- `dgamma = [-2.449490, 0., 2.449490]`
- `dx` is all zeros (shape `(2,3)`)


