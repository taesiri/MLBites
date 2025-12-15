# Softmax (NumPy)

## Problem
Softmax converts a vector (or tensor) of real-valued scores (often called *logits*) into a probability distribution.

In practice, logits can be large, so a correct implementation must be **numerically stable** (avoid overflow/underflow).

## Task
Implement a numerically-stable `softmax` using **NumPy**.

## Function Signature
```python
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    ...
```

## Inputs and Outputs
- **Inputs**:
  - `x`: `np.ndarray` of any shape (commonly 1D or 2D). Values are logits/scores.
  - `axis`: int, the axis along which to compute the softmax (default `-1`).
- **Output**:
  - `y`: `np.ndarray` with the **same shape** as `x`, where values are in `(0, 1)` and sum to `1` along `axis`.

## Constraints
- Use **NumPy only**.
- Must be **numerically stable** (e.g. should handle logits around `1000` without overflow).
- Keep it interview-friendly (clean, vectorized, ~20 minutes).

## Examples
Example 1 (1D):

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = softmax(x)
# Expected (rounded to 6 decimals):
# np.array([0.090031, 0.244728, 0.665241])
```

Example 2 (2D, row-wise softmax):

```python
import numpy as np

x = np.array([[0.0, 0.0],
              [0.0, 1.0]])
y = softmax(x, axis=1)
# Expected (rounded to 6 decimals):
# np.array([[0.5     , 0.5     ],
#           [0.268941, 0.731059]])
```


