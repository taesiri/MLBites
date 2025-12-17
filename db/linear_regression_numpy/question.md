# Linear Regression

## Problem
Linear regression is one of the most fundamental supervised learning algorithms. It fits a linear model to data by finding the weights that minimize the sum of squared errors between predictions and targets. The closed-form solution (normal equation) allows computing the optimal weights directly without iterative optimization.

## Task
Implement linear regression using the normal equation in NumPy:
- `fit(X, y)`: compute optimal weights using the closed-form solution
- `predict(X, weights)`: compute predictions for given features and weights

These two functions form the core of a simple linear regression model.

## Function Signature

```python
import numpy as np

def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit linear regression weights using the normal equation."""
    ...

def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute predictions given features and weights."""
    ...
```

## Inputs and Outputs
- **inputs**:
  - `fit`:
    - `X`: NumPy array of shape `(n_samples, n_features)` (float) — feature matrix
    - `y`: NumPy array of shape `(n_samples,)` (float) — target values
  - `predict`:
    - `X`: NumPy array of shape `(n_samples, n_features)` (float) — feature matrix
    - `weights`: NumPy array of shape `(n_features,)` (float) — model weights
- **outputs**:
  - `fit`: NumPy array of shape `(n_features,)` with the optimal weights
  - `predict`: NumPy array of shape `(n_samples,)` with predicted values

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal implementation without extra features.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Assume `X` has full column rank (the normal equation is solvable).
- No bias/intercept term is required — assume it's included in `X` if needed.
- Allowed libs: NumPy (`numpy`) and Python standard library.

## Examples

### Example 1 (simple 1D regression)
```python
import numpy as np

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.0, 6.0, 8.0])

weights = fit(X, y)
# expected: array([2.0])

predictions = predict(X, weights)
# expected: array([2.0, 4.0, 6.0, 8.0])
```

### Example 2 (2D features)
```python
import numpy as np

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])
y = np.array([1.0, 2.0, 3.0])

weights = fit(X, y)
# expected: array([1.0, 2.0])

predictions = predict(X, weights)
# expected: array([1.0, 2.0, 3.0])
```

### Example 3 (with bias column)
```python
import numpy as np

# Include bias as first column of ones
X = np.array([
    [1.0, 1.0],
    [1.0, 2.0],
    [1.0, 3.0]
])
y = np.array([3.0, 5.0, 7.0])  # y = 1 + 2*x

weights = fit(X, y)
# expected: array([1.0, 2.0])  # bias=1, slope=2

predictions = predict(X, weights)
# expected: array([3.0, 5.0, 7.0])
```
