# Linear Regression

## Problem
Linear regression fits a line (or hyperplane) that best predicts a continuous target from input features by minimizing mean squared error.

## Task
Implement a minimal ordinary least squares (OLS) linear regression model in NumPy with:
- `fit(X, y)` to learn coefficients (and an optional intercept)
- `predict(X)` to produce predictions

Use a least-squares solve (e.g., `np.linalg.lstsq`) rather than explicitly computing a matrix inverse.

## Function Signature

```python
import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept: bool = True) -> None: ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

## Inputs and Outputs
- **inputs**:
  - `X`: NumPy array of shape `(n_samples, n_features)` (float)
  - `y`: NumPy array of shape `(n_samples,)` (float)
  - `fit_intercept`: if `True`, learn an intercept term
- **outputs**:
  - `fit(...)` returns `self` and sets:
    - `coef_`: NumPy array of shape `(n_features,)`
    - `intercept_`: float
  - `predict(X)` returns NumPy array of shape `(n_samples,)`

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal implementation (no scikit-learn, no extra features).
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: NumPy (`numpy`) and Python standard library.

## Examples

### Example 1 (1 feature with intercept)
```python
import numpy as np

X = np.array([[0.0], [1.0], [2.0]])
y = np.array([1.0, 3.0, 5.0])  # y = 2*x + 1

model = LinearRegression(fit_intercept=True).fit(X, y)
model.coef_      # expected: array([2.0])
model.intercept_ # expected: 1.0

model.predict(np.array([[3.0], [4.0]]))  # expected: array([7.0, 9.0])
```

### Example 2 (2 features, no noise)
```python
import numpy as np

X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([3.5, -1.5, 1.5])  # y = 3*x1 - 2*x2 + 0.5

model = LinearRegression(fit_intercept=True).fit(X, y)
model.coef_      # expected: array([3.0, -2.0])
model.intercept_ # expected: 0.5
```


