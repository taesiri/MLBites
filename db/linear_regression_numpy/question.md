# Linear Regression Fit + Predict (NumPy)

## Problem
Implement basic **linear regression** in NumPy: fit weights from data, then use them to make predictions.

## Task
Implement two functions:
- `linear_regression_fit`: compute weights (and optional bias/intercept) that minimize squared error, with optional L2 (ridge) regularization.
- `linear_regression_predict`: use the learned parameters to predict targets for new inputs.

## Function Signature
```python
import numpy as np

def linear_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2_reg: float = 0.0,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float]:
    ...

def linear_regression_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    ...
```

## Inputs and Outputs
- **Inputs (`linear_regression_fit`)**:
  - `X`: `np.ndarray` of shape `(N, D)` (features)
  - `y`: `np.ndarray` of shape `(N,)` or `(N, 1)` (targets)
  - `l2_reg`: non-negative float, L2 regularization strength (ridge)
  - `fit_intercept`: if `True`, fit an intercept term `b`
- **Outputs (`linear_regression_fit`)**:
  - `w`: `np.ndarray` of shape `(D,)` (weights)
  - `b`: float (intercept). If `fit_intercept=False`, return `b = 0.0`.

- **Inputs (`linear_regression_predict`)**:
  - `X`: `np.ndarray` of shape `(N, D)`
  - `w`: `np.ndarray` of shape `(D,)`
  - `b`: float
- **Outputs (`linear_regression_predict`)**:
  - `y_pred`: `np.ndarray` of shape `(N,)`

## Constraints
- Use **NumPy only**.
- Keep it interview-friendly (20–30 minutes): vectorized math, minimal boilerplate.
- `l2_reg` must be non-negative.
- If you use ridge regularization, **do not regularize the intercept**.

## Examples
### Example 1 (perfect line with intercept)
```python
X = np.array([[0.0], [1.0], [2.0]])
y = np.array([1.0, 3.0, 5.0])  # y = 2*x + 1
w, b = linear_regression_fit(X, y, l2_reg=0.0, fit_intercept=True)
y_pred = linear_regression_predict(X, w, b)

# Expected:
# w == array([2.0])
# b == 1.0
# y_pred == array([1.0, 3.0, 5.0])
```

### Example 2 (2D features, no intercept)
```python
X = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])
y = np.array([1.0, 2.0, 3.0])  # y = 1*x1 + 2*x2
w, b = linear_regression_fit(X, y, l2_reg=0.0, fit_intercept=False)
y_pred = linear_regression_predict(X, w, b)

# Expected:
# w == array([1.0, 2.0])
# b == 0.0
# y_pred == array([1.0, 2.0, 3.0])
```


