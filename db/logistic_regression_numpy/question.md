# Implement Logistic Regression

## Problem
Logistic regression is a classic baseline for binary classification. Given feature vectors \(X\) and binary labels \(y \in \{0, 1\}\), it learns weights \(w\) and bias \(b\) so that:
\[
p(y=1 \mid x) = \sigma(x^\top w + b)
\]
where \(\sigma(\cdot)\) is the sigmoid function.

## Task
Implement a **minimal** logistic regression trainer using **full-batch gradient descent** in NumPy. Your function should:
- initialize parameters to zeros,
- perform `num_steps` gradient descent updates on the **average** binary cross-entropy loss,
- optionally include L2 regularization on `w`,
- return the learned `(w, b)`.

## Function Signature

```python
def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    num_steps: int = 1000,
    l2: float = 0.0,
) -> tuple[np.ndarray, float]:
    ...
```

## Inputs and Outputs
- **inputs**:
  - `X`: `np.ndarray` of shape `(n, d)` (float), design matrix
  - `y`: `np.ndarray` of shape `(n,)` (0/1), binary labels
  - `lr`: learning rate (float)
  - `num_steps`: number of gradient descent steps (int)
  - `l2`: L2 regularization strength (float). Use L2 penalty \(\tfrac{1}{2} \lambda \|w\|_2^2\).
- **outputs**:
  - `w`: `np.ndarray` of shape `(d,)`
  - `b`: `float` scalar bias

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: keep it minimal and fully vectorized.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: NumPy (`numpy`) and Python standard library.
- Use **full-batch** gradient descent and the **average** loss over examples.

## Examples

### Example 1 (one gradient step)

```python
import numpy as np

X = np.array([[1.0, 0.0], [0.0, 1.0]])
y = np.array([1.0, 0.0])

w, b = fit_logistic_regression(X, y, lr=1.0, num_steps=1, l2=0.0)
print(w, b)
# Expected:
# w ≈ [ 0.25 -0.25]
# b = 0.0
```

### Example 2 (one gradient step)

```python
import numpy as np

X = np.array([[0.0], [1.0], [2.0]])
y = np.array([0.0, 0.0, 1.0])

w, b = fit_logistic_regression(X, y, lr=0.5, num_steps=1, l2=0.0)
print(w, b)
# Expected:
# w ≈ [0.08333333]
# b ≈ -0.08333333
```


