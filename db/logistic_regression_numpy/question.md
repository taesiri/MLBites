# Logistic Regression from Scratch

## Problem

Logistic regression is a fundamental binary classification algorithm. It uses the sigmoid function to map linear combinations of features to probabilities, and is trained by minimizing the binary cross-entropy loss using gradient descent.

## Task

Implement the following functions in NumPy:

1. `sigmoid(z)` — compute the sigmoid activation
2. `compute_loss(X, y, w, b)` — compute the binary cross-entropy loss
3. `compute_gradients(X, y, w, b)` — compute gradients of loss w.r.t. weights and bias
4. `train(X, y, lr, n_iters)` — train logistic regression using gradient descent

## Function Signature

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray: ...

def compute_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float: ...

def compute_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]: ...

def train(
    X: np.ndarray, y: np.ndarray, lr: float, n_iters: int
) -> tuple[np.ndarray, float]: ...
```

## Inputs and Outputs

### `sigmoid`
- **Input**: `z` — array of any shape, float
- **Output**: element-wise sigmoid values, same shape as input

### `compute_loss`
- **Input**:
  - `X` — shape `(n_samples, n_features)`, feature matrix
  - `y` — shape `(n_samples,)`, binary labels (0 or 1)
  - `w` — shape `(n_features,)`, weight vector
  - `b` — scalar bias term
- **Output**: scalar float, the mean binary cross-entropy loss

### `compute_gradients`
- **Input**: same as `compute_loss`
- **Output**: tuple `(dw, db)` where:
  - `dw` — shape `(n_features,)`, gradient w.r.t. weights
  - `db` — scalar, gradient w.r.t. bias

### `train`
- **Input**:
  - `X` — shape `(n_samples, n_features)`, feature matrix
  - `y` — shape `(n_samples,)`, binary labels (0 or 1)
  - `lr` — learning rate (positive float)
  - `n_iters` — number of gradient descent iterations
- **Output**: tuple `(w, b)` — trained weights and bias (initialized to zeros)

## Constraints

- Must be solvable in 20–30 minutes.
- Interview-friendly: no frameworks, no extra features.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: NumPy (`numpy`) and Python standard library.
- Use numerical stability best practices (clip values to avoid log(0)).

## Examples

### Example 1 (sigmoid)
```python
import numpy as np

z = np.array([0.0, 2.0, -2.0])
s = sigmoid(z)
# expected: [0.5, 0.88079708, 0.11920292]
```

### Example 2 (loss computation)
```python
import numpy as np

X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
y = np.array([0, 0, 1])
w = np.array([0.1, 0.2])
b = -0.5

loss = compute_loss(X, y, w, b)
# expected: approximately 0.546
```

### Example 3 (training)
```python
import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0, 0, 0, 1])  # AND gate
w, b = train(X, y, lr=1.0, n_iters=1000)
# After training, sigmoid(X @ w + b) should be close to y
```
