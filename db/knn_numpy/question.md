# K-Nearest Neighbors Classifier

## Problem
K-Nearest Neighbors (KNN) is a simple, non-parametric classification algorithm. Given a test point, it finds the K closest training points and predicts the majority class among them.

## Task
Implement a KNN classifier in NumPy:
- `knn_predict(X_train, y_train, X_test, k)`: predict labels for test points based on the K nearest neighbors from the training set.

## Function Signature

```python
import numpy as np

def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int
) -> np.ndarray:
    """Predict labels for test points using K-Nearest Neighbors."""
    ...
```

## Inputs and Outputs
- **inputs**:
  - `X_train`: NumPy array of shape `(n_train, n_features)` — training data points (float)
  - `y_train`: NumPy array of shape `(n_train,)` — training labels (integers in `[0, num_classes-1]`)
  - `X_test`: NumPy array of shape `(n_test, n_features)` — test data points (float)
  - `k`: int, number of neighbors to consider (1 ≤ k ≤ n_train)
- **outputs**:
  - NumPy array of shape `(n_test,)` — predicted labels for each test point (integers)

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal implementation without extra features.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Use Euclidean distance to measure similarity.
- For tie-breaking in majority voting, return the smallest label among the tied classes.
- Allowed libs: NumPy (`numpy`) and Python standard library.

## Examples

### Example 1 (simple 2D classification)
```python
import numpy as np

X_train = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [5.0, 5.0],
    [6.0, 5.0],
    [5.0, 6.0]
])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [0.5, 0.5],   # closer to class 0
    [5.5, 5.5]    # closer to class 1
])

predictions = knn_predict(X_train, y_train, X_test, k=3)
# expected: array([0, 1])
```

### Example 2 (k=1, nearest neighbor)
```python
import numpy as np

X_train = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
y_train = np.array([0, 0, 0, 1, 1])

X_test = np.array([[0.5], [10.5]])

predictions = knn_predict(X_train, y_train, X_test, k=1)
# expected: array([0, 1])
```

### Example 3 (tie-breaking)
```python
import numpy as np

X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
y_train = np.array([0, 1, 0, 1])

X_test = np.array([[1.5]])  # equidistant from 1.0 (label 1) and 2.0 (label 0)

predictions = knn_predict(X_train, y_train, X_test, k=2)
# expected: array([0])  # tie between labels 0 and 1, return smallest (0)
```




