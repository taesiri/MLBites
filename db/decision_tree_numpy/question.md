# Decision Tree Classifier

## Problem
A decision tree is a supervised learning algorithm that recursively splits data based on feature thresholds to create a tree structure for classification. At each node, the algorithm finds the best split that minimizes impurity (e.g., Gini impurity) in the resulting child nodes.

## Task
Implement the two core building blocks of a decision tree classifier in NumPy:
- `compute_gini(y)`: compute the Gini impurity of a set of labels
- `find_best_split(X, y)`: find the best feature and threshold to split the data

These two functions form the foundation for building decision tree classifiers.

## Function Signature

```python
import numpy as np

def compute_gini(y: np.ndarray) -> float:
    """Compute Gini impurity of a label array."""
    ...

def find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Find the best feature and threshold to split the data."""
    ...
```

## Inputs and Outputs
- **inputs**:
  - `compute_gini`:
    - `y`: NumPy array of shape `(n_samples,)` with integer class labels
  - `find_best_split`:
    - `X`: NumPy array of shape `(n_samples, n_features)` (float)
    - `y`: NumPy array of shape `(n_samples,)` with integer class labels
- **outputs**:
  - `compute_gini`: float, the Gini impurity (between 0 and 1)
  - `find_best_split`: tuple of `(best_feature_idx, best_threshold, best_gini)`
    - `best_feature_idx`: int, index of the feature to split on
    - `best_threshold`: float, threshold value for the split
    - `best_gini`: float, weighted Gini impurity after the split

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal implementation without extra features.
- Assume inputs satisfy the documented contract; avoid extra validation.
- For `find_best_split`: consider all unique values of each feature as potential thresholds. Split rule: left child gets samples where `X[:, feature] <= threshold`.
- If no valid split exists (e.g., all samples identical), return `(-1, 0.0, float('inf'))`.
- Allowed libs: NumPy (`numpy`) and Python standard library.

## Examples

### Example 1 (pure node)
```python
import numpy as np

y = np.array([0, 0, 0, 0])
gini = compute_gini(y)
# expected: 0.0 (perfectly pure)
```

### Example 2 (maximum impurity for binary)
```python
import numpy as np

y = np.array([0, 0, 1, 1])
gini = compute_gini(y)
# expected: 0.5 (maximum impurity for 2 classes)
```

### Example 3 (three classes)
```python
import numpy as np

y = np.array([0, 1, 2])
gini = compute_gini(y)
# expected: 0.6666... (1 - 1/9 - 1/9 - 1/9 = 1 - 1/3 = 2/3)
```

### Example 4 (simple split)
```python
import numpy as np

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])
y = np.array([0, 0, 1, 1])

best_feature, best_threshold, best_gini = find_best_split(X, y)
# expected: (0, 2.0, 0.0)
# Splitting at x <= 2.0 gives left=[0,0] (gini=0), right=[1,1] (gini=0)
# Weighted gini = 0.5 * 0 + 0.5 * 0 = 0.0
```

### Example 5 (two features)
```python
import numpy as np

X = np.array([
    [1.0, 5.0],
    [2.0, 4.0],
    [3.0, 3.0],
    [4.0, 2.0]
])
y = np.array([0, 0, 1, 1])

best_feature, best_threshold, best_gini = find_best_split(X, y)
# expected: (0, 2.0, 0.0) or (1, 3.0, 0.0)
# Both features can achieve perfect split with gini=0
```


