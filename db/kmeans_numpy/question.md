# K-Means Clustering

## Problem
K-means is a classic unsupervised learning algorithm that partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids to be the mean of assigned points.

## Task
Implement the two core steps of k-means clustering in NumPy:
- `assign_clusters(X, centroids)`: assign each point to the nearest centroid
- `update_centroids(X, assignments, k)`: compute new centroids as the mean of assigned points

These two functions form the building blocks of the k-means algorithm.

## Function Signature

```python
import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest centroid."""
    ...

def update_centroids(X: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
    """Update centroids to be the mean of assigned points."""
    ...
```

## Inputs and Outputs
- **inputs**:
  - `assign_clusters`:
    - `X`: NumPy array of shape `(n_samples, n_features)` (float)
    - `centroids`: NumPy array of shape `(k, n_features)` (float)
  - `update_centroids`:
    - `X`: NumPy array of shape `(n_samples, n_features)` (float)
    - `assignments`: NumPy array of shape `(n_samples,)` with integer cluster indices in `[0, k-1]`
    - `k`: int, the number of clusters
- **outputs**:
  - `assign_clusters`: NumPy array of shape `(n_samples,)` with integer cluster indices
  - `update_centroids`: NumPy array of shape `(k, n_features)` with updated centroid positions

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal implementation without extra features.
- Assume inputs satisfy the documented contract; avoid extra validation.
- For `update_centroids`: if a cluster has no assigned points, keep its centroid at the origin (all zeros).
- Allowed libs: NumPy (`numpy`) and Python standard library.

## Examples

### Example 1 (simple 2D clustering)
```python
import numpy as np

X = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [5.0, 5.0],
    [6.0, 5.0]
])
centroids = np.array([
    [0.5, 0.0],  # cluster 0
    [5.5, 5.0]   # cluster 1
])

assignments = assign_clusters(X, centroids)
# expected: array([0, 0, 1, 1])

new_centroids = update_centroids(X, assignments, k=2)
# expected: array([[0.5, 0.0], [5.5, 5.0]])
```

### Example 2 (1D data)
```python
import numpy as np

X = np.array([[1.0], [2.0], [10.0], [11.0], [12.0]])
centroids = np.array([[0.0], [15.0]])

assignments = assign_clusters(X, centroids)
# expected: array([0, 0, 1, 1, 1])

new_centroids = update_centroids(X, assignments, k=2)
# expected: array([[1.5], [11.0]])
```

### Example 3 (empty cluster)
```python
import numpy as np

X = np.array([[1.0, 1.0], [2.0, 2.0]])
assignments = np.array([0, 0])  # all points in cluster 0, cluster 1 is empty

new_centroids = update_centroids(X, assignments, k=2)
# expected: array([[1.5, 1.5], [0.0, 0.0]])  # empty cluster at origin
```


