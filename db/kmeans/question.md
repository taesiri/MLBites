# K-Means Clustering

## Problem Statement

Implement **K-Means Clustering** from scratch. K-Means is an iterative algorithm that partitions data into k clusters by minimizing within-cluster variance.

Your task is to:

1. Initialize centroids (random or k-means++)
2. Iterate: assign points to nearest centroid, update centroids
3. Handle convergence detection
4. Support different initialization methods

## Requirements

- Do **NOT** use sklearn's KMeans
- Implement both random and k-means++ initialization
- Return cluster assignments and centroids
- Handle empty cluster edge cases

## Function Signature

```python
class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, init: str = 'kmeans++'):
        pass
    
    def fit(self, X: torch.Tensor) -> 'KMeans':
        """Fit K-Means to data."""
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Assign points to nearest cluster."""
        pass
    
    @property
    def inertia_(self) -> float:
        """Total within-cluster sum of squares."""
        pass
```

## K-Means Algorithm

```
1. Initialize k centroids
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids as mean of assigned points
3. Return assignments and centroids
```

## Example

```python
X = torch.randn(100, 2)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

labels = kmeans.predict(X)
centroids = kmeans.centroids
```

## Hints

- K-means++ initialization: choose first centroid randomly, then choose subsequent centroids with probability proportional to squared distance from nearest centroid
- Check for empty clusters and handle by reinitializing
- Convergence: centroids don't change or max iterations reached
