# K-Nearest Neighbors (kNN)

## Problem Statement

Implement **K-Nearest Neighbors (kNN)** from scratch using PyTorch/NumPy. kNN is a simple yet powerful algorithm that classifies samples based on the majority vote of their k nearest neighbors.

Your task is to:

1. Implement distance computation (Euclidean, Manhattan)
2. Find the k nearest neighbors for each query point
3. Predict labels using majority voting
4. Support both classification and regression

## Requirements

- Do **NOT** use sklearn's KNeighborsClassifier
- Implement efficient distance computation using broadcasting
- Support different distance metrics  
- Handle ties in voting

## Function Signature

```python
class KNN:
    def __init__(self, k: int = 5, distance: str = 'euclidean'):
        pass
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Store training data."""
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict labels for query points."""
        pass
```

## Example

```python
# Training data
X_train = torch.randn(100, 2)
y_train = torch.randint(0, 3, (100,))

# Create and fit kNN
knn = KNN(k=5)
knn.fit(X_train, y_train)

# Predict
X_test = torch.randn(10, 2)
predictions = knn.predict(X_test)
```

## Distance Formulas

| Metric | Formula |
|--------|---------|
| Euclidean | `√(Σ(x_i - y_i)²)` |
| Manhattan | `Σ|x_i - y_i|` |
| Cosine | `1 - (x·y)/(||x|| ||y||)` |

## Hints

- Use `torch.cdist` or manual broadcasting for efficient distance computation
- `torch.topk` can find the k smallest distances
- `torch.mode` or `torch.bincount` for majority voting
