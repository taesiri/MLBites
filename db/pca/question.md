# PCA from Scratch

## Problem Statement

Implement **Principal Component Analysis (PCA)** from scratch. PCA is a linear dimensionality reduction technique that projects data onto directions of maximum variance.

Your task is to:

1. Center the data (mean subtraction)
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors
4. Project data onto top-k principal components

## Requirements

- Do **NOT** use sklearn's PCA
- Implement using SVD or eigendecomposition
- Return both transformed data and components
- Support explained variance ratio

## Function Signature

```python
class PCA:
    def __init__(self, n_components: int):
        pass
    
    def fit(self, X: torch.Tensor) -> 'PCA':
        """Compute principal components."""
        pass
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project data to lower dimension."""
        pass
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        pass
    
    @property
    def explained_variance_ratio_(self) -> torch.Tensor:
        """Proportion of variance explained by each component."""
        pass
```

## PCA Algorithm

```
1. Center data: X_centered = X - mean(X)
2. Compute covariance: C = X_centered.T @ X_centered / (n-1)
3. Eigendecomposition: C = V @ Λ @ V.T
4. Sort by eigenvalue (descending)
5. Take top-k eigenvectors
6. Project: X_reduced = X_centered @ V[:, :k]
```

## Example

```python
X = torch.randn(100, 10)  # 100 samples, 10 features

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)  # 100 samples, 3 features

print(f"Explained variance: {pca.explained_variance_ratio_}")
```

## Hints

- Use `torch.linalg.eigh` for eigendecomposition (symmetric matrix)
- Or use `torch.linalg.svd` directly: U, S, V = svd(X_centered)
- Components are in V (right singular vectors)
- Variance is proportional to squared singular values
