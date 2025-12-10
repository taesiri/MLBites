"""
PCA from Scratch - Starting Point

Implement PCA from scratch.
Fill in the TODO sections to complete the implementation.
"""

import torch


class PCA:
    """Principal Component Analysis."""
    
    def __init__(self, n_components: int):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self._explained_variance_ratio = None
    
    def fit(self, X: torch.Tensor) -> 'PCA':
        """
        Compute principal components.
        
        Args:
            X: Data matrix (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # TODO: Center the data (subtract mean)
        
        # TODO: Compute covariance matrix
        # C = X_centered.T @ X_centered / (n_samples - 1)
        
        # TODO: Eigendecomposition
        # Use torch.linalg.eigh (for symmetric matrices)
        
        # TODO: Sort eigenvalues/vectors in descending order
        
        # TODO: Store top-k components
        
        # TODO: Compute explained variance ratio
        
        pass
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project data onto principal components."""
        # TODO: Center and project
        pass
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced: torch.Tensor) -> torch.Tensor:
        """Reconstruct original data from reduced representation."""
        # TODO: Project back and add mean
        pass
    
    @property
    def explained_variance_ratio_(self) -> torch.Tensor:
        return self._explained_variance_ratio


class PCASVD:
    """PCA using SVD (often more numerically stable)."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
    
    def fit(self, X: torch.Tensor) -> 'PCASVD':
        """Fit using SVD."""
        # TODO: Center data
        
        # TODO: Compute SVD
        # U, S, Vt = torch.linalg.svd(X_centered)
        
        # TODO: Store components (rows of Vt)
        
        pass
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create sample data
    X = torch.randn(100, 10)
    
    # Fit PCA
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
