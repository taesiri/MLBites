"""
PCA from Scratch - Solution

Complete implementation of PCA from scratch.
"""

import torch


class PCA:
    """Principal Component Analysis using eigendecomposition."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self._explained_variance_ratio = None
    
    def fit(self, X: torch.Tensor) -> 'PCA':
        """Compute principal components."""
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = X_centered.T @ X_centered / (n_samples - 1)
        
        # Eigendecomposition (returns ascending order)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store top-k components
        self.components_ = eigenvectors[:, :self.n_components].T  # (n_components, n_features)
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Compute explained variance ratio
        total_variance = eigenvalues.sum()
        self._explained_variance_ratio = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project data onto principal components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced: torch.Tensor) -> torch.Tensor:
        """Reconstruct original data from reduced representation."""
        return X_reduced @ self.components_ + self.mean_
    
    @property
    def explained_variance_ratio_(self) -> torch.Tensor:
        return self._explained_variance_ratio


class PCASVD:
    """PCA using SVD (often more numerically stable)."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self._explained_variance_ratio = None
    
    def fit(self, X: torch.Tensor) -> 'PCASVD':
        """Fit using SVD."""
        n_samples = X.shape[0]
        
        # Center data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Store components
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]
        
        # Explained variance
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        self._explained_variance_ratio = explained_variance[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)
    
    @property
    def explained_variance_ratio_(self) -> torch.Tensor:
        return self._explained_variance_ratio


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create sample data
    X = torch.randn(100, 10)
    
    # Fit PCA (eigenvalue method)
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    
    print("=== PCA (Eigenvalue) ===")
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.tolist()}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Test reconstruction
    X_reconstructed = pca.inverse_transform(X_reduced)
    recon_error = ((X - X_reconstructed) ** 2).mean()
    print(f"Reconstruction error: {recon_error:.4f}")
    
    # Fit PCA (SVD method)
    pca_svd = PCASVD(n_components=3)
    X_reduced_svd = pca_svd.fit_transform(X)
    
    print("\n=== PCA (SVD) ===")
    print(f"Explained variance ratio: {pca_svd.explained_variance_ratio_.tolist()}")
