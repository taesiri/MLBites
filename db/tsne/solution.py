"""
t-SNE from Scratch - Solution

Complete implementation of t-SNE from scratch.
"""

import torch
import numpy as np


class TSNE:
    """t-Distributed Stochastic Neighbor Embedding."""
    
    def __init__(
        self, 
        n_components: int = 2, 
        perplexity: float = 30.0,
        learning_rate: float = 200.0, 
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        early_exaggeration_iter: int = 250
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
    
    def _compute_pairwise_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared Euclidean distances."""
        sum_X = (X ** 2).sum(dim=1)
        D = sum_X.unsqueeze(0) + sum_X.unsqueeze(1) - 2 * X @ X.T
        return torch.clamp(D, min=0)
    
    def _compute_conditional_prob(self, D_row: torch.Tensor, sigma: float, i: int) -> torch.Tensor:
        """Compute P_j|i for a single row."""
        P = torch.exp(-D_row / (2 * sigma ** 2))
        P[i] = 0  # Set self-similarity to 0
        sum_P = P.sum()
        if sum_P > 1e-10:
            P = P / sum_P
        return P
    
    def _compute_perplexity_value(self, P: torch.Tensor) -> float:
        """Compute perplexity from probability distribution."""
        entropy = -torch.sum(P * torch.log2(P + 1e-10))
        return 2 ** entropy.item()
    
    def _binary_search_sigma(self, D_row: torch.Tensor, i: int, tol: float = 1e-5, max_iter: int = 50) -> float:
        """Find sigma using binary search to match target perplexity."""
        sigma_min, sigma_max = 1e-10, 1e4
        sigma = 1.0
        
        for _ in range(max_iter):
            P = self._compute_conditional_prob(D_row, sigma, i)
            perp = self._compute_perplexity_value(P)
            
            if abs(perp - self.perplexity) < tol:
                break
            
            if perp > self.perplexity:
                sigma_max = sigma
            else:
                sigma_min = sigma
            
            sigma = (sigma_min + sigma_max) / 2
        
        return sigma
    
    def _compute_high_dim_affinities(self, X: torch.Tensor) -> torch.Tensor:
        """Compute symmetric pairwise affinities P."""
        n = X.shape[0]
        D = self._compute_pairwise_distances(X)
        P = torch.zeros(n, n)
        
        for i in range(n):
            sigma = self._binary_search_sigma(D[i], i)
            P[i] = self._compute_conditional_prob(D[i], sigma, i)
        
        # Symmetrize
        P = (P + P.T) / (2 * n)
        P = torch.clamp(P, min=1e-12)
        
        return P
    
    def _compute_low_dim_affinities(self, Y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise affinities Q using Student's t distribution."""
        D = self._compute_pairwise_distances(Y)
        
        # Student's t with 1 degree of freedom (Cauchy)
        Q = 1 / (1 + D)
        Q.fill_diagonal_(0)
        
        # Normalize
        Q = Q / Q.sum()
        Q = torch.clamp(Q, min=1e-12)
        
        return Q
    
    def _compute_gradient(self, P: torch.Tensor, Q: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute gradient of KL divergence."""
        n = Y.shape[0]
        D = self._compute_pairwise_distances(Y)
        
        # (P - Q) * (1 + D)^(-1)
        PQ_diff = P - Q
        inv_dist = 1 / (1 + D)
        
        grad = torch.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y  # (n, d)
            grad[i] = 4 * (PQ_diff[i] * inv_dist[i]).unsqueeze(1) * diff
            grad[i] = grad[i].sum(dim=0)
        
        return grad
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Reduce dimensionality of X."""
        n = X.shape[0]
        
        # Compute high-dimensional affinities
        P = self._compute_high_dim_affinities(X)
        
        # Early exaggeration
        P = P * self.early_exaggeration
        
        # Initialize Y randomly
        Y = torch.randn(n, self.n_components) * 1e-4
        
        # Gradient descent with momentum
        velocity = torch.zeros_like(Y)
        momentum = 0.5
        
        for iteration in range(self.n_iter):
            # Compute Q
            Q = self._compute_low_dim_affinities(Y)
            
            # Compute gradient
            grad = self._compute_gradient(P, Q, Y)
            
            # Update with momentum
            velocity = momentum * velocity - self.learning_rate * grad
            Y = Y + velocity
            
            # Center Y
            Y = Y - Y.mean(dim=0)
            
            # Increase momentum after initial iterations
            if iteration == 250:
                momentum = 0.8
            
            # Stop early exaggeration
            if iteration == self.early_exaggeration_iter:
                P = P / self.early_exaggeration
        
        return Y


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate sample data (3 clusters)
    X = torch.cat([
        torch.randn(30, 10) + torch.tensor([0.0] * 10),
        torch.randn(30, 10) + torch.tensor([5.0] * 10),
        torch.randn(30, 10) + torch.tensor([10.0] * 10),
    ])
    
    print(f"Input shape: {X.shape}")
    
    # Run t-SNE (use fewer iterations for demo)
    tsne = TSNE(n_components=2, perplexity=10, n_iter=500)
    Y = tsne.fit_transform(X)
    
    print(f"Output shape: {Y.shape}")
    print(f"Output range: [{Y.min():.2f}, {Y.max():.2f}]")
    
    # Clusters should be separated
    print("\nCluster centroids in 2D:")
    print(f"  Cluster 1: {Y[:30].mean(dim=0).tolist()}")
    print(f"  Cluster 2: {Y[30:60].mean(dim=0).tolist()}")
    print(f"  Cluster 3: {Y[60:].mean(dim=0).tolist()}")
