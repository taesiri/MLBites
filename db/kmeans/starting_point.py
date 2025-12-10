"""
K-Means Clustering - Starting Point

Implement K-Means from scratch.
Fill in the TODO sections to complete the implementation.
"""

import torch


class KMeans:
    """K-Means clustering algorithm."""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, tol: float = 1e-4, init: str = 'kmeans++'):
        """
        Initialize K-Means.
        
        Args:
            n_clusters: Number of clusters (k)
            max_iter: Maximum iterations
            tol: Tolerance for convergence
            init: Initialization method ('random' or 'kmeans++')
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self._inertia = None
    
    def _init_centroids_random(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize centroids by randomly selecting k data points."""
        # TODO: Randomly select k indices and return those points
        pass
    
    def _init_centroids_kmeanspp(self, X: torch.Tensor) -> torch.Tensor:
        """
        Initialize centroids using k-means++ algorithm.
        
        1. Choose first centroid randomly
        2. For each subsequent centroid:
           - Compute squared distance to nearest centroid
           - Choose next centroid with probability proportional to distance
        """
        # TODO: Implement k-means++ initialization
        pass
    
    def _compute_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute distances from each point to each centroid."""
        # TODO: Return distance matrix (n_samples, n_clusters)
        pass
    
    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        """Assign each point to nearest centroid."""
        # TODO: Return cluster assignments (n_samples,)
        pass
    
    def _update_centroids(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Update centroids as mean of assigned points."""
        # TODO: Compute new centroids
        # Handle empty clusters
        pass
    
    def fit(self, X: torch.Tensor) -> 'KMeans':
        """
        Fit K-Means to data.
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            self
        """
        # TODO: Initialize centroids
        
        # TODO: Iterate until convergence
        # 1. Assign clusters
        # 2. Update centroids
        # 3. Check convergence
        
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Assign points to nearest cluster."""
        # TODO: Return cluster assignments
        pass
    
    @property
    def inertia_(self) -> float:
        """Total within-cluster sum of squares."""
        return self._inertia


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate sample clustered data
    cluster1 = torch.randn(30, 2) + torch.tensor([0.0, 0.0])
    cluster2 = torch.randn(30, 2) + torch.tensor([5.0, 0.0])
    cluster3 = torch.randn(30, 2) + torch.tensor([2.5, 4.0])
    X = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=3, init='kmeans++')
    kmeans.fit(X)
    
    labels = kmeans.predict(X)
    
    print(f"Cluster assignments: {labels[:10].tolist()}...")
    print(f"Centroids:\n{kmeans.centroids}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
