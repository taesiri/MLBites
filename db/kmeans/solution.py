"""
K-Means Clustering - Solution

Complete implementation of K-Means from scratch.
"""

import torch


class KMeans:
    """K-Means clustering algorithm."""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, tol: float = 1e-4, init: str = 'kmeans++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self._inertia = None
    
    def _init_centroids_random(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize centroids by randomly selecting k data points."""
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples)[:self.n_clusters]
        return X[indices].clone()
    
    def _init_centroids_kmeanspp(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize centroids using k-means++ algorithm."""
        n_samples = X.shape[0]
        centroids = []
        
        # Choose first centroid randomly
        idx = torch.randint(0, n_samples, (1,)).item()
        centroids.append(X[idx])
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute squared distance to nearest centroid
            cent_tensor = torch.stack(centroids)
            distances = torch.cdist(X, cent_tensor)
            min_distances = distances.min(dim=1).values ** 2
            
            # Choose next centroid with probability proportional to distance
            probs = min_distances / min_distances.sum()
            idx = torch.multinomial(probs, 1).item()
            centroids.append(X[idx])
        
        return torch.stack(centroids)
    
    def _compute_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute distances from each point to each centroid."""
        return torch.cdist(X, self.centroids)
    
    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        """Assign each point to nearest centroid."""
        distances = self._compute_distances(X)
        return distances.argmin(dim=1)
    
    def _update_centroids(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Update centroids as mean of assigned points."""
        new_centroids = torch.zeros_like(self.centroids)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # Handle empty cluster by reinitializing to random point
                idx = torch.randint(0, X.shape[0], (1,)).item()
                new_centroids[k] = X[idx]
        
        return new_centroids
    
    def _compute_inertia(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute total within-cluster sum of squares."""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                inertia += ((X[mask] - self.centroids[k]) ** 2).sum().item()
        return inertia
    
    def fit(self, X: torch.Tensor) -> 'KMeans':
        """Fit K-Means to data."""
        # Initialize centroids
        if self.init == 'kmeans++':
            self.centroids = self._init_centroids_kmeanspp(X)
        else:
            self.centroids = self._init_centroids_random(X)
        
        # Iterate until convergence
        for i in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = (new_centroids - self.centroids).norm()
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        # Compute final inertia
        self._inertia = self._compute_inertia(X, labels)
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Assign points to nearest cluster."""
        return self._assign_clusters(X)
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and return cluster assignments."""
        self.fit(X)
        return self.predict(X)
    
    @property
    def inertia_(self) -> float:
        return self._inertia


def elbow_method(X: torch.Tensor, max_k: int = 10) -> list[float]:
    """Compute inertia for different k values to find optimal k."""
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias


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
    
    # Elbow method
    print("\nElbow method (inertias for k=1..5):")
    inertias = elbow_method(X, max_k=5)
    for k, inertia in enumerate(inertias, 1):
        print(f"  k={k}: {inertia:.2f}")
