"""
K-Nearest Neighbors (kNN) - Solution

Complete implementation of kNN from scratch.
"""

import torch


class KNN:
    """K-Nearest Neighbors classifier."""
    
    def __init__(self, k: int = 5, distance: str = 'euclidean'):
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Store training data."""
        self.X_train = X
        self.y_train = y
        self.n_classes = int(y.max()) + 1
    
    def _compute_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute distances between X and training data."""
        if self.distance == 'euclidean':
            # Efficient computation using broadcasting
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
            X_sq = (X ** 2).sum(dim=1, keepdim=True)
            train_sq = (self.X_train ** 2).sum(dim=1, keepdim=True).T
            cross = X @ self.X_train.T
            distances = torch.sqrt(X_sq + train_sq - 2 * cross + 1e-8)
            
        elif self.distance == 'manhattan':
            # |x - y| using broadcasting
            distances = torch.abs(X.unsqueeze(1) - self.X_train.unsqueeze(0)).sum(dim=2)
            
        elif self.distance == 'cosine':
            # 1 - cosine_similarity
            X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
            train_norm = self.X_train / (self.X_train.norm(dim=1, keepdim=True) + 1e-8)
            distances = 1 - X_norm @ train_norm.T
            
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        return distances
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict labels for query points."""
        # Compute distances to all training points
        distances = self._compute_distances(X)
        
        # Find k nearest neighbors (smallest distances)
        _, indices = torch.topk(distances, k=self.k, dim=1, largest=False)
        
        # Get labels of k nearest neighbors
        neighbor_labels = self.y_train[indices]  # (n_queries, k)
        
        # Majority vote for classification
        predictions = []
        for labels in neighbor_labels:
            counts = torch.bincount(labels, minlength=self.n_classes)
            predictions.append(counts.argmax())
        
        return torch.stack(predictions)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        distances = self._compute_distances(X)
        _, indices = torch.topk(distances, k=self.k, dim=1, largest=False)
        neighbor_labels = self.y_train[indices]
        
        probas = []
        for labels in neighbor_labels:
            counts = torch.bincount(labels, minlength=self.n_classes).float()
            probas.append(counts / self.k)
        
        return torch.stack(probas)


class KNNRegressor:
    """K-Nearest Neighbors for regression."""
    
    def __init__(self, k: int = 5, weights: str = 'uniform'):
        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        self.X_train = X
        self.y_train = y
    
    def _compute_distances(self, X: torch.Tensor) -> torch.Tensor:
        X_sq = (X ** 2).sum(dim=1, keepdim=True)
        train_sq = (self.X_train ** 2).sum(dim=1, keepdim=True).T
        cross = X @ self.X_train.T
        return torch.sqrt(X_sq + train_sq - 2 * cross + 1e-8)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict by averaging k nearest neighbor values."""
        distances = self._compute_distances(X)
        top_distances, indices = torch.topk(distances, k=self.k, dim=1, largest=False)
        neighbor_values = self.y_train[indices]  # (n_queries, k)
        
        if self.weights == 'uniform':
            predictions = neighbor_values.mean(dim=1)
        elif self.weights == 'distance':
            # Weight by inverse distance
            weights = 1 / (top_distances + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)
            predictions = (neighbor_values * weights).sum(dim=1)
        
        return predictions


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate sample data
    X_train = torch.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()
    
    X_test = torch.randn(10, 2)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).long()
    
    # Test classifier
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print(f"Test labels:  {y_test.tolist()}")
    print(f"Predictions:  {predictions.tolist()}")
    print(f"Accuracy: {(predictions == y_test).float().mean():.2%}")
    
    # Test probabilities
    probas = knn.predict_proba(X_test)
    print(f"\nClass probabilities:\n{probas}")
    
    # Test regression
    print("\n--- Regression ---")
    y_reg = X_train[:, 0] + 2 * X_train[:, 1] + torch.randn(100) * 0.1
    knn_reg = KNNRegressor(k=5, weights='distance')
    knn_reg.fit(X_train, y_reg)
    
    y_pred = knn_reg.predict(X_test[:3])
    y_true = X_test[:3, 0] + 2 * X_test[:3, 1]
    print(f"True: {y_true.tolist()}")
    print(f"Pred: {y_pred.tolist()}")
