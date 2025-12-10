"""
K-Nearest Neighbors (kNN) - Starting Point

Implement kNN from scratch.
Fill in the TODO sections to complete the implementation.
"""

import torch


class KNN:
    """K-Nearest Neighbors classifier."""
    
    def __init__(self, k: int = 5, distance: str = 'euclidean'):
        """
        Initialize kNN.
        
        Args:
            k: Number of neighbors
            distance: Distance metric ('euclidean', 'manhattan', 'cosine')
        """
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Store training data (kNN is a lazy learner).
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        # TODO: Store training data
        pass
    
    def _compute_distances(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between X and training data.
        
        Args:
            X: Query points (n_queries, n_features)
            
        Returns:
            Distance matrix (n_queries, n_train)
        """
        # TODO: Implement distance computation based on self.distance
        # Euclidean: sqrt(sum((x - y)^2))
        # Manhattan: sum(|x - y|)
        # Cosine: 1 - (x·y)/(||x|| ||y||)
        
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for query points.
        
        Args:
            X: Query points (n_queries, n_features)
            
        Returns:
            Predicted labels (n_queries,)
        """
        # TODO: Compute distances to all training points
        
        # TODO: Find k nearest neighbors
        # Hint: Use torch.topk with largest=False
        
        # TODO: Get labels of k nearest neighbors
        
        # TODO: Majority vote for classification
        # Hint: Use torch.mode or torch.bincount
        
        pass
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Returns:
            Probability matrix (n_queries, n_classes)
        """
        # TODO: Return proportion of each class among k neighbors
        pass


class KNNRegressor:
    """K-Nearest Neighbors for regression."""
    
    def __init__(self, k: int = 5, weights: str = 'uniform'):
        """
        Args:
            k: Number of neighbors
            weights: 'uniform' or 'distance'
        """
        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict by averaging k nearest neighbor values."""
        # TODO: Implement regression prediction
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate sample data
    X_train = torch.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()  # Simple linear boundary
    
    X_test = torch.randn(10, 2)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).long()
    
    # Create and fit kNN
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    
    # Predict
    predictions = knn.predict(X_test)
    
    print(f"Test labels:  {y_test.tolist()}")
    print(f"Predictions:  {predictions.tolist()}")
    print(f"Accuracy: {(predictions == y_test).float().mean():.2%}")
