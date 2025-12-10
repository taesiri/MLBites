"""
Decision Tree from Scratch - Starting Point

Implement a Decision Tree classifier from scratch.
Fill in the TODO sections to complete the implementation.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    """Decision tree node."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[int] = None  # Class prediction for leaf nodes
    
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    """Decision Tree classifier."""
    
    def __init__(
        self, 
        max_depth: int = None, 
        min_samples_split: int = 2,
        criterion: str = 'gini'
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes = None
    
    def _gini(self, y: torch.Tensor) -> float:
        """Compute Gini impurity."""
        # TODO: Gini = 1 - Σ p_i²
        pass
    
    def _entropy(self, y: torch.Tensor) -> float:
        """Compute entropy."""
        # TODO: Entropy = -Σ p_i * log₂(p_i)
        pass
    
    def _impurity(self, y: torch.Tensor) -> float:
        """Compute impurity based on criterion."""
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _information_gain(self, y: torch.Tensor, y_left: torch.Tensor, y_right: torch.Tensor) -> float:
        """Compute information gain from a split."""
        # TODO: IG = H(parent) - weighted average of H(children)
        pass
    
    def _best_split(self, X: torch.Tensor, y: torch.Tensor) -> tuple:
        """Find the best feature and threshold for splitting."""
        # TODO: Try all features and thresholds, return (feature_idx, threshold)
        # For each feature:
        #   For each unique value as threshold:
        #     Compute information gain
        #     Track best split
        pass
    
    def _build_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int = 0) -> Node:
        """Recursively build the decision tree."""
        n_samples = X.shape[0]
        n_classes = len(torch.unique(y))
        
        # TODO: Check stopping conditions
        # - max_depth reached
        # - min_samples_split not met
        # - node is pure (only one class)
        
        # TODO: Find best split
        
        # TODO: Split data and recursively build left and right subtrees
        
        # TODO: Return internal node or leaf node
        pass
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the decision tree."""
        self.n_classes = len(torch.unique(y))
        self.root = self._build_tree(X, y)
    
    def _predict_single(self, x: torch.Tensor, node: Node) -> int:
        """Predict class for a single sample."""
        # TODO: Traverse tree until leaf node
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict classes for all samples."""
        return torch.tensor([self._predict_single(x, self.root) for x in X])


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Simple dataset
    X = torch.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    # Train
    tree = DecisionTree(max_depth=5)
    tree.fit(X, y)
    
    # Predict
    predictions = tree.predict(X)
    accuracy = (predictions == y).float().mean()
    
    print(f"Training accuracy: {accuracy:.2%}")
