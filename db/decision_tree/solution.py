"""
Decision Tree from Scratch - Solution

Complete implementation of Decision Tree classifier.
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
    value: Optional[int] = None
    
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
        if len(y) == 0:
            return 0.0
        counts = torch.bincount(y)
        probs = counts.float() / len(y)
        return (1 - (probs ** 2).sum()).item()
    
    def _entropy(self, y: torch.Tensor) -> float:
        """Compute entropy."""
        if len(y) == 0:
            return 0.0
        counts = torch.bincount(y)
        probs = counts.float() / len(y)
        probs = probs[probs > 0]
        return (-probs * torch.log2(probs)).sum().item()
    
    def _impurity(self, y: torch.Tensor) -> float:
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _information_gain(self, y: torch.Tensor, y_left: torch.Tensor, y_right: torch.Tensor) -> float:
        """Compute information gain from a split."""
        p = len(y)
        if len(y_left) == 0 or len(y_right) == 0:
            return 0.0
        
        parent_impurity = self._impurity(y)
        left_impurity = self._impurity(y_left)
        right_impurity = self._impurity(y_right)
        
        weighted_child = (len(y_left) / p) * left_impurity + (len(y_right) / p) * right_impurity
        
        return parent_impurity - weighted_child
    
    def _best_split(self, X: torch.Tensor, y: torch.Tensor) -> tuple:
        """Find the best feature and threshold for splitting."""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = torch.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold.item()
        
        return best_feature, best_threshold
    
    def _most_common_class(self, y: torch.Tensor) -> int:
        """Return the most common class."""
        counts = torch.bincount(y)
        return counts.argmax().item()
    
    def _build_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int = 0) -> Node:
        """Recursively build the decision tree."""
        n_samples = X.shape[0]
        n_classes = len(torch.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            return Node(value=self._most_common_class(y))
        
        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        
        if feature_idx is None:
            return Node(value=self._most_common_class(y))
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the decision tree."""
        self.n_classes = len(torch.unique(y))
        self.root = self._build_tree(X, y)
    
    def _predict_single(self, x: torch.Tensor, node: Node) -> int:
        """Predict class for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict classes for all samples."""
        return torch.tensor([self._predict_single(x, self.root) for x in X])


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Simple dataset
    X = torch.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    # Train
    tree = DecisionTree(max_depth=5, criterion='gini')
    tree.fit(X, y)
    
    # Predict
    predictions = tree.predict(X)
    accuracy = (predictions == y).float().mean()
    
    print(f"Training accuracy: {accuracy:.2%}")
    
    # Test on new data
    X_test = torch.randn(20, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).long()
    test_pred = tree.predict(X_test)
    test_acc = (test_pred == y_test).float().mean()
    
    print(f"Test accuracy: {test_acc:.2%}")
