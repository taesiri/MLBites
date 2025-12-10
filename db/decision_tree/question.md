# Decision Tree from Scratch

## Problem Statement

Implement a **Decision Tree** classifier from scratch. Decision trees recursively split data based on feature thresholds to maximize class purity.

Your task is to:

1. Compute impurity measures (Gini, Entropy)
2. Find the best split at each node
3. Recursively build the tree
4. Implement prediction by traversing the tree

## Requirements

- Do **NOT** use sklearn's DecisionTreeClassifier
- Implement both Gini impurity and Information Gain
- Support max_depth and min_samples_split parameters
- Handle both binary and multi-class classification

## Function Signature

```python
class DecisionTree:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, criterion: str = 'gini'):
        pass
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass
```

## Impurity Measures

**Gini Impurity:**
```
Gini = 1 - Σ p_i²
```

**Entropy / Information Gain:**
```
Entropy = -Σ p_i * log₂(p_i)
IG = H(parent) - Σ (n_child/n_parent) * H(child)
```

## Example

```python
X = torch.randn(100, 4)
y = torch.randint(0, 2, (100,))

tree = DecisionTree(max_depth=5)
tree.fit(X, y)

predictions = tree.predict(X)
```

## Hints

- Each node stores: feature index, threshold, left/right children, or class prediction
- Find best split by trying all features and thresholds
- Use recursion for building and predicting
- Stop splitting when max_depth reached or node is pure
