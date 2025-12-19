# Softmax + Cross-Entropy Loss (Numerically Stable)

## Problem
Softmax converts raw logits into a probability distribution, and cross-entropy measures the difference between predicted probabilities and true labels. Naively computing `exp(logits)` can overflow for large values, and `log(softmax)` can underflow for small probabilities. A numerically stable implementation avoids these issues.

## Task
Implement two functions in NumPy:
1. `softmax(logits)` — numerically stable softmax
2. `cross_entropy_loss(logits, targets)` — numerically stable cross-entropy loss (combining softmax and log-loss)

The key insight is to subtract the maximum logit before exponentiating, and to compute log-softmax directly using the log-sum-exp trick.

## Function Signature

```python
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray: ...

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float: ...
```

## Inputs and Outputs

### `softmax`
- **Input**: `logits` — shape `(n_samples, n_classes)`, float
- **Output**: probabilities — shape `(n_samples, n_classes)`, each row sums to 1

### `cross_entropy_loss`
- **Input**: 
  - `logits` — shape `(n_samples, n_classes)`, float
  - `targets` — shape `(n_samples,)`, integer class indices in `[0, n_classes)`
- **Output**: scalar float, the mean cross-entropy loss over all samples

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: no frameworks, no extra features.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: NumPy (`numpy`) and Python standard library.
- Must be numerically stable (no overflow/underflow for reasonable inputs).

## Examples

### Example 1 (basic softmax)
```python
import numpy as np

logits = np.array([[1.0, 2.0, 3.0]])
probs = softmax(logits)
# expected: approximately [[0.09003, 0.24473, 0.66524]]
# rows sum to 1.0
```

### Example 2 (stability with large logits)
```python
import numpy as np

logits = np.array([[1000.0, 1001.0, 1002.0]])
probs = softmax(logits)
# expected: approximately [[0.09003, 0.24473, 0.66524]]
# naive exp(1000) would overflow, but stable version handles it
```

### Example 3 (cross-entropy loss)
```python
import numpy as np

logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = np.array([0, 1])  # correct classes
loss = cross_entropy_loss(logits, targets)
# expected: approximately 0.4076
```


