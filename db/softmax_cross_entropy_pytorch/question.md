# Softmax, LogSoftmax, and CrossEntropy

## Problem
Softmax is a fundamental activation function that converts raw logits into a probability distribution. Log-softmax is the logarithm of softmax, and cross-entropy loss measures the difference between predicted probabilities and true labels. These functions are ubiquitous in classification tasks. A key challenge is implementing them in a **numerically stable** way to avoid overflow/underflow when dealing with large or small logit values.

## Task
Implement three standalone functions from scratch (without using `torch.softmax`, `torch.log_softmax`, `torch.nn.functional.softmax`, `torch.nn.functional.log_softmax`, or `torch.nn.functional.cross_entropy`):

1. **`softmax(x, dim)`**: Compute the softmax along a specified dimension. Must be numerically stable (handle large logits without overflow).

2. **`log_softmax(x, dim)`**: Compute the log-softmax along a specified dimension. Must be numerically stable using the log-sum-exp trick.

3. **`cross_entropy(logits, targets)`**: Compute the cross-entropy loss for classification. Takes raw logits and integer class indices. Should use your `log_softmax` internally. Return the mean loss over the batch.

## Function Signatures

```python
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor: ...

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor: ...

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs

### `softmax`
- **Inputs**:
  - `x`: `torch.Tensor` of any shape containing raw logits
  - `dim`: `int` specifying the dimension along which to compute softmax (default: -1)
- **Outputs**:
  - `torch.Tensor` of the same shape as `x`, with values in (0, 1) that sum to 1 along `dim`

### `log_softmax`
- **Inputs**:
  - `x`: `torch.Tensor` of any shape containing raw logits
  - `dim`: `int` specifying the dimension along which to compute log-softmax (default: -1)
- **Outputs**:
  - `torch.Tensor` of the same shape as `x`, containing log-probabilities (negative values)

### `cross_entropy`
- **Inputs**:
  - `logits`: `torch.Tensor` of shape `(N, C)` where N is batch size and C is number of classes
  - `targets`: `torch.Tensor` of shape `(N,)` containing integer class indices in range `[0, C-1]`
- **Outputs**:
  - Scalar `torch.Tensor` containing the mean cross-entropy loss

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal boilerplate.
- Assume inputs satisfy the documented contract (valid tensors, correct shapes).
- Allowed libs: PyTorch (`torch`) and Python standard library.
- **Do NOT use**: `torch.softmax`, `torch.log_softmax`, `torch.nn.functional.softmax`, `torch.nn.functional.log_softmax`, `torch.nn.functional.cross_entropy`, `torch.nn.CrossEntropyLoss`.

## Examples

### Example 1 (softmax)
```python
x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
result = softmax(x, dim=-1)
# Each row sums to 1.0
# result ≈ tensor([[0.0900, 0.2447, 0.6652],
#                  [0.0900, 0.2447, 0.6652]])
```

### Example 2 (log_softmax)
```python
x = torch.tensor([[1.0, 2.0, 3.0]])
result = log_softmax(x, dim=-1)
# result ≈ tensor([[-2.4076, -1.4076, -0.4076]])
# exp(result) should give softmax values
```

### Example 3 (cross_entropy)
```python
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
targets = torch.tensor([0, 1])  # first sample: class 0, second sample: class 1
loss = cross_entropy(logits, targets)
# loss ≈ tensor(0.3185)
```

### Example 4 (numerical stability)
```python
# Large logits that would overflow without stability trick
x = torch.tensor([[1000.0, 1001.0, 1002.0]])
result = softmax(x, dim=-1)
# Should NOT produce inf or nan
# result ≈ tensor([[0.0900, 0.2447, 0.6652]])
```




