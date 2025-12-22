# LSTM from Scratch

## Problem
Implement a single-layer LSTM module in PyTorch that processes sequences step-by-step.

## Function Signature

```python
class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: `torch.Tensor` of shape `(B, T, input_size)`
  - `hx` (optional): tuple of `(h_0, c_0)`, each of shape `(B, hidden_size)`. If `None`, both are initialized to zeros.
- **outputs**:
  - `output`: `torch.Tensor` of shape `(B, T, hidden_size)`
  - `(h_n, c_n)`: tuple of final hidden state and cell state, each of shape `(B, hidden_size)`

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- You may assume:
  - `input_size > 0` and `hidden_size > 0`
  - Input tensors have the correct shapes

## Examples

```python
import torch

torch.manual_seed(42)
lstm = SimpleLSTM(input_size=4, hidden_size=3, bias=True)
x = torch.randn(2, 5, 4)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # torch.Size([2, 5, 3])
print(h_n.shape)     # torch.Size([2, 3])
print(c_n.shape)     # torch.Size([2, 3])
```




