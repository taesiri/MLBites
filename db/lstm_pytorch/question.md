# Implement LSTM from Scratch

## Problem
Long Short-Term Memory (LSTM) is a recurrent neural network architecture designed to learn long-term dependencies. Unlike vanilla RNNs, LSTMs use gating mechanisms (forget, input, output gates) to control information flow, addressing the vanishing gradient problem.

## Task
Implement a minimal, interview-friendly PyTorch module that performs a single-layer LSTM over a batch of sequences. The module should process sequences step-by-step, updating hidden and cell states at each timestep.

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
  - `x`: `torch.Tensor` of shape `(B, T, input_size)` where:
    - `B` = batch size
    - `T` = sequence length
    - `input_size` = input feature dimension
  - `hx` (optional): tuple of `(h_0, c_0)` where:
    - `h_0`: initial hidden state of shape `(B, hidden_size)`
    - `c_0`: initial cell state of shape `(B, hidden_size)`
    - If `None`, both are initialized to zeros.
- **outputs**:
  - `output`: `torch.Tensor` of shape `(B, T, hidden_size)` — hidden states at each timestep
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

### Example 1 (basic forward pass)

```python
import torch

torch.manual_seed(42)
lstm = SimpleLSTM(input_size=4, hidden_size=3, bias=True)
x = torch.randn(2, 5, 4)  # (B=2, T=5, input_size=4)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # torch.Size([2, 5, 3])
print(h_n.shape)     # torch.Size([2, 3])
print(c_n.shape)     # torch.Size([2, 3])
```

### Example 2 (with initial hidden state)

```python
import torch

torch.manual_seed(42)
lstm = SimpleLSTM(input_size=4, hidden_size=3, bias=False)
x = torch.randn(2, 5, 4)
h_0 = torch.zeros(2, 3)
c_0 = torch.zeros(2, 3)

output, (h_n, c_n) = lstm(x, hx=(h_0, c_0))
# output[:, -1, :] should equal h_n
print(torch.allclose(output[:, -1, :], h_n))  # True
```

