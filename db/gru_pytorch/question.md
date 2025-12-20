# GRU from Scratch

## Problem
Gated Recurrent Unit (GRU) is a recurrent neural network architecture that, like LSTM, uses gating mechanisms to control information flow. GRU is simpler than LSTM—it has only two gates (reset and update) and no separate cell state. Despite its simplicity, GRU often achieves comparable performance to LSTM on many tasks.

## Task
Implement a minimal, interview-friendly PyTorch module that performs a single-layer GRU over a batch of sequences. The module should process sequences step-by-step, updating the hidden state at each timestep using reset and update gates.

## Function Signature

```python
class SimpleGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: `torch.Tensor` of shape `(B, T, input_size)` where:
    - `B` = batch size
    - `T` = sequence length
    - `input_size` = input feature dimension
  - `h0` (optional): initial hidden state of shape `(B, hidden_size)`
    - If `None`, initialized to zeros.
- **outputs**:
  - `output`: `torch.Tensor` of shape `(B, T, hidden_size)` — hidden states at each timestep
  - `h_n`: final hidden state of shape `(B, hidden_size)`

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
gru = SimpleGRU(input_size=4, hidden_size=3, bias=True)
x = torch.randn(2, 5, 4)  # (B=2, T=5, input_size=4)
output, h_n = gru(x)

print(output.shape)  # torch.Size([2, 5, 3])
print(h_n.shape)     # torch.Size([2, 3])
```

### Example 2 (with initial hidden state)

```python
import torch

torch.manual_seed(42)
gru = SimpleGRU(input_size=4, hidden_size=3, bias=False)
x = torch.randn(2, 5, 4)
h_0 = torch.zeros(2, 3)

output, h_n = gru(x, h0=h_0)
# output[:, -1, :] should equal h_n
print(torch.allclose(output[:, -1, :], h_n))  # True
```




