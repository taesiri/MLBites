# Basic RNN from Scratch

## Problem

A Recurrent Neural Network (RNN) processes sequential data by maintaining a hidden state that gets updated at each timestep. Your task is to implement a basic RNN layer from scratch in PyTorch, matching the behavior of `torch.nn.RNN`.

## Task

Implement a minimal `RNN` class that:
- Takes input of shape `(seq_len, batch, input_size)` and optional initial hidden state
- Maintains learnable parameters
- Processes the sequence one timestep at a time
- Returns the output sequence and the final hidden state

## Function Signature

```python
class RNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None: ...

    def __call__(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

## Inputs and Outputs

- **inputs**:
  - `input_size`: dimension of input features at each timestep
  - `hidden_size`: dimension of hidden state
  - `x`: input tensor of shape `(seq_len, batch, input_size)`
  - `h0`: optional initial hidden state of shape `(1, batch, hidden_size)`. If `None`, initialize to zeros.

- **outputs**:
  - `output`: tensor of shape `(seq_len, batch, hidden_size)` containing hidden states at each timestep
  - `h_n`: final hidden state of shape `(1, batch, hidden_size)`

## Constraints

- Must be solvable in 20–30 minutes.
- Interview-friendly: no need to subclass `torch.nn.Module`.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (single timestep)

```python
torch.manual_seed(42)
rnn = RNN(input_size=3, hidden_size=4)
x = torch.randn(1, 2, 3)  # seq_len=1, batch=2, input_size=3
output, h_n = rnn(x)
# output.shape == (1, 2, 4)
# h_n.shape == (1, 2, 4)
# output and h_n should be equal for seq_len=1
```

### Example 2 (multiple timesteps)

```python
torch.manual_seed(42)
rnn = RNN(input_size=3, hidden_size=4)
x = torch.randn(5, 2, 3)  # seq_len=5, batch=2, input_size=3
output, h_n = rnn(x)
# output.shape == (5, 2, 4)
# h_n.shape == (1, 2, 4)
# h_n should equal output[-1:, :, :]
```

### Example 3 (with initial hidden state)

```python
torch.manual_seed(42)
rnn = RNN(input_size=3, hidden_size=4)
x = torch.randn(3, 2, 3)
h0 = torch.randn(1, 2, 4)
output, h_n = rnn(x, h0)
# output.shape == (3, 2, 4)
# h_n.shape == (1, 2, 4)
```




