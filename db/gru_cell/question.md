# GRU Cell from Scratch

## Problem Statement

Implement a **Gated Recurrent Unit (GRU)** cell from scratch. GRUs are a simplified version of LSTMs that combine the forget and input gates into a single "update" gate.

Your task is to:

1. Implement reset and update gates
2. Compute candidate hidden state
3. Interpolate between previous and candidate hidden states
4. Support batch processing

## Requirements

- Do **NOT** use `nn.GRUCell` or `nn.GRU`
- Implement both gates manually
- Use proper weight initialization
- Support batched inputs

## Function Signature

```python
class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        pass
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, input_size)
            h: Previous hidden state (batch, hidden_size)
        Returns:
            New hidden state
        """
        pass
```

## GRU Equations

```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)    # Reset gate
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)    # Update gate
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)  # Candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t        # New hidden state
```

## Example

```python
gru_cell = GRUCell(input_size=10, hidden_size=20)

x = torch.randn(32, 10)
h = torch.zeros(32, 20)

h_new = gru_cell(x, h)
```

## GRU vs LSTM

| Aspect | GRU | LSTM |
|--------|-----|------|
| Gates | 2 (reset, update) | 3 (forget, input, output) |
| State | Hidden only | Hidden + Cell |
| Parameters | Fewer | More |
| Performance | Similar | Similar |

## Hints

- GRU only has hidden state (no cell state like LSTM)
- Reset gate controls how much past info to forget
- Update gate controls interpolation between old and new
