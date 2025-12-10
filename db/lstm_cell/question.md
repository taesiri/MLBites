# LSTM Cell from Scratch

## Problem Statement

Implement a **Long Short-Term Memory (LSTM)** cell from scratch. LSTMs solve the vanishing gradient problem in vanilla RNNs using gating mechanisms.

Your task is to:

1. Implement all four gates: forget, input, cell, and output
2. Maintain both hidden state (h) and cell state (c)
3. Process sequences step by step
4. Support batch processing

## Requirements

- Do **NOT** use `nn.LSTMCell` or `nn.LSTM`
- Implement all gate computations manually
- Use proper weight initialization
- Support batched inputs

## Function Signature

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        pass
    
    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        """
        Args:
            x: Input (batch, input_size)
            state: Tuple of (h, c) each (batch, hidden_size)
        Returns:
            Tuple of new (h, c)
        """
        pass
```

## LSTM Equations

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) # Cell gate (candidate)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       # New cell state
h_t = o_t ⊙ tanh(c_t)                  # New hidden state
```

## Example

```python
lstm_cell = LSTMCell(input_size=10, hidden_size=20)

x = torch.randn(32, 10)  # batch=32
h = torch.zeros(32, 20)
c = torch.zeros(32, 20)

h_new, c_new = lstm_cell(x, (h, c))
```

## Hints

- Concatenate h and x for efficiency: [h, x] @ W
- Can compute all gates at once and split with chunk()
- Cell state c flows with minimal modification (solves vanishing gradient)
- Use sigmoid for gates, tanh for candidate activation
