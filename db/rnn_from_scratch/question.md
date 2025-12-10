# RNN Cell from Scratch

## Problem Statement

Implement a **Vanilla RNN Cell** and **LSTM Cell** from scratch using only basic tensor operations. Understanding the internals of recurrent cells is crucial for grasping how sequence models learn temporal dependencies.

Your task is to:

1. Implement a vanilla RNN cell with the update equation: `h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)`
2. Implement an LSTM cell with forget, input, cell, and output gates
3. Process a full sequence through your cells

## Requirements

- Do **NOT** use `nn.RNNCell`, `nn.LSTMCell`, `nn.RNN`, or `nn.LSTM`
- Implement the gate equations for LSTM manually
- Support batch processing
- Initialize weights properly

## Function Signature

```python
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """Initialize vanilla RNN cell."""
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor
    ) -> torch.Tensor:
        """Single RNN cell step.
        
        Args:
            x: Input at current timestep (batch, input_size)
            h: Hidden state from previous timestep (batch, hidden_size)
            
        Returns:
            New hidden state (batch, hidden_size)
        """
        pass


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """Initialize LSTM cell."""
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single LSTM cell step.
        
        Args:
            x: Input at current timestep (batch, input_size)
            state: Tuple of (h, c) from previous timestep
            
        Returns:
            Tuple of new (h, c)
        """
        pass
```

## Example

```python
import torch

# Vanilla RNN
rnn_cell = VanillaRNNCell(input_size=10, hidden_size=20)
x = torch.randn(32, 10)  # batch=32, input_size=10
h = torch.zeros(32, 20)  # initial hidden state

h_new = rnn_cell(x, h)
print(f"Hidden shape: {h_new.shape}")  # (32, 20)

# LSTM
lstm_cell = LSTMCell(input_size=10, hidden_size=20)
h = torch.zeros(32, 20)
c = torch.zeros(32, 20)

h_new, c_new = lstm_cell(x, (h, c))
print(f"LSTM hidden shape: {h_new.shape}")  # (32, 20)
```

## Hints

- **Vanilla RNN**: `h_t = tanh(x @ W_ih.T + h @ W_hh.T + b_ih + b_hh)`
- **LSTM gates**:
  - Forget gate: `f = sigmoid(x @ W_if.T + h @ W_hf.T + b_f)`
  - Input gate: `i = sigmoid(x @ W_ii.T + h @ W_hi.T + b_i)`
  - Cell gate: `g = tanh(x @ W_ig.T + h @ W_hg.T + b_g)`
  - Output gate: `o = sigmoid(x @ W_io.T + h @ W_ho.T + b_o)`
  - Cell update: `c = f * c_prev + i * g`
  - Hidden: `h = o * tanh(c)`
