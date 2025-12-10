"""
LSTM Cell from Scratch - Solution

Complete implementation of LSTM cell from scratch.
"""

import math
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """LSTM cell implemented from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrices for all gates (more efficient)
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        
        # Bias terms
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.W_ih, self.W_hh]:
            nn.init.uniform_(weight, -stdv, stdv)
        for bias in [self.b_ih, self.b_hh]:
            nn.init.zeros_(bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LSTM cell forward pass."""
        h, c = state
        
        # Compute all gates at once
        gates = x @ self.W_ih.T + h @ self.W_hh.T + self.b_ih + self.b_hh
        
        # Split into forget, input, cell, output gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate (candidate)
        o = torch.sigmoid(o)  # Output gate
        
        # Compute new cell state
        c_new = f * c + i * g
        
        # Compute new hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class LSTM(nn.Module):
    """Full LSTM layer that processes sequences."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create LSTM cells for each layer
        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, state: tuple = None):
        """Process full sequence."""
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)
        
        # Initialize state if None
        if state is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = state
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(x_t, (h[layer_idx], c[layer_idx]))
                x_t = h[layer_idx]
            
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=1)
        
        h_n = torch.stack(h)
        c_n = torch.stack(c)
        
        return output, (h_n, c_n)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test LSTM cell
    lstm_cell = LSTMCell(input_size=10, hidden_size=20)
    
    x = torch.randn(32, 10)
    h = torch.zeros(32, 20)
    c = torch.zeros(32, 20)
    
    h_new, c_new = lstm_cell(x, (h, c))
    
    print(f"Input shape: {x.shape}")
    print(f"Output hidden shape: {h_new.shape}")
    print(f"Output cell shape: {c_new.shape}")
    
    # Test full LSTM
    print("\n--- Full LSTM ---")
    lstm = LSTM(input_size=10, hidden_size=20, num_layers=2)
    x_seq = torch.randn(32, 15, 10)  # (batch, seq, features)
    
    output, (h_n, c_n) = lstm(x_seq)
    
    print(f"Sequence input: {x_seq.shape}")
    print(f"Output: {output.shape}")
    print(f"Final hidden: {h_n.shape}")
