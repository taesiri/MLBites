"""
GRU Cell from Scratch - Solution

Complete implementation of GRU cell from scratch.
"""

import math
import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """GRU cell implemented from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrices for efficiency
        # For reset and update gates
        self.W_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        
        # For candidate
        self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))
        
        # Biases
        self.b_ir = nn.Parameter(torch.zeros(hidden_size))
        self.b_hr = nn.Parameter(torch.zeros(hidden_size))
        self.b_iz = nn.Parameter(torch.zeros(hidden_size))
        self.b_hz = nn.Parameter(torch.zeros(hidden_size))
        self.b_in = nn.Parameter(torch.zeros(hidden_size))
        self.b_hn = nn.Parameter(torch.zeros(hidden_size))
        
        self._init_weights()
    
    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """GRU cell forward pass."""
        # Reset gate
        r = torch.sigmoid(x @ self.W_ir.T + self.b_ir + h @ self.W_hr.T + self.b_hr)
        
        # Update gate
        z = torch.sigmoid(x @ self.W_iz.T + self.b_iz + h @ self.W_hz.T + self.b_hz)
        
        # Candidate hidden state (reset gate applied to h)
        n = torch.tanh(x @ self.W_in.T + self.b_in + r * (h @ self.W_hn.T + self.b_hn))
        
        # Interpolate: new hidden state
        h_new = (1 - z) * h + z * n
        
        return h_new


class GRUCellEfficient(nn.Module):
    """GRU cell with combined weight matrices (more efficient)."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combined weights: 3 gates (r, z, n) combined
        self.W_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        
        self._init_weights()
    
    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in [self.W_ih, self.W_hh]:
            nn.init.uniform_(p, -stdv, stdv)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = x @ self.W_ih.T + self.b_ih
        gh = h @ self.W_hh.T + self.b_hh
        
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        
        return (1 - z) * h + z * n


class GRU(nn.Module):
    """Full GRU layer that processes sequences."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.cells = nn.ModuleList([
            GRUCellEfficient(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)
        
        if h is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h = [h[i] for i in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(x_t, h[layer_idx])
                x_t = h[layer_idx]
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h)
        
        return output, h_n


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test GRU cell
    gru_cell = GRUCell(input_size=10, hidden_size=20)
    
    x = torch.randn(32, 10)
    h = torch.zeros(32, 20)
    
    h_new = gru_cell(x, h)
    
    print(f"Input shape: {x.shape}")
    print(f"Output hidden shape: {h_new.shape}")
    
    # Test full GRU
    print("\n--- Full GRU ---")
    gru = GRU(input_size=10, hidden_size=20, num_layers=2)
    x_seq = torch.randn(32, 15, 10)
    
    output, h_n = gru(x_seq)
    
    print(f"Sequence input: {x_seq.shape}")
    print(f"Output: {output.shape}")
    print(f"Final hidden: {h_n.shape}")
