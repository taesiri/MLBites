from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        """A minimal single-layer LSTM module (batch-first).

        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
            bias: If False, the layer does not use bias weights.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined projection for all 4 gates: (i, f, g, o)
        self.input_linear = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_linear = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run LSTM over the input sequence.

        Args:
            x: Input tensor of shape (B, T, input_size).
            hx: Optional tuple of (h_0, c_0), each of shape (B, hidden_size).
                If None, both are initialized to zeros.

        Returns:
            output: Tensor of shape (B, T, hidden_size) — hidden states at each timestep.
            (h_n, c_n): Tuple of final hidden state and cell state.
        """
        B, T, _ = x.shape

        # Initialize hidden and cell states
        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)
            c_t = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_t, c_t = hx

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]

            # Compute all gates in one pass
            gates = self.input_linear(x_t) + self.hidden_linear(h_t)

            # Split into individual gates
            i, f, g, o = gates.chunk(4, dim=1)

            # Apply activations
            i = torch.sigmoid(i)  # input gate
            f = torch.sigmoid(f)  # forget gate
            g = torch.tanh(g)     # cell gate (candidate)
            o = torch.sigmoid(o)  # output gate

            # Update cell and hidden state
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)

            outputs.append(h_t)

        # Stack all hidden states: list of (B, H) -> (B, T, H)
        output = torch.stack(outputs, dim=1)

        return output, (h_t, c_t)


