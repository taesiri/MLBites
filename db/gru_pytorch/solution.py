from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        """A minimal single-layer GRU module (batch-first).

        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
            bias: If False, the layer does not use bias weights.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined projection for reset (r) and update (z) gates
        self.input_linear_rz = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.hidden_linear_rz = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)

        # Separate projection for new gate (n) - applied after reset gate
        self.input_linear_n = nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_linear_n = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GRU over the input sequence.

        Args:
            x: Input tensor of shape (B, T, input_size).
            h0: Optional initial hidden state of shape (B, hidden_size).
                If None, initialized to zeros.

        Returns:
            output: Tensor of shape (B, T, hidden_size) — hidden states at each timestep.
            h_n: Final hidden state of shape (B, hidden_size).
        """
        B, T, _ = x.shape

        # Initialize hidden state
        if h0 is None:
            h_t = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_t = h0

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]

            # Compute reset and update gates
            gates_rz = self.input_linear_rz(x_t) + self.hidden_linear_rz(h_t)
            r, z = gates_rz.chunk(2, dim=1)

            # Apply sigmoid to gates
            r = torch.sigmoid(r)  # reset gate
            z = torch.sigmoid(z)  # update gate

            # Compute new gate (candidate hidden state)
            # n = tanh(W_in @ x_t + b_in + r * (W_hn @ h_t + b_hn))
            n = torch.tanh(self.input_linear_n(x_t) + r * self.hidden_linear_n(h_t))

            # Update hidden state: h_t = (1 - z) * n + z * h_{t-1}
            h_t = (1 - z) * n + z * h_t

            outputs.append(h_t)

        # Stack all hidden states: list of (B, H) -> (B, T, H)
        output = torch.stack(outputs, dim=1)

        return output, h_t




