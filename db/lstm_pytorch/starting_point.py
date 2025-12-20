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
        # TODO: store input_size and hidden_size
        # TODO: create linear layer for input-to-hidden: nn.Linear(input_size, 4 * hidden_size, bias=bias)
        #       name it self.input_linear (projects input to 4 gates: i, f, g, o)
        # TODO: create linear layer for hidden-to-hidden: nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        #       name it self.hidden_linear
        raise NotImplementedError

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
        # TODO: get B, T, _ from x.shape
        # TODO: initialize h_t and c_t (from hx or zeros)
        # TODO: create list to collect hidden states
        # TODO: loop over timesteps t = 0 to T-1:
        #       - x_t = x[:, t, :]
        #       - gates = self.input_linear(x_t) + self.hidden_linear(h_t)
        #       - split gates into i, f, g, o (each of size hidden_size)
        #       - i = sigmoid(i), f = sigmoid(f), o = sigmoid(o), g = tanh(g)
        #       - c_t = f * c_t + i * g
        #       - h_t = o * tanh(c_t)
        #       - append h_t to output list
        # TODO: stack outputs -> (B, T, hidden_size)
        # TODO: return output, (h_t, c_t)
        raise NotImplementedError




