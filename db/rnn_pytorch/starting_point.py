from __future__ import annotations

import torch


class RNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        """A minimal vanilla RNN layer (Elman RNN) from scratch.

        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.

        Notes:
            - Initialize weights using uniform distribution in [-k, k] where k = 1/sqrt(hidden_size).
            - Create four learnable parameters:
                - W_ih: weight matrix for input-to-hidden, shape (hidden_size, input_size)
                - W_hh: weight matrix for hidden-to-hidden, shape (hidden_size, hidden_size)
                - b_ih: bias for input-to-hidden, shape (hidden_size,)
                - b_hh: bias for hidden-to-hidden, shape (hidden_size,)
        """
        # TODO: store input_size and hidden_size
        # TODO: compute k = 1 / sqrt(hidden_size) for weight initialization
        # TODO: create W_ih, W_hh, b_ih, b_hh as nn.Parameter or requires_grad tensors
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input sequence through the RNN.

        Args:
            x: Input tensor of shape (seq_len, batch, input_size).
            h0: Optional initial hidden state of shape (1, batch, hidden_size).
                If None, defaults to zeros.

        Returns:
            output: Tensor of shape (seq_len, batch, hidden_size) with hidden states
                at each timestep.
            h_n: Final hidden state of shape (1, batch, hidden_size).

        The recurrence relation is:
            h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)
        """
        # TODO: get seq_len, batch from x.shape
        # TODO: if h0 is None, create zeros of shape (1, batch, hidden_size)
        # TODO: squeeze h0 to get h of shape (batch, hidden_size)
        # TODO: iterate through each timestep, compute new h, collect outputs
        # TODO: stack outputs and return (output, h_n with leading dim of 1)
        raise NotImplementedError

