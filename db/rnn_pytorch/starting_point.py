from __future__ import annotations

import torch


class RNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        """A minimal vanilla RNN layer from scratch.

        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
        """
        # TODO: initialize learnable parameters
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
        """
        # TODO: process sequence and return outputs
        raise NotImplementedError




