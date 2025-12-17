from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for 28x28 grayscale image classification.

    Architecture:
        - Conv1: 1 -> 32 channels, 3x3 kernel, padding=1, ReLU, 2x2 max pool -> 14x14x32
        - Conv2: 32 -> 64 channels, 3x3 kernel, padding=1, ReLU, 2x2 max pool -> 7x7x64
        - Flatten: 7*7*64 = 3136
        - FC1: 3136 -> 128, ReLU
        - FC2: 128 -> 10 (logits)

    Args:
        None

    Returns:
        forward(x) returns logits of shape (batch_size, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: define conv1 (1 -> 32 channels, 3x3 kernel, padding=1)
        # TODO: define conv2 (32 -> 64 channels, 3x3 kernel, padding=1)
        # TODO: define fc1 (3136 -> 128)
        # TODO: define fc2 (128 -> 10)
        # TODO: define pooling and activation layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SimpleCNN.

        Args:
            x: input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            logits tensor of shape (batch_size, 10)
        """
        # TODO: apply conv1 -> relu -> pool
        # TODO: apply conv2 -> relu -> pool
        # TODO: flatten
        # TODO: apply fc1 -> relu
        # TODO: apply fc2
        raise NotImplementedError

