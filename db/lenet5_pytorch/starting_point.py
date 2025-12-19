from __future__ import annotations

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5 convolutional neural network for 32x32 grayscale image classification.

    Architecture:
        - Conv1: 1 -> 6 channels, 5x5 kernel, ReLU, 2x2 max pool -> 14x14x6
        - Conv2: 6 -> 16 channels, 5x5 kernel, ReLU, 2x2 max pool -> 5x5x16
        - Flatten: 5*5*16 = 400
        - FC1: 400 -> 120, ReLU
        - FC2: 120 -> 84, ReLU
        - FC3: 84 -> 10 (logits)

    Args:
        None

    Returns:
        forward(x) returns logits of shape (batch_size, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: define conv1 (1 -> 6 channels, 5x5 kernel)
        # TODO: define conv2 (6 -> 16 channels, 5x5 kernel)
        # TODO: define fc1 (400 -> 120)
        # TODO: define fc2 (120 -> 84)
        # TODO: define fc3 (84 -> 10)
        # TODO: define pooling and activation layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LeNet-5.

        Args:
            x: input tensor of shape (batch_size, 1, 32, 32)

        Returns:
            logits tensor of shape (batch_size, 10)
        """
        # TODO: apply conv1 -> relu -> pool
        # TODO: apply conv2 -> relu -> pool
        # TODO: flatten
        # TODO: apply fc1 -> relu
        # TODO: apply fc2 -> relu
        # TODO: apply fc3
        raise NotImplementedError


