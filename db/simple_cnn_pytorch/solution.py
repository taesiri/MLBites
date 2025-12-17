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
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SimpleCNN.

        Args:
            x: input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            logits tensor of shape (batch_size, 10)
        """
        # Conv block 1: 28x28 -> 28x28 (padding=1) -> 14x14 (pool)
        x = self.pool(self.relu(self.conv1(x)))

        # Conv block 2: 14x14 -> 14x14 (padding=1) -> 7x7 (pool)
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten: (batch, 64, 7, 7) -> (batch, 3136)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

