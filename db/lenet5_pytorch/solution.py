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
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LeNet-5.

        Args:
            x: input tensor of shape (batch_size, 1, 32, 32)

        Returns:
            logits tensor of shape (batch_size, 10)
        """
        # Conv block 1: 32x32 -> 28x28 -> 14x14
        x = self.pool(self.relu(self.conv1(x)))

        # Conv block 2: 14x14 -> 10x10 -> 5x5
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten: (batch, 16, 5, 5) -> (batch, 400)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x




