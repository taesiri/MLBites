from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe:
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        lr: float = 0.1,
    ) -> None:
        """A linear probe classifier for pre-extracted features.

        Args:
            feature_dim: Dimensionality of input features.
            num_classes: Number of target classes.
            lr: Learning rate for SGD optimization.
        """
        self.linear = nn.Linear(feature_dim, num_classes)
        self.lr = lr

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
    ) -> list[float]:
        """Train the linear probe on the given features and labels.

        Args:
            features: Tensor of shape (N, feature_dim), pre-extracted features.
            labels: Tensor of shape (N,), integer class labels in [0, num_classes).
            epochs: Number of training epochs.

        Returns:
            List of loss values (one per epoch).
        """
        criterion = nn.CrossEntropyLoss()
        losses: list[float] = []

        for _ in range(epochs):
            # Forward pass
            logits = self.linear(features)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Manual SGD update (no momentum)
            with torch.no_grad():
                for param in self.linear.parameters():
                    param -= self.lr * param.grad

            # Zero gradients
            self.linear.zero_grad()

            losses.append(loss.item())

        return losses

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict class labels for the given features.

        Args:
            features: Tensor of shape (M, feature_dim).

        Returns:
            Tensor of shape (M,) with predicted class labels (integers).
        """
        logits = self.linear(features)
        return logits.argmax(dim=1)

