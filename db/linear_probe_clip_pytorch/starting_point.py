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
        """Initialize a linear probe classifier.

        Args:
            feature_dim: Dimensionality of input features.
            num_classes: Number of target classes.
            lr: Learning rate.
        """
        raise NotImplementedError

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
    ) -> list[float]:
        """Train the classifier on the given features and labels.

        Args:
            features: Tensor of shape (N, feature_dim).
            labels: Tensor of shape (N,), integer class labels in [0, num_classes).
            epochs: Number of training epochs.

        Returns:
            List of loss values (one per epoch).
        """
        raise NotImplementedError

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict class labels for the given features.

        Args:
            features: Tensor of shape (M, feature_dim).

        Returns:
            Tensor of shape (M,) with predicted class labels (integers).
        """
        raise NotImplementedError




