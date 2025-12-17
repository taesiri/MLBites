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
            feature_dim: Dimensionality of input features (e.g., 512 for CLIP ViT-B/32).
            num_classes: Number of target classes.
            lr: Learning rate for SGD optimization.

        Notes:
            - Initialize a single nn.Linear layer for classification.
            - Store hyperparameters for use in fit().
        """
        # TODO: Create nn.Linear(feature_dim, num_classes)
        # TODO: Store learning rate
        raise NotImplementedError

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

        Notes:
            - Use cross-entropy loss (nn.CrossEntropyLoss).
            - Use vanilla SGD: w = w - lr * grad (no momentum).
            - Process full batch each epoch.
        """
        # TODO: Set up cross-entropy loss
        # TODO: Training loop for `epochs` iterations:
        #   1. Forward pass: logits = linear(features)
        #   2. Compute loss
        #   3. Backward pass
        #   4. Manual SGD update: param -= lr * param.grad
        #   5. Zero gradients
        #   6. Record loss
        raise NotImplementedError

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict class labels for the given features.

        Args:
            features: Tensor of shape (M, feature_dim).

        Returns:
            Tensor of shape (M,) with predicted class labels (integers).
        """
        # TODO: Forward pass through linear layer
        # TODO: Return argmax of logits
        raise NotImplementedError

