from __future__ import annotations

import numpy as np


def linear_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2_reg: float = 0.0,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float]:
    """Fit linear regression (optionally ridge-regularized) parameters.

    This should solve for parameters that minimize squared error:
        min_{w,b} ||Xw + b - y||^2 + l2_reg * ||w||^2

    Notes:
    - If fit_intercept is False, return b = 0.0.
    - Do NOT regularize the intercept term.

    Args:
        X: Feature matrix of shape (N, D).
        y: Targets of shape (N,) or (N, 1).
        l2_reg: Non-negative L2 regularization strength.
        fit_intercept: Whether to fit an intercept term.

    Returns:
        w: Weights of shape (D,).
        b: Intercept (float). If fit_intercept=False, b == 0.0.
    """
    # TODO: implement
    raise NotImplementedError


def linear_regression_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict targets for a linear model y = Xw + b.

    Args:
        X: Feature matrix of shape (N, D).
        w: Weights of shape (D,).
        b: Intercept (float).

    Returns:
        y_pred: Predicted targets of shape (N,).
    """
    # TODO: implement
    raise NotImplementedError


