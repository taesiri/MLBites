from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept: bool = True) -> None:
        """A minimal ordinary least squares (OLS) linear regression model.

        Args:
            fit_intercept: If True, learn an intercept term.

        Attributes (after fit):
            coef_: Shape (n_features,). The learned coefficients.
            intercept_: The learned intercept (float).
        """
        # TODO: store fit_intercept and initialize coef_/intercept_ placeholders
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit OLS linear regression using least squares.

        Args:
            X: Shape (n_samples, n_features).
            y: Shape (n_samples,).

        Returns:
            self
        """
        # TODO: if fit_intercept, build a design matrix with a column of ones
        # TODO: solve least squares for parameters (e.g., via np.linalg.lstsq)
        # TODO: set self.coef_ and self.intercept_
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for inputs X.

        Args:
            X: Shape (n_samples, n_features).

        Returns:
            Predicted y values of shape (n_samples,).
        """
        # TODO: return X @ coef_ + intercept_
        raise NotImplementedError


