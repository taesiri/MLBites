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
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit OLS linear regression using least squares.

        Args:
            X: Shape (n_samples, n_features).
            y: Shape (n_samples,).

        Returns:
            self
        """
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=X.dtype)
            X_design = np.concatenate([X, ones], axis=1)
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            self.coef_ = beta[:-1]
            self.intercept_ = float(beta[-1])
        else:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.coef_ = beta
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for inputs X.

        Args:
            X: Shape (n_samples, n_features).

        Returns:
            Predicted y values of shape (n_samples,).
        """
        coef = self.coef_
        if coef is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
        return X @ coef + self.intercept_


