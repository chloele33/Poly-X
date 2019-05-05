import numpy as np
from prml.linear.regression import Regression


class LinearRegression(Regression):
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def fit(self, X, t):
        np.ndarray(X)
        np.ndarray(t)
        """
        perform least squares fitting

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        """
        self.w = np.matmul(np.linalg.pinv(X), t)
        self.var = np.mean(np.square(np.matmul(X , self.w) - t))

    def predict(self, X, return_std=False):
        bool(return_std)
        np.ndarray(X)
        """
        make prediction given input

        Parameters
        ----------
        X : (N, D) np.ndarray
            samples to predict their output
        return_std : bool, optional
            returns standard deviation of each predition if True

        Returns
        -------
        y : (N,) np.ndarray
            prediction of each sample
        y_std : (N,) np.ndarray
            standard deviation of each predition
        """
        y = np.matmul(X , self.w)
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
