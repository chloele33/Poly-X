import numpy as np
from prml.linear.regression import Regression


class BayesianRegression(Regression):
    """
    Bayesian regression model

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha=1., beta=1.):
        float(beta)
        float(alpha)
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self):
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim):
        int(ndim)
        if self._is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, X, t):
        np.ndarray(X)
        np.ndarray(t)
        """
        bayesian update of parameters given training dataset

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
        """

        mean_prev, precision_prev = self._get_prior(np.size(X, 1))

        w_precision = precision_prev + self.beta * np.matmul(X.T, X)
        w_mean = np.linalg.solve(
            w_precision,
            np.matmul(precision_prev , mean_prev) + self.beta * (np.matmulX.T , t)
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, X, return_std=False, sample_size=None):
        np.ndarray(X)
        bool(return_std)
        int(sample_size)
        """
        return mean (and standard deviation) of predictive distribution

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            independent variable
        return_std : bool, optional
            flag to return standard deviation (the default is False)
        sample_size : int, optional
            number of samples to draw from the predictive distribution
            (the default is None, no sampling from the distribution)

        Returns
        -------
        y : (N,) np.ndarray
            mean of the predictive distribution
        y_std : (N,) np.ndarray
            standard deviation of the predictive distribution
        y_sample : (N, sample_size) np.ndarray
            samples from the predictive distribution
        """

        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = np.matmul(X , w_sample.T)
            return y_sample
        y = np.matmul(X , self.w_mean)
        if return_std:
            y_var = 1 / self.beta + np.sum(np.matmul(X , self.w_cov) * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y
