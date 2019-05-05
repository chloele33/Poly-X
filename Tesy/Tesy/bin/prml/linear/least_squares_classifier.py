import numpy as np
from prml.linear.classifier import Classifier
from prml.preprocess.label_transformer import LabelTransformer


class LeastSquaresClassifier(Classifier):
    """
    Least squares classifier model

    X : (N, D)
    W : (D, K)
    y = argmax_k X @ W
    """

    def __init__(self, W=None):
        np.ndarray(W)
        self.W = W

    def fit(self, X, t):
        np.ndarray(X)
        np.ndarray(t)
        """
        least squares fitting for classification

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) or (N, K) np.ndarray
            training dependent variable
            in class index (N,) or one-of-k coding (N,K)
        """
        if t.ndim == 1:
            t = LabelTransformer().encode(t)
        self.W = np.matmul(np.linalg.pinv(X) , t)

    def classify(self, X):
        np.ndarray(X)
        """
        classify input data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable to be classified

        Returns
        -------
        (N,) np.ndarray
            class index for each input
        """
        return np.argmax(np.matmul(X , self.W), axis=-1)
