import numpy as np

class Ridge:
    """
    Ridge regression model
    """
    def __init__(self, alpha: float = 1.0):
        """Initialize the Ridge regression model

        Args:
            alpha (float, optional): Regularization parameter. Defaults to 1.0.
        """
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the data

        Args:
            X (np.ndarray): features
            y (np.ndarray): target
        """
        X_ones = np.hstack((np.ones((X.shape[0], 1)), X))
        U, D, Vt = np.linalg.svd(X_ones, full_matrices=True)
        self.beta_hat = Vt.T @ np.linalg.inv(D.T @ D + self.alpha * np.eye(X_ones.shape[1])) @ D @ U.T @ y
