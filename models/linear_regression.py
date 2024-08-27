import numpy as np
from models.model_utils import get_condition_number

class LinearRegressionModel():
    def __init__(self):
        """
        A linear regression model
        """
        self.beta_hat = None # model coefficients
        self.fitted = False

    def _validate_X_y(self, X=None, y=None):
        # validate input data shape, type, and contents
        if X is not None:
            if not isinstance(X, np.ndarray):
              raise TypeError('X must be a numpy array')
            if X.ndim != 2:
              raise ValueError(f'X must be a 2D array of shape (num_samples, num_features), instead got {X.shape}')
            if np.any(np.isnan(X)):
                raise ValueError('X contains NaN values')
            if np.any(np.isinf(X)):
                raise ValueError('X contains inf values')

            if self.fitted: 
                if not self.beta_hat.shape[0] == (X.shape[1] + 1):
                    raise ValueError(f'Number of features in X must match coefficients (-1 for intercept). Instead got {X.shape[1]} and {self.beta_hat.shape[0]}')
          
        if y is not None:
            if not isinstance(y, np.ndarray):
              raise TypeError('y must be a numpy array')
            if y.ndim != 1:
              raise ValueError(f'y must be a 1D array of shape (num_samples, ), instead got {y.shape}')

        if y is not None and X is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError(f'Number of samples in X must match number of samples in y (X.shape[0] = {X.shape[0]}, y.shape[0] = {y.shape[0]})')

    def _add_intercept_column(self, X):
        # adds a column filled with ones for each sample so that self.beta_hat[0] is the intercept term
        intercept_column = np.ones(X.shape[0])
        intercept_column = intercept_column.reshape(-1, 1)
        X_intercept = np.concat([intercept_column, X], axis=1)
        return X_intercept

    def fit(self, X: np.ndarray, y: np.ndarray, check_condition=True, condition_threshold=1e8):
        """Fits linear regression model to input data
        
        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)
            y (np.ndarray): target data of shape (num_samples, )
            check_condition (bool): whether to check condition number of X
            condition_threshold (float): threshold for condition number of X. If condition number exceeds threshold, raises ValueError
        
        Raises:
            ValueError: if X is ill-conditioned (condition number > condition_threshold)

        Returns:
            None
        """
        # check X condition to make sure X.T @ X can be stably inverted
        condition_number = get_condition_number(X) # higher = bad

        # when inverting X, will lose log10(condition_number) digits of precision. So 1e8 corresponds to 8 digits of precision lost
        if check_condition and condition_number > condition_threshold:
            raise ValueError(f"""X is ill-conditioned: condition number = {condition_number}. Inversion of X.T @ X will not be numerically stable. 
            Consider dimensionality reduction techniques (ex. PCA) or feature selection (ex. variance-based selection).""")
        
        # reset model
        self.fitted=False
        self.beta_hat = None
        
        self._validate_X_y(X, y)

        X_intercept = self._add_intercept_column(X)

        # fit model
        try:
            beta_hat = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y
        except np.linalg.LinAlgError as e:
            raise ValueError('X.T @ X is singular.')

        # store coefficients
        self.beta_hat = beta_hat # shape = (number of features + 1, )

        self.fitted = True

    def __call__(self, X: np.ndarray):
        """Predicts target variable for input data

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)

        Raises:
            ValueError: if model is not fitted

        Returns:
            y_hat (np.ndarray): predicted target variable of shape (num_samples, )
        """
        if not self.fitted == True:
            raise ValueError('model not fitted')
            
        self._validate_X_y(X)

        X_intercept = self._add_intercept_column(X)
        
        # evaluate X using coefficients
        y_hat = X_intercept @ self.beta_hat

        return y_hat

    def r_squared(self, X: np.ndarray, y: np.ndarray):
        """Computes R^2 value for model on input data

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)
            y (np.ndarray): target data of shape (num_samples, )

        Raises:
            ValueError: if model is not fitted

        Returns:
            float: R^2 value
        """
        if not self.fitted == True:
            raise ValueError('model not fitted')
            
        self._validate_X_y(X, y)
        y_hat = self.__call__(X)

        y_mean = y.mean()
        
        sum_squares_residual = ((y_hat - y) ** 2).sum()
        sum_squares_total = ((y - y_mean) ** 2).sum()
    
        r_squared = 1 - (sum_squares_residual / sum_squares_total)

        return r_squared