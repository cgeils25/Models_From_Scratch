import numpy as np

class KNearestNeighborsClassifier:
    """
    A K-Nearest Neighbors Classifier
    """
    def __init__(self, type='binary'):
        """Initialize a K-Nearest Neighbors Classifier. Supports binary and muliclass classification.

        Args:
            type (str, optional): Type of classification. Defaults to 'binary'. Valid types are 'binary' and 'multiclass'.

        Raises:
            ValueError: if type is not 'binary' or 'multiclass'
        """
        self.fitted = False
        valid_types = ['binary', 'multiclass']

        if type not in valid_types:
            raise ValueError(f'"{type}" is not a valid type. Choose from {valid_types}')
        
        self.type = type

        self.valid_distance_metrics = ['euclidean']
    
    def _validate_X_y(self, X: np.ndarray, y: np.ndarray):
        # check type, shape, and values of X and y
        if X is not None:
            if X.ndim != 2:
                raise ValueError('X must be 2D and of shape (num_samples, num_features)')
            if X.shape[0] == 0:
                raise ValueError('X must not be empty')
            if X.shape[1] == 0:
                raise ValueError('X must not have zero features')
            if not isinstance(X, np.ndarray):
                raise ValueError('X must be a numpy array')
            if np.any(np.isnan(X)):
                raise ValueError('X must not contain any NaN values')
            if np.any(np.isinf(X)):
                raise ValueError('X must not contain any infinite values')
            if X.shape[0] <= 1:
                raise ValueError('Decision boundary cannot be fit to fewer than 2 samples')

        if y is not None:
            if y.shape[0] == 0:
                raise ValueError('y must not be empty')
            if not isinstance(y, np.ndarray):
                raise ValueError('y must be a numpy array')
            if np.any(np.isnan(y)):
                raise ValueError('y must not contain any NaN values')
            if np.any(np.isinf(y)):
                raise ValueError('y must not contain any infinite values')
            if self.type == 'binary':
                if y.ndim != 1:
                    raise ValueError('y must be 1D and of shape (num_samples,)')
                if not np.all(np.isin(y, [0, 1])):
                    raise ValueError(f'y must contain only binary class labels 0 and 1, instead got {np.unique(y)}')
            if self.type == 'multiclass':
                if y.ndim != 2:
                    raise ValueError('y must be 2D and of shape (num_samples, num_classes)')
                if not np.all(np.isin(y, [0, 1])):
                    raise ValueError(f'y must contain only values 0 and 1 (one-hot encoding), instead got {np.unique(y)}')
                
        if X is not None and y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError('X and y must have the same number of samples')

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data (aka just save it)

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target vector or matrix. If binary must be 1D, if multiclass must be 2D one-hot encoded
        """
        self._validate_X_y(X, y)
        self.X_train = X
        self.y_train = y
        self.fitted = True

    def __call__(self, X: np.ndarray, k: int = 3, distance_metric: str = 'euclidean'):
        """Make predictions on new data

        Args:
            X (np.ndarray): feature matrix
            k (int, optional): size of neighborhood. Defaults to 3.
            distance_metric (str, optional): method for calculating distance from new samples to training
            samples to determine neighborhood. Defaults to 'euclidean'. Currently only supports 'euclidean'.

        Raises:
            ValueError: if model is not fitted
            ValueError: if distance_metric is not supported
            ValueError: if k is less than 1 or greater than the number of samples in the training set

        Returns:
            y_hat (np.ndarray): predicted target vector
        """
        if not self.fitted:
            raise ValueError('Model must be fit before making predictions')
        
        if not distance_metric in self.valid_distance_metrics:
            raise ValueError(f'Distance metric "{distance_metric}" not supported. Choose from {self.valid_distance_metrics}')
        
        self._validate_X_y(X, None)

        if k < 1:
            raise ValueError('k must be greater than 0')
        if k > self.X_train.shape[0]:
            raise ValueError('k must not be greater than the number of samples in the training set')

        if distance_metric == 'euclidean':

            distances = np.linalg.norm(X - np.expand_dims(self.X_train, axis=1), axis=2)

            nearest_idx = np.argsort(distances, axis=0)[:k, :]

            if self.type == 'binary':
                y_hat = (self.y_train[nearest_idx].mean(axis=0) > 0.5).astype(int)
            if self.type == 'multiclass':
                max_class_idxs = np.argmax(np.mean(self.y_train[nearest_idx], axis=0), axis=1)
                y_hat = np.zeros((X.shape[0], self.y_train.shape[1]))
                y_hat[np.arange(X.shape[0]), max_class_idxs] = 1
            
        else:
            raise ValueError(f'Distance metric "{distance_metric}" not supported')

        return y_hat
    
    def accuracy(self, X: np.ndarray, y: np.ndarray, k: int = 3, distance_metric='euclidean'):
        """Compute the accuracy of the model on new data

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target vector or matrix. If binary must be 1D, if multiclass must be 2D one-hot encoded
            k (int, optional): size of neighborhood. Defaults to 3.
            distance_metric (str, optional): method for calculating distance from new samples to training
            samples to determine neighborhood. Defaults to 'euclidean'. Currently only supports 'euclidean'.

        Returns:
            accuracy (np.float64): model accuracy
        """
        y_hat = self.__call__(X, k, distance_metric) # no checking fitted or valdating X and y because __call__ does
        
        if self.type == 'binary':
            accuracy = (y == y_hat).mean()
        else:
            accuracy = np.all(y == y_hat, axis=1).mean()

        return accuracy
