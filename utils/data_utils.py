import numpy as np

def _validate_X_y(X=None, y=None):
        """Basic data validation to check X and y type, shape, and consistency"""
        if X is not None:
          if not isinstance(X, np.ndarray):
              raise TypeError('X must be a numpy array')
          if len(X.shape) != 2:
              raise ValueError(f'X must be a 2D array of shape (num_samples, num_features), instead got {X.shape}')

        if y is not None:
          if not isinstance(y, np.ndarray):
              raise TypeError('y must be a numpy array')
          if len(y.shape) != 1:
              raise ValueError(f'y must be a 1D array of shape (num_samples, ), instead got {y.shape}')

        if y is not None and X is not None:
          if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X must match number of samples in y (X.shape[0] = {X.shape[0]}, y.shape[0] = {y.shape[0]})')


def split_dataset(X, y, test_size=0.2, shuffle=True):
    """Split dataset into training and testing sets

    Args:
        X (np.ndarray): feature matrix
        y (np.ndarray): target vector
        test_size (float, optional): percentage of data to be used for testing. Defaults to 0.2.
        shuffle (bool, optional): shuffle data before splitting. Defaults to True.

    Returns:
        tuple[np.ndarray]: X_train, y_train, X_test, y_test
    """
    _validate_X_y(X, y)

    conjoined = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    if shuffle:
        np.random.shuffle(conjoined)

    split_idx = int(X.shape[0] * (1 - test_size))
    conjoined_train = conjoined[:split_idx]
    conjoined_test = conjoined[split_idx:]

    X_train = conjoined_train[:, :-1]
    y_train = conjoined_train[:, -1]

    X_test = conjoined_test[:, :-1]
    y_test = conjoined_test[:, -1]

    return X_train, y_train, X_test, y_test