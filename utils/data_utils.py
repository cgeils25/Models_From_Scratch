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
          if y.ndim not in [1, 2]:
              raise ValueError(f'y must be a 1D or 2D array, instead got shape {y.shape}')

        if y is not None and X is not None:
          if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X must match number of samples in y (X.shape[0] = {X.shape[0]}, y.shape[0] = {y.shape[0]})')


def split_dataset(X: np.ndarray, y: np.ndarray, test_size=0.2, shuffle=True, seed=None):
    """Split dataset into training and testing sets

    Args:
        X (np.ndarray): feature matrix
        y (np.ndarray): target vector or matrix
        test_size (float, optional): percentage of data to be used for testing. Defaults to 0.2.
        shuffle (bool, optional): shuffle data before splitting. Defaults to True.

    Returns:
        tuple[np.ndarray]: X_train, y_train, X_test, y_test
    """
    _validate_X_y(X, y)

    if y.ndim == 1:
        conjoined = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    else:
        conjoined = np.concatenate([X, y], axis=1)

    if shuffle:
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(conjoined)
        else:
            np.random.shuffle(conjoined)

    split_idx = int(X.shape[0] * (1 - test_size))
    conjoined_train = conjoined[:split_idx]
    conjoined_test = conjoined[split_idx:]
    
    if y.ndim == 1:
        X_train = conjoined_train[:, :-1]
        y_train = conjoined_train[:, -1]

        X_test = conjoined_test[:, :-1]
        y_test = conjoined_test[:, -1]
    else:
        num_outputs = y.shape[1]
        X_train = conjoined_train[:, :-num_outputs]
        y_train = conjoined_train[:, -num_outputs:]

        X_test = conjoined_test[:, :-num_outputs]
        y_test = conjoined_test[:, -num_outputs:]

    return X_train, y_train, X_test, y_test
