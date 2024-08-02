from utils.data_utils import split_dataset
import numpy as np

def test_split_dataset():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])

    X_train, y_train, X_test, y_test = split_dataset(X, y, test_size=0.2, shuffle=False)

    assert X_train.shape == (4, 2)
    assert y_train.shape == (4, )
    assert X_test.shape == (1, 2)
    assert y_test.shape == (1, )

    assert np.array_equal(X_train, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    assert np.array_equal(y_train, np.array([0, 1, 0, 1]))
    assert np.array_equal(X_test, np.array([[9, 10]]))
    assert np.array_equal(y_test, np.array([0]))

