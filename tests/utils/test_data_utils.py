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

def test_split_dataset_seeded():
    # test that using the same seed produces the same result
    X_1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_1 = np.array([0, 1, 0, 1, 0])
    X_2 = X_1.copy()
    y_2 = y_1.copy()

    X_train_1, y_train_1, X_test_1, y_test_1 = split_dataset(X_1, y_1, test_size=0.2, shuffle=True, seed=1738)
    X_train_2, y_train_2, X_test_2, y_test_2 = split_dataset(X_2, y_2, test_size=0.2, shuffle=True, seed=1738)

    assert X_train_1.shape == X_train_2.shape == (4, 2)
    assert y_train_1.shape == y_train_2.shape == (4, )
    assert X_test_1.shape == X_test_2.shape == (1, 2)
    assert y_test_1.shape == y_test_2.shape == (1, )

    assert np.array_equal(X_train_1, X_train_2)
    assert np.array_equal(y_train_1, y_train_2)
    assert np.array_equal(X_test_1, X_test_2)
    assert np.array_equal(y_test_1, y_test_2)

def test_split_dataset_multi_output():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])

    X_train, y_train, X_test, y_test = split_dataset(X, y, test_size=0.2, shuffle=False)

    assert X_train.shape == (4, 2)
    assert y_train.shape == (4, 2)
    assert X_test.shape == (1, 2)
    assert y_test.shape == (1, 2)

    assert np.array_equal(X_train, X[:4]), f"X not split correctly into X_train, expected {X[:4]} but got {X_train}"
    assert np.array_equal(y_train, y[:4]), f"y not split correctly into y_train, expected {y[:4]} but got {y_train}"
    assert np.array_equal(X_test, X[4:]), f"X not split correctly into X_test, expected {X[4:]} but got {X_test}"
    assert np.array_equal(y_test, y[4:]), f"y not split correctly into y_test, expected {y[4:]} but got {y_test}"
