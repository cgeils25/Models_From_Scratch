import numpy as np
from models import KNearestNeighborsClassifier

def test_KNN_binary_all_same():
    # create toy binary classification data with all negative samples
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 0, 0])

    model = KNearestNeighborsClassifier()
    model.fit(X, y)

    # test prediction
    for k in range(1, X.shape[0]):
        y_hat = model(X, k=k)
        assert np.all(y_hat == y), f'all binary predictions should be negative (0) for all values of k (failed for {k})'
        assert model.accuracy(X, y, k=k) == 1.0, f'accuracy should be 100% for all values of k (failed for {k})'

    # same but all positive
    y = np.array([1, 1, 1, 1])

    model = KNearestNeighborsClassifier()
    model.fit(X, y)

    # test prediction
    for k in range(1, X.shape[0]):
        y_hat = model(X, k=k)
        assert np.all(y_hat == y), f'all binary predictions should be positive (1) for all values of k (failed for {k})'
        assert model.accuracy(X, y, k=k) == 1.0, f'accuracy should be 100% for all values of k (failed for {k})'

def test_binary_one_neighbor():
    # because no train/test split and k=1, predictions should be 100% accurate
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    model = KNearestNeighborsClassifier()
    model.fit(X, y)

    # test prediction
    y_hat = model(X, k=1)
    assert np.all(y_hat == y), f'binary predictions should be 100% accurate for k=1'
    assert model.accuracy(X, y, k=1) == 1.0, f'accuracy should be 100% for k=1'

def test_multiclass_one_neighbor():
    # because no train/test split and k=1, predictions should be 100% accurate
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

    model = KNearestNeighborsClassifier(type='multiclass')
    model.fit(X, y)

    # test prediction
    y_hat = model(X, k=1)
    assert np.all(y_hat == y), f'multiclass predictions should be 100% accurate for k=1'
    assert model.accuracy(X, y, k=1) == 1.0, f'accuracy should be 100% for k=1'

def test_multiclass_all_same():
    # create toy multiclass classification data with all samples in the same class
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])

    model = KNearestNeighborsClassifier(type='multiclass')
    model.fit(X, y)

    # test prediction
    for k in range(1, X.shape[0]):
        y_hat = model(X, k=k)
        assert y_hat.shape == y.shape, f'predictions should have the same shape as y (failed for {k})' # technically redundant but easier to debug
        assert np.all(y_hat == y), f'all multiclass predictions should be the same for all values of k (failed for {k})'
        assert model.accuracy(X, y, k=k) == 1.0, f'accuracy should be 100% for all values of k (failed for {k})'

    # same deal but all second class
    y = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])

    model = KNearestNeighborsClassifier(type='multiclass')
    model.fit(X, y)

    # test prediction
    for k in range(1, X.shape[0]):
        y_hat = model(X, k=k)
        assert y_hat.shape == y.shape, f'predictions should have the same shape as y (failed for {k})' # technically redundant but easier to debug
        assert np.all(y_hat == y), f'all multiclass predictions should be the same for all values of k (failed for {k})'
        assert model.accuracy(X, y, k=k) == 1.0, f'accuracy should be 100% for all values of k (failed for {k})'
