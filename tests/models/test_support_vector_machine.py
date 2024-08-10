from models import SupportVectorMachine
import numpy as np

# raw X
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [8, 9]])

all_y = []

# toy dataset half/half
y_half = np.array([1, 1, 1, -1, -1, -1])

# toy dataset all negative
y_all_neg = np.zeros_like(y_half) - 1

# toy dataset all positive
y_all_pos = np.ones_like(y_half)

# toy dataset one negative
y_one_neg = np.array([1, 1, 1, 1, 1, -1])

# toy dataset one positive
y_one_pos = np.array([1, -1, -1, -1, -1, -1])

all_y = [y_half, y_all_neg, y_all_pos, y_one_neg, y_one_pos]

def test_SupportVectorMachine_classification():
    # get all kernels
    dummy_model = SupportVectorMachine()
    kernels = dummy_model.valid_kernels
    del dummy_model

    # test all targets
    for y in all_y:

        svm_models = [SupportVectorMachine() for _ in kernels]

        for kernel, model in zip(kernels, svm_models):
            # to eliminate randomness in params

            model.fit(X, y, C=10, itrs=1000, kernel=kernel)

        for model in svm_models:
            assert model.w.shape == (X.shape[1], )

            y_hat = model(X)
            assert y_hat.shape == (X.shape[0], )
            assert np.all(np.isin(y_hat, [-1, 1]))

            accuracy = model.accuracy(X, y)
            assert accuracy > 0.8