from models import LogisticRegressionModel
import numpy as np

# toy dataset parameterized
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
w, b = np.array([0.05, 0.05]), -0.5  # weight, bias
z = np.dot(X, w) + b
y_hat_fake = 1 / (1 + np.exp(-z))
y_param = (y_hat_fake > 0.5).astype(int)

# toy dataset all negative
y_all_neg = np.zeros_like(y_param)

# toy dataset all positive
y_all_pos = np.ones_like(y_param)

all_y = [y_param, y_all_neg, y_all_pos]

def test_LogisticRegressionModel():
    # get all solvers
    dummy_model = LogisticRegressionModel()
    solvers = dummy_model.solvers
    del dummy_model

    # test all targets
    for y in all_y:

        log_reg_models = [LogisticRegressionModel() for _ in solvers]

        for solver, model in zip(solvers, log_reg_models):
            # to eliminate randomness in params
            model.w = np.zeros_like(model.w)
            model.b = 0

            model.fit(X, y, solver=solver)

        for model in log_reg_models:
            assert model.w.shape == (X.shape[1], )

            y_hat = model(X)
            assert y_hat.shape == (X.shape[0], )
            assert np.all(y_hat >= 0) and np.all(y_hat <= 1)

            accuracy = model.accuracy(X, y)
            assert accuracy == 1.0