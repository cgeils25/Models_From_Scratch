import numpy as np
from models.linear_regression import LinearRegressionModel


X = np.array([[1, 3], [4, -3], [8, 0], [9, 2], [5, 6]])

beta_1 = np.array([1, 2, 3])
beta_2 = np.array([1, 1, 1])
beta_3 = np.array([0, 0, 0])
beta_4 = np.array([-1, -2, -3])
beta_5 = np.array([-1, -1, -1])

betas = [beta_1, beta_2, beta_3]

intercept_column = np.ones(X.shape[0])
intercept_column = intercept_column.reshape(-1, 1)
X_intercept = np.concat([intercept_column, X], axis=1)

all_y = [X_intercept @ beta for beta in betas]

def test_linear_regression_model():
    models = [LinearRegressionModel() for _ in range(len(all_y))]
    
    for model, y, beta in zip(models, all_y, betas):
        model.fit(X, y)

        # check coefficients match shape of data
        assert model.beta_hat.shape[0] == X.shape[1] + 1

        # check predicted coefficients match true values
        assert np.allclose(model.beta_hat, beta)

        # check outputs match true values
        assert np.allclose(model(X), y)

        # check r squared is 1, since there is no noise
        if not (y == y[0]).all(): # if all elements are the same, r_squared will be nan because sum_squares_total = 0
            assert np.isclose(model.r_squared(X, y), 1)

