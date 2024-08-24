import numpy as np
from models.model_utils import get_condition_number
import pdb

# singular matrix
X_singular = np.array([[1, 1], [1, 1]])

def test_get_condition_number_singular_matrix():
    condition_number = get_condition_number(X_singular)

    assert condition_number > 1e10 or condition_number in [np.nan, np.inf], '''condition number for singular matrix should be infinite or nan. Large also works because
    division by 0 and numerical precision... you know the drill'''


def test_get_condition_number_nonsingular_constructed_matrix():
    # construct a matrix from known eigenvalues and eigenvectors
    eigenvalues = np.array([1, 2])
    eigenvectors = np.array([[1, 1], [1, -1]])

    X_condition_true = eigenvalues.max() / eigenvalues.min()

    X_constructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)

    X_condition = get_condition_number(X_constructed)

    assert np.isclose(X_condition, X_condition_true), 'condition number for constructed non-singular matrix should be max_eigenvalue/min_eigenvalue'