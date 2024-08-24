"""
Utility functions used by models, not intended to be used directly (hence no input validation)
"""

import numpy as np

def get_condition_number(X):
    """Computes condition number of matrix X from singular value decomposition

    Args:
        X (np.ndarray): input matrix

    Returns:
        condition_number (float): condition number of X
    """
    # computes matrix condition number of X. Higher condition number implies X is ill-conditioned and inversion will not be numerically stable
    _, S, _ = np.linalg.svd(X)

    # because S (singular values) is sorted in descending order
    max_eigenvalue = S[0]
    min_eigenvalue = S[-1]

    assert min_eigenvalue > 0, 'minimum eigenvalue <= 0'

    condition_number = max_eigenvalue / min_eigenvalue

    return condition_number