import numpy as np

# had problems with numerical precision, so needed these
def is_greater_or_close(a, b, rtol=1e-5, atol=1e-8):
    return np.logical_or(a > b, np.isclose(a, b, rtol=rtol, atol=atol))

def is_less_or_close(a, b, rtol=1e-5, atol=1e-8):
    return np.logical_or(a < b, np.isclose(a, b, rtol=rtol, atol=atol))

class SupportVectorMachine():
    def __init__(self, task='classification'):
        """A support vector machine model. Currently only supports binary classification with linear kernel.

        Args:
            task (str, optional): model task. Defaults to 'classification'.

        Raises:
            ValueError: if task not supported (only 'classification' available currently.)
        """
        self.valid_kernels = ['linear']
        self.valid_tasks = ['classification']

        if task not in self.valid_tasks:
            raise ValueError(f'Task {self.task} not supported')
        else:
            self.task = task 

        self.kernel = None # kernel function

        # for linear kernel only
        self.w = None # weights
        self.b = None # bias

        # for non-linear kernels
        self.X_suppor_vectors = None   
        self.y_suppor_vectors = None
        self.alpha = None # lagrange multipliers

        # whether model has been fitted to data
        self.fitted = False

    def _validate_X_y(self, X, y):
        # check type, shape, and values of X and y
        if X is not None:
            if X.ndim != 2:
                raise ValueError('X must be 2D and of shape (num_samples, num_features)')
            if X.shape[0] == 0:
                raise ValueError('X must not be empty')
            if X.shape[1] == 0:
                raise ValueError('X must not have zero features')
            if not isinstance(X, np.ndarray):
                raise ValueError('X must be a numpy array')
            if np.any(np.isnan(X)):
                raise ValueError('X must not contain any NaN values')
            if np.any(np.isinf(X)):
                raise ValueError('X must not contain any infinite values')
            if X.shape[0] <= 1:
                raise ValueError('Decision boundary cannot be fit to fewer than 2 samples')

            if self.fitted:
                if self.kernel == 'linear':
                    if X.shape[1] != self.w.shape[0]:
                        raise ValueError(f'X must have {self.w.shape[0]} features')

        if y is not None:
            if y.ndim != 1:
                raise ValueError('y must be 1D and of shape (num_samples,)')
            if y.shape[0] == 0:
                raise ValueError('y must not be empty')
            if not isinstance(y, np.ndarray):
                raise ValueError('y must be a numpy array')
            if np.any(np.isnan(y)):
                raise ValueError('y must not contain any NaN values')
            if np.any(np.isinf(y)):
                raise ValueError('y must not contain any infinite values')
            if self.task == 'classification':
                if not np.all(np.isin(y, [-1, 1])):
                    raise ValueError('y values must be binary and in set {-1, 1} for classification')

        if X is not None and y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError('X and y must have the same number of samples')

    def _get_initial_alpha(self, X):
        # initialize lagrange multipliers to zero
        alpha = np.zeros(shape=(X.shape[0],))
        return alpha

    def _check_KKT(self, y, alpha, u, C, epsilon=0.001):
        # check Karush-Kuhn-Tucker conditions to a tolerance epsilon. Returns boolean array of results (True if satisfied, False otherwise)
        kkt_results = np.zeros_like(alpha).astype(bool)

        u_y = u * y

        assert np.all(is_greater_or_close(alpha, 0)) and np.all(is_less_or_close(alpha, C)), 'alpha must be in range [0, C].'

        # classify each alpha - determines which condition to check
        alpha_condition = np.zeros_like(alpha)

        alpha_condition[np.isclose(alpha, 0)] = 1
        alpha_condition[np.isclose(alpha, C)] = 3
        alpha_condition[alpha_condition == 0] = 2 # the ones that weren't approx. = 0 or C

        assert np.all(np.isin(alpha_condition, [1, 2, 3])), 'failed to classify lagrange multipliers into == 0, == C, and in [0, C]'

        # Condition 1: α = 0 ⇒ y * u >= 1
        condition_1_mask = alpha_condition == 1
        condition_1_bool = is_greater_or_close(u_y[condition_1_mask], 1, rtol=0, atol=epsilon)
        kkt_results[condition_1_mask] = condition_1_bool

        # Condition 2: 0 < α < C ⇒ y * u = 1
        condition_2_mask = alpha_condition == 2
        condition_3_bool = np.isclose(u_y[condition_2_mask], 1, rtol=0, atol=epsilon)
        kkt_results[condition_2_mask] = condition_3_bool

        # Condition 3: α = C ⇒ y * u <= 1
        condition_3_mask = alpha_condition == 3
        condition_2_bool = is_less_or_close(u_y[condition_3_mask], 1, rtol=0, atol=epsilon)
        kkt_results[condition_3_mask] = condition_2_bool

        return kkt_results, alpha_condition

    def _get_psi(self, alpha, K, y):
        # objective function: minimize
        first_part = np.sum(np.outer(y, y) * np.outer(alpha, alpha) * K)

        second_part = np.sum(alpha)

        psi = (first_part * 0.5) - second_part

        return psi

    def _g(self, u):
        # sign function
        y_hat = (u > 0).astype(int)
        y_hat[y_hat==0] = -1
        return y_hat

    def _get_kernel(self, x1, x2, kernel_type = 'linear'):
        # kernel function. works for individual samples or matrices of samples
        if kernel_type == 'linear':
            return np.dot(x1, x2.T)

        if kernel_type == 'gaussian':
            raise NotImplementedError()

        raise ValueError(f'Kernel {kernel_type} not supported')

    def _get_L_H(self, alpha, y, i, j, C):
        # get upper (H) and lower (L) bounds for alpha_j
        alpha_i = alpha[i]
        alpha_j = alpha[j]
        y_i = y[i]
        y_j = y[j]

        # targets not equal
        if y_i != y_j:
            L = np.max([0, alpha_j - alpha_i])
            H = np.min([C, C + alpha_j - alpha_i])

        # targets equal
        else:
            L = np.max([0, alpha_i + alpha_j - C])
            H = np.min([C, alpha_i + alpha_j])

        return L, H

    def _get_support_vectors(self, X, y, alpha):
        # support vectors are samples where alpha > 0
        nonzero_alpha_mask = alpha > 0

        X_support_vectors = X[nonzero_alpha_mask]
        y_support_vectors = y[nonzero_alpha_mask]
        alpha_support_vectors = alpha[nonzero_alpha_mask]

        return X_support_vectors, y_support_vectors, alpha_support_vectors

    def _fit_linear(self, X, y, alpha, C, epsilon, itrs, dev):
        # simplified version of sequential minimal optimization for linear kernel

        # handle case where all y are the same
        if np.all(y == -1) or np.all(y == 1):
            y_val = y[0]
            w = np.zeros(shape=(X.shape[1],))
            b = y_val
            return w, b

        kernel_function = lambda x1, x2: self._get_kernel(x1, x2, kernel_type='linear')

        itr = 0

        while itr < itrs:
            # express w in terms of lagrange mutlipliers (alphas), X, and y
            w = (y * alpha).T @ X

            assert w.shape[0] == X.shape[1], f'mismatch between number of weights produced {w.shape[0]} and number of features {X.shape[1]}'

            # update b
            if np.all(alpha == 0):
                b = 0
            else:
                positive_alpha_mask = alpha > 0
                b = np.mean(y[positive_alpha_mask] - (X[positive_alpha_mask] @ w))

            # forward pass
            u = (X @ w) + b

            if dev:
                K = kernel_function(X, X)
                psi = self._get_psi(alpha, K, y)
                accuracy = (self._g(u) == y).astype(int).mean()

                print(f'Iteration {itr}: psi = {psi}, accuracy = {accuracy}')

            # kkt results
            kkt_results, alpha_condition = self._check_KKT(y, alpha, u, C, epsilon)

            if np.all(kkt_results):
                break

            # get error values
            E = u - y

            # select i and j
            all_idx = np.arange(0, len(y))

            all_idx_failed_kkt = all_idx[~kkt_results]

            E_failed_kkt = E[~kkt_results]

            # don't need to handle len(all_idx_failed_kkt) == 0 because the loop would have been broken

            if np.all(y[all_idx_failed_kkt] == y[all_idx_failed_kkt][0]):
                # handle case where the only samples failing kkt have the same target and so no progress can be made
                i = all_idx_failed_kkt[0]
                j = np.random.choice(all_idx[(all_idx != i)])

            elif len(all_idx_failed_kkt) == 1:
                # pair the one that failed kkt with a random sample
                i = all_idx_failed_kkt[0]
                j = np.random.choice(all_idx[(all_idx != i)])

            else:
                # pair the maximum and minimum error for samples that fail kkt.
                max_E_idx_failed_kkt = np.argmax(E_failed_kkt)
                i = all_idx_failed_kkt[max_E_idx_failed_kkt]

                # remove i
                all_idx_failed_kkt = np.delete(all_idx_failed_kkt, max_E_idx_failed_kkt)
                E_failed_kkt = np.delete(E_failed_kkt, max_E_idx_failed_kkt)

                min_E_idx_failed_kkt = np.argmin(E_failed_kkt)
                j = all_idx_failed_kkt[min_E_idx_failed_kkt]
                # (mostly) guarantees they will have different targets and so optimization can happen

            assert i != j and i in all_idx and j in all_idx, 'i and j not valid'

            # upper (H) and lower (L) bounds for alpha_j
            L, H = self._get_L_H(alpha, y, i, j, C)

            assert is_less_or_close(L,  H), 'L > H'

            eta = kernel_function(X[i], X[i]) + kernel_function(X[j], X[j]) - (2 * kernel_function(X[i], X[j]))

            if eta <= 0:
                raise NotImplementedError('handling for eta <= 0 not implemented')

            # get alpha_j_new_unclipped
            alpha_j_new_unclipped = alpha[j] + ((y[j] * (E[i] - E[j])) / eta)

            # clip it so it's in range [L, H]
            alpha_j_new = np.clip(a=alpha_j_new_unclipped, a_min=L, a_max=H)

            # compute alpha_i_new from alpha_j_new
            alpha_i_new = alpha[i] + (y[i] * y[j] * (alpha[j] - alpha_j_new))

            # update alpha
            alpha[i] = alpha_i_new
            alpha[j] = alpha_j_new

            itr += 1

        return w, b

    def fit(self, X: np.ndarray, y: np.ndarray, C=10, epsilon=0.001, itrs=1_000, kernel='linear', dev=False):
        """Fit support vector machine to data

        Args:
            X (np.ndarray): feature matrix of shape (num_samples, num_features)
            y (np.ndarray): target vector of shape (num_samples,)
            C (int, optional): regularization parameter. Small values result in 'softer' or larger margins with more misclassification.
             Large values result in stricter/smaller margin but potentially overfitting. Defaults to 10.
            epsilon (float, optional): tolerance for satisfying Karush-Kuhn-Tucker conditions. Defaults to 0.001.
            itrs (int, optional): maximum number of training iterations. Defaults to 1_000.
            kernel (str, optional): kernel function. Determines how samples are compared. Defaults to 'linear'.
            dev (bool, optional): whether to print training data. Defaults to False.

        Raises:
            ValueError: if kernel not supported
        """
        self._validate_X_y(X, y)

        if C <= 0:
            raise ValueError('C must be greater than 0')
        
        if epsilon <= 0:
            raise ValueError('epsilon must be greater than 0')

        alpha = self._get_initial_alpha(X)

        if kernel not in self.valid_kernels:
            raise ValueError(f'Kernel {kernel} not supported')

        self.kernel = kernel

        if self.kernel == 'linear':
            w, b = self._fit_linear(X, y, alpha, C, epsilon, itrs, dev)

            self.w = w
            self.b = b

        self.fitted = True

    def _forward(X, X_support_vectors, y_support_vectors, alpha_support_vectors, kernel_function, b):
        # evaluate SVM for arbitrary kernel. Not currently used

        first_part = ((y_support_vectors * alpha_support_vectors) @ kernel_function(X_support_vectors, X))

        u = first_part + b

        return u

    def __call__(self, X: np.ndarray):
        """Predict target values for input data

        Args:
            X (np.ndarray): feature matrix of shape (num_samples, num_features)

        Raises:
            ValueError: if model not fitted

        Returns:
            np.ndarray: predicted target values
        """
        if self.fitted == False:
            raise ValueError('model not fitted')

        self._validate_X_y(X, None)

        if self.task == 'classification':
            if self.kernel == 'linear':
                return self._g(X @ self.w + self.b)

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        """Compute accuracy of model on input data

        Args:
            X (np.ndarray): feature matrix of shape (num_samples, num_features)
            y (np.ndarray): target vector of shape (num_samples,)

        Raises:
            ValueError: if model not fitted

        Returns:
            float: accuracy of model on input data
        """
        if self.fitted == False:
            raise ValueError('model not fitted')

        self._validate_X_y(X, y)
        y_hat = self.__call__(X)
        return np.mean(y_hat == y)