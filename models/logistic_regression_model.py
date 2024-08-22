import numpy as np
from tqdm import tqdm
import warnings

class LogisticRegressionModel():
    def __init__(self):
        """
        A Logistic Regression model. Currently only supported for binary classification with an arbitrary number of features. 
        """
        self.fitted = False
        self.num_features = None
        self.b = None
        self.w = None
        self.params = {'w': self.w, 'b': self.b}
        self.solvers = ['GD', 'SGD', 'MBGD', 'SAG']

    def _validate_X_y(self, X=None, y=None):
        # validate input data
        if X is not None:
          if not isinstance(X, np.ndarray):
              raise TypeError('X must be a numpy array')
          if len(X.shape) != 2:
              raise ValueError(f'X must be a 2D array of shape (num_samples, num_features), instead got {X.shape}')
          if X.shape[1] != self.num_features:
            raise ValueError(f'Number of features in X must match number of features in model. X.shape[1] = {X.shape[1]}, num_features = {self.num_features}')
          if np.any(np.isnan(X)):
            raise ValueError('X contains NaN values')
          if np.any(np.isinf(X)):
            raise ValueError('X contains inf values')


        if y is not None:
          if not isinstance(y, np.ndarray):
              raise TypeError('y must be a numpy array')
          if len(y.shape) != 1:
              raise ValueError(f'y must be a 1D array of shape (num_samples, ), instead got {y.shape}')
          if not np.all(np.isin(y, [0, 1])):
            raise ValueError("all values of y must be either 0 or 1. We don't do multiclass here")

        if y is not None and X is not None:
          if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X must match number of samples in y (X.shape[0] = {X.shape[0]}, y.shape[0] = {y.shape[0]})')

    def _sigmoid(self, z):
        # numerically stable; resistant to overflow
        # about 2.5 x slower than normal sigmoid
        positive_mask = z >= 0
        negative_mask = z <= 0

        y_hat = np.empty_like(z)

        y_hat[positive_mask] = 1. / (1 + np.exp(-z[positive_mask]))

        exp_z = np.exp(z[negative_mask])
        y_hat[negative_mask] = exp_z / (1 + exp_z)

        return y_hat

    def _cross_entropy_stable(self, y, z, stable_threshold=20):
        # Takes in raw logits
        stable_mask = z <= stable_threshold
        unstable_mask = ~stable_mask

        cross_entropy_loss = np.empty_like(z)

        # for large z (in this case >= 20), log(e^z + 1) can be approximated by z. This makes loss calculation robust to overflow
        cross_entropy_loss[stable_mask] = np.log(np.exp(z[stable_mask]) + 1) - (y[stable_mask]* z[stable_mask])
        cross_entropy_loss[unstable_mask] = z[unstable_mask] - (y[unstable_mask] * z[unstable_mask])

        return cross_entropy_loss

    def _cross_entropy_stable_derivative(self, y, z):
        # ... Also takes in raw logits
        stable_mask = z <= 20
        unstable_mask = ~stable_mask

        d_z = np.empty_like(z)

        # derivatives based on piecewise function defined in _cross_entropy_stable
        d_z[stable_mask] = (np.exp(z[stable_mask]) / (np.exp(z[stable_mask]) + 1)) - y[stable_mask] # could make more efficient
        d_z[unstable_mask] = 1 - y[unstable_mask]

        return d_z

    def _compute_gradients(self, X, y):
        # partial derivatives of cross entropy loss w.r.t. parameters
        z = np.dot(X, self.w) + self.b
        d_z = self._cross_entropy_stable_derivative(y, z)

        d_w = np.dot(X.T, d_z) / d_z.shape[0]
        d_b = np.mean(d_z)

        return d_w, d_b

    def _shuffle_X_y(self, X, y):
        conjoined = np.concatenate([X, y.reshape(-1, 1)], axis=1)

        np.random.shuffle(conjoined)

        X = conjoined[:, :-1]

        y = conjoined[:, -1]

        return X, y

    def _fit_gd(self, X, y, num_iterations, lr, lr_decay_rate):
        # (batch) gradient descent

        for itr in tqdm(iterable=range(num_iterations), desc='Fitting model'):
            d_w, d_b = self._compute_gradients(X, y)

            self.w -= (d_w * lr)
            self.b -= (d_b * lr)

            lr *= lr_decay_rate

    def _fit_bgd(self, X, y, batch_size, num_iterations, lr, lr_decay_rate):
        # batch size = 1 --> stochastic gradient descent
        # batch size =/ 1 --> mini-batch gradient descent

        if num_iterations < X.shape[0] / batch_size:
            warnings.warn('number of iterations less than number of batches. Increase num_iterations to ensure SGD / MBGD computes gradients and fits to all data')

        # make batches
        num_batches = X.shape[0] // batch_size

        x_batches = np.array_split(X, num_batches)
        y_batches = np.array_split(y, num_batches)

        batch_idx = 0

        for itr in tqdm(iterable=range(num_iterations), desc='Fitting model'):
            x_batch = x_batches[batch_idx]
            y_batch = y_batches[batch_idx]

            d_w, d_b = self._compute_gradients(x_batch, y_batch)

            self.w -= (d_w * lr)
            self.b -= (d_b * lr)

            lr *= lr_decay_rate

            batch_idx = (batch_idx + 1) % num_batches # iterate through batches

    def _fit_sag(self, X, y, num_iterations, lr, lr_decay_rate):
        # stochastic average gradient
        num_samples = X.shape[0]

        # memory of all gradients for each training sample
        all_d_w = np.zeros(shape=(num_samples, self.num_features))
        all_d_b = np.zeros(shape=(num_samples, 1))

        if num_iterations < num_samples:
            warnings.warn('number of samples greater than number of iterations. Increase num_iterations to ensure SAG computes gradients and fits to all data')

        for itr in tqdm(iterable=range(num_iterations), desc='Fitting model'):
            # pick a random sample and compute gradients for it
            sample_idx_for_update = np.random.randint(num_samples)

            X_i = np.expand_dims(X[sample_idx_for_update], axis=0) # unsqeeze so gradient function works
            y_i = np.expand_dims(y[sample_idx_for_update], axis=0)

            d_w_i, d_b_i = self._compute_gradients(X_i, y_i)

            # update gradient memory for random sample
            all_d_w[sample_idx_for_update] = d_w_i
            all_d_b[sample_idx_for_update] =  d_b_i

            # update weights and biases
            self.w -= (np.mean(all_d_w, axis=0) * lr)
            self.b -= (np.mean(all_d_b) * lr)

            lr *= lr_decay_rate


    def fit(self, X: np.ndarray, y: np.ndarray, solver='SGD', num_iterations=25_000, shuffle=True, lr=0.1, lr_decay_rate=0.9999):
        """Fit model to data

        Available solvers : 'GD', 'MBGD', 'SGD', 'SAG'
        'GD' : Gradient Descent
        'MBGD' : Mini-Batch Gradient Descent (batch size = 10)
        'SGD' : Stochastic Gradient Descent
        'SAG' : Stochastic Average Gradient

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target vector
            solver (str, optional): solver to use to fit model to data. Defaults to 'SGD'.
            num_iterations (int, optional): number of training iterations. Defaults to 25_000.
            shuffle (bool, optional): shuffle data before fitting. Defaults to True.
            lr (float, optional): learning rate. Defaults to 0.1.
            lr_decay_rate (float, optional): exponential learning rate decay. Computed each iteration as lr *= lr_decay_rate. Defaults to 0.9999.

        Raises:
            ValueError: if solver is not one of 'GD', 'MBGD', 'SGD', 'SAG'
        """
        self.num_features = X.shape[1]

        # initialize weights and biases
        self.w = np.random.randn(self.num_features)
        self.b = np.random.randn(1)

        self._validate_X_y(X, y)

        if shuffle:
            X, y = self._shuffle_X_y(X, y)

        if solver == 'GD':
            self._fit_gd(X, y, num_iterations, lr, lr_decay_rate)

        elif solver == "MBGD":
            # in case num_samples < 10
            batch_size = np.min([X.shape[0], 10])
            self._fit_bgd(X, y, batch_size, num_iterations, lr, lr_decay_rate)

        elif solver == "SGD":
            self._fit_bgd(X, y, 1, num_iterations, lr, lr_decay_rate)

        elif solver == 'SAG':
            self._fit_sag(X, y, num_iterations, lr, lr_decay_rate)

        else:
            raise ValueError(f'"{solver}" is not a valid solver. Choose one of "GD", "MBGD", "SGD", or "SAG"')
        
        self.fitted = True

    def loss(self, X: np.ndarray, y: np.ndarray):
        """Compute cross entropy loss

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target vector

        Returns:
            float: cross entropy loss
        """
        if self.fitted == False:
            raise ValueError('Model must be fitted before computing loss')
        
        self._validate_X_y(X, y)

        z = np.dot(X, self.w) + self.b
        cross_entropy_loss = self._cross_entropy_stable(y, z)
        return cross_entropy_loss.mean()

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        """Compute model accuracy

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target vector

        Returns:
            float: model accuracy
        """
        if self.fitted == False:
            raise ValueError('Model must be fitted before computing accuracy')

        self._validate_X_y(X, y)

        y_hat = self.__call__(X)

        model_prediction = np.empty_like(y_hat)
        model_prediction[y_hat > 0.5] = 1
        model_prediction[y_hat < 0.5] = 0

        num_correct = np.sum(model_prediction == y)

        p_correct = num_correct / len(y)

        return p_correct

    def __call__(self, X: np.ndarray):
        """Make predictions

        Args:
            X (np.ndarray): feature matrix of shape (num_samples, num_features)

        Returns:
            np.ndarray: predictions, in range [0, 1]
        """
        if self.fitted == False:
            raise ValueError('Model must be fitted before making predictions')
        
        self._validate_X_y(X, None)

        return self._sigmoid(np.dot(X, self.w) + self.b)
    
