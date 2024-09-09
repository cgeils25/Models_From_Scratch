import numpy as np

class GaussianNaiveBayesClassifer:
    def __init__(self):
        """
        A Naive Bayes classifier which calculates probabilities of each class given a sample's features using a Gaussian distribution
        """
        # means and standard deviations obtained from training data: used for gaussian pdf
        # shape: number of features, number of y-values
        self.means = None
        self.stds = None

        # unique y-values from fitting model
        self.y_vals = None

        # proportions of y-values in training data
        self.prior_probabilities = None

        # whether or not model has been fitted to data
        self.fitted = False

    def _validate_X_y(self, X=None, y=None):
        # validate input data shape, type, and contents
        if X is not None:
            if not isinstance(X, np.ndarray):
              raise TypeError('X must be a numpy array')
            if X.ndim != 2:
              raise ValueError(f'X must be a 2D array of shape (num_samples, num_features), instead got {X.shape}')
            if np.any(np.isnan(X)):
                raise ValueError('X contains NaN values')
            if np.any(np.isinf(X)):
                raise ValueError('X contains inf values')

            if self.fitted: 
               if not self.means.shape[1] == self.stds.shape[1] == X.shape[1]:
                   raise ValueError(f'number of features in X ({X.shape[1]}) does not match stored mean or standard devation values ({self.means.shape[1]})')
          
        if y is not None:
            if not isinstance(y, np.ndarray):
              raise TypeError('y must be a numpy array')
            if y.ndim != 1:
              raise ValueError(f'y must be a 1D array of shape (num_samples, ), instead got {y.shape}')
            if not np.all(np.unique(y) == np.arange(np.unique(y).shape[0])):
                raise ValueError(f'unique values in y must be ordered counting from 0: [0, 1, 2 ...], instead got {np.unique(y)}')

        if y is not None and X is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError(f'Number of samples in X must match number of samples in y (X.shape[0] = {X.shape[0]}, y.shape[0] = {y.shape[0]})')

    def _calculate_gaussian_probability(self, X, mean, std):
        z_score = (X - mean) / std
    
        curve_height = 1 / (std * np.sqrt(2 * np.pi))
    
        p = curve_height * np.exp(-0.5 * np.square(z_score))
    
        return p

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model to input data

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)
            y (np.ndarray): target data of shape (num_samples, )
        """
        self._validate_X_y(X, y)
        
        self.y_vals = np.unique(y)
        self.means = np.empty(shape=(self.y_vals.shape[0], X.shape[1]))
        self.stds = np.empty_like(self.means)
        self.prior_probabilities = np.empty_like(self.y_vals, dtype=np.float64)
        
        # compute mean and standard deviation for each unique y value, then store it
        for i, y_val in enumerate(self.y_vals):
            # to extract samples from X for which target is a specific value
            y_val_mask = y == y_val

            # compute means and standard deviation for X given specific y value
            y_val_mean = X[y_val_mask].mean(axis=0)
            y_val_std = X[y_val_mask].std(axis=0) 

            # store them
            self.means[i] = y_val_mean
            self.stds[i] = y_val_std

            # compute proportion of data corresponding to particular y-value aka prior probability
            y_val_prior_probability = (y == y_val).mean()

            # store
            self.prior_probabilities[i] = y_val_prior_probability

        # sanity checks
        assert self.means.shape[0] == self.stds.shape[0] == self.y_vals.shape[0]
        assert self.means.shape[1] == self.stds.shape[1] == X.shape[1]

        self.fitted = True

    def calculate_posterior_probability(self, X: np.ndarray):
        """Calculate probability of each class given input data (aka posterior probability)

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)

        Raises:
            ValueError: if model is not fitted

        Returns:
            p_y_vals_given_x (np.ndarray): posterior probabilities (probability of each class given features of samples)
            of shape (num_samples, num_y_values)
        """
        if not self.fitted:
            raise ValueError('model not yet fitted')
            
        self._validate_X_y(X)
        
        # shape is (number of samples, number of unique targets)
        p_x_given_y_vals = np.empty(shape=(X.shape[0], self.y_vals.shape[0]))

        # iterate over means, standard deviations, and corresponding y-values to compute P(x | y_val)
        for i, (mean, std, y_val) in enumerate(zip(self.means, self.stds, self.y_vals)):
            # independent probabilities for each feature given y_val
            p_feature_given_y_val = self._calculate_gaussian_probability(X, mean, std)

            # sanity checks
            assert p_feature_given_y_val.shape == X.shape, 'feature probabilities shape does not match X shape'

            # assume conditional independence, so P(A) and P(B) and ... = P(A) * P(B) * ...
            p_x_given_y_val = np.prod(p_feature_given_y_val, axis=1)
            
            p_x_given_y_vals[:, i] = p_x_given_y_val

        # calculate P(y_val | x) = P(x | y_val) * P(y_val) / (sum(P(x | y) * P(y)) for all y_vals)
        p_y_vals_given_x = (p_x_given_y_vals * self.prior_probabilities) / \
        (p_x_given_y_vals @ self.prior_probabilities).reshape(p_x_given_y_vals.shape[0], 1)

        # more sanity checks
        assert p_y_vals_given_x.shape == p_x_given_y_vals.shape, 'posterior probabilities shape does not match conditional'
        assert np.allclose(p_y_vals_given_x.sum(axis=1), 1), 'posterior probabilities do not sum to 1 for all samples'
        
        return p_y_vals_given_x

    def __call__(self, X: np.ndarray):
        """Predict class for input data. Class is the given by determined based on the highest posterior probability

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)

        Raises:
            ValueError: if model is not fitted

        Returns:
            y_hat (np.ndarray): predicted class of shape (num_samples, )
        """
        if not self.fitted:
            raise ValueError('model not yet fitted') # this is redundant because of calculate_posterior_probability but keeping it just in case
    
        self._validate_X_y(X)

        # get probabilities of each y-value given features
        posterior_probabilities = self.calculate_posterior_probability(X)

        # predicted y-value is the one with the highest P(y | x)
        y_hat = np.argmax(posterior_probabilities, axis=1)
        # can do argmax here because _validate_X_y forces unique y-values to be like [0, 1, 2...]

        return y_hat

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        """Computes accuracy of model on input data

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)
            y (np.ndarray): target data of shape (num_samples, )

        Returns:
            p_correct (float): proportion of correct predictions
        """
        self._validate_X_y(X, y)
        
        y_hat = self.__call__(X)

        p_correct = (y == y_hat).mean()

        return p_correct