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

        if not np.all(np.unique(y) == np.arange(np.unique(y).shape[0])):
            raise ValueError(f'unique values in y must be ordered counting from 0: [0, 1, 2 ...], instead got {np.unique(y)}')
        
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

    def _calculate_joint_probability(self, X: np.ndarray):
        '''Calculate likelihood of input data given each class. Assumes conditional independence of features. 
        Shape is of returned matrix p_x_given_y_vals is (num_samples, num_y_values)
        where rows correspond to samples and columns correspond to y-values
        '''
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
        
        return p_x_given_y_vals
        
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
        
        # get joint probability of each sample given each class
        p_x_given_y_vals = self._calculate_joint_probability(X)

        # calculate P(y_val | x) = P(x | y_val) * P(y_val) / (sum(P(x | y) * P(y)) for all y_vals)
        p_y_vals_given_x = (p_x_given_y_vals * self.prior_probabilities) / \
        (p_x_given_y_vals @ self.prior_probabilities).reshape(p_x_given_y_vals.shape[0], 1)

        if np.any((p_x_given_y_vals @ self.prior_probabilities).reshape(p_x_given_y_vals.shape[0], 1) == 0):
            raise ValueError('''Total probability of evidence is 0 for some samples. This will result in division by zero 
            and nan values in posterior probabilities. Can be due to large class imbalance.''')

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
        joint_probabilities = self._calculate_joint_probability(X)
        # can make predictions based on joint probabilities because P(x) (the divisor for posteriors) is constant for all y-values

        # predicted y-value is the one with the highest P(y | x)
        y_hat = np.argmax(joint_probabilities, axis=1)
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

    def _precision_binary(self, X: np.ndarray, y: np.ndarray):
        # compute precision on positive class for binary classification
        y_hat = self.__call__(X)

        positive_mask = y == 1

        num_true_positive = (y[positive_mask] == y_hat[positive_mask]).sum()

        num_positive = (y_hat == 1).sum() # aka TP + FP

        precision = num_true_positive / num_positive

        return precision

    def _precision_multiclass(self, X: np.ndarray, y: np.ndarray):
        # compute precision for multiclass classification
        y_hat = self.__call__(X)

        # precision for each class
        precisions = np.empty(shape=(self.y_vals.shape[0]))

        # iterate through classes to calculate precision
        for i, y_val in enumerate(self.y_vals):
            y_val_mask = y == y_val

            num_true_positive = (y[y_val_mask] == y_hat[y_val_mask]).sum()

            num_positive = (y_hat == y_val).sum()

            precision = num_true_positive / num_positive

            precisions[i] = precision

        return precisions
    
    def precision(self, X: np.ndarray, y: np.ndarray, average = False):
        """Computes precision of model on input data.

        For binary classification: Precision = TP / (TP + FP) where TP = true positive, FP = false positive

        For multiclass classification: Precision = TP / (TP + FP) for each class. Optionally average across classes.

        Interpretation: If the model predicts a sample is a given positive class, we expect the model to be correct [precision] % of the time. 

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)
            y (np.ndarray): target data of shape (num_samples, )
            average (bool): whether to take a macro-average of precision across classes. Only applies to multiclass classification

        Returns:
            precision (float): proportion of true positive predictions
            If multiclass classification and average is True, returns average precision across classes.
            Otherwise, returns array of precisions for each class. Index corresponds to class value (ex: 0, 1, 2, ...)
        """
        self._validate_X_y(X, y)

        # single class case
        if self.y_vals.shape[0] == 1:
            return 1

        # binary case
        elif self.y_vals.shape[0] == 2:
            return self._precision_binary(X, y)

        # multiclass case
        elif self.y_vals.shape[0] > 2:
            if average:
                # return average precision across classes
                return self._precision_multiclass(X, y).mean()
            # return array of precisions for each class
            return self._precision_multiclass(X, y)
    
    def _recall_binary(self, X: np.ndarray, y: np.ndarray):
        # compute recall for binary classification
        y_hat = self.__call__(X)
        
        positive_mask = y == 1

        num_true_positive = (y_hat[positive_mask] == y[positive_mask]).sum()

        num_false_negative = (y_hat[positive_mask] != y[positive_mask]).sum()

        recall = num_true_positive / (num_true_positive + num_false_negative)

        return recall

    def _recall_multiclass(self, X: np.ndarray, y: np.ndarray):
        # compute recall for multiclass classification
        y_hat = self.__call__(X)

        # recall for each class
        recalls = np.empty(shape=(self.y_vals.shape[0]))

        # iterate through classes to calculate recall
        for i, y_val in enumerate(self.y_vals):
            y_val_mask = y == y_val

            num_true_positive = (y[y_val_mask] == y_hat[y_val_mask]).sum()

            num_false_negative = (y_hat[y_val_mask] != y[y_val_mask]).sum()

            recall = num_true_positive / (num_true_positive + num_false_negative)

            recalls[i] = recall

        return recalls

    def recall(self, X: np.ndarray, y: np.ndarray, average=False):
        """Computes recall of model on input data.

        For binary classification: Recall = TP / (TP + FN) where TP = true positive, FN = false negative

        For multiclass classification: Recall = TP / (TP + FN) for each class. Optionally average across classes.

        Interpretation: If a sample belongs to a given positive class, we expect the model to correctly identify it [recall] % of the time.

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)
            y (np.ndarray): target data of shape (num_samples, )
            average (bool): whether to take a macro-average of recall across classes. Only applies to multiclass classification

        Returns:
            recall (float): proportion of true positive predictions
            If multiclass classification and average is True, returns average recall across classes.
            Otherwise, returns array of recalls for each class. Index corresponds to class value (ex: 0, 1, 2, ...)
        """
        self._validate_X_y(X, y)

        # single class case
        if self.y_vals.shape[0] == 1:
            return 1

        # binary case
        elif self.y_vals.shape[0] == 2:
            return self._recall_binary(X, y)
        
        # multiclass case
        elif self.y_vals.shape[0] > 2:
            if average:
                # return average recall across classes 
                return np.mean(self._recall_multiclass(X, y))
            # return array of recalls for each class
            return self._recall_multiclass(X, y)

    def f1_score(self, X: np.ndarray, y: np.ndarray):
        """Computes F1 score of model on input data. F1 score is the harmonic mean of precision and recall.

        F1 = 2 * (precision * recall) / (precision + recall)

        for multiclass, automatically averages precision and recall to compute F1 score

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)
            y (np.ndarray): target data of shape (num_samples, )

        Returns:
            f1 (float): F1 score of model
        """
        self._validate_X_y(X, y)

        precision = self.precision(X, y, average=True)
        recall = self.recall(X, y, average=True)

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1