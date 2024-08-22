import numpy as np
import warnings

class PCA:
    def __init__(self, num_components: int = None, p_variance: float = None):
        """Principal Component Analysis (PCA) for dimensionality reduction

        Args:
            num_components (int, optional): number of principle components to keep. Defaults to None.
            p_variance (float, optional): minimum variance captured by components. Must be in range [0, 1]. Defaults to None.

        Raises:
            ValueError: if pass in neither num_components nor p_variance or both
            ValueError: if p_variance is not in range [0, 1]
        """
        # p_variance will be minimum variance captured by components
        if sum(arg == None for arg in (num_components, p_variance)) != 1:
            raise ValueError('must pass in one (and only one) of num_components, p_variance')
        
        if num_components is not None and num_components < 1:
            raise ValueError('num_components must be greater than 0')

        if p_variance is not None and (p_variance > 1 or p_variance < 0):
            raise ValueError('p_variance must be in range [0, 1]')

        self.num_components = num_components
        self.p_variance = p_variance

        if self.p_variance == 1:
            warnings.warn('p_variance is 1, PCA is trivial and will not reduce dimensionality')    

        self.components=None
        self.mean = None

    def _validate_X(self, X):
        # check X type, shape, and for NaN or inf values
        if not isinstance(X, np.ndarray):
          raise TypeError('X must be a numpy array')
        if X.ndim != 2:
          raise ValueError(f'X must be a 2D array of shape (num_samples, num_features), instead got {X.shape}')
        if np.any(np.isnan(X)):
            raise ValueError('X contains NaN values')
        if np.any(np.isinf(X)):
            raise ValueError('X contains inf values')

    def fit(self, X):   
        """Fit PCA model to input data

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)

        Returns:
            None
        """   
        self._validate_X(X)
        
        # get feature means
        X_mean = X.mean(axis=0)

        self.mean = X_mean

        # get covariance matrix for features
        num_samples = X.shape[0] # aka n
        cov_mat = ((X-X_mean).T @ (X-X_mean)) / num_samples

        # get eigenvectors and eigenvalues of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        # sort both by eigenvalues (decreasing)
        sorted_eigenval_idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[sorted_eigenval_idx]
        eigenvectors = eigenvectors[sorted_eigenval_idx]

        if self.num_components:
            self.components = eigenvectors[:, :self.num_components]
            
        elif self.p_variance:
            eigenvalues_norm = eigenvalues / eigenvalues.sum()

            assert np.isclose(eigenvalues_norm.sum(), 1), 'eigenvalues not normalized correctly'

            eigenvalues_norm_cumsum = np.cumsum(eigenvalues_norm)

            assert np.isclose(eigenvalues_norm_cumsum[-1], 1), "cumulative sum didn't work: last element in cumsum of normalized vector should be 1"

            # get the set of eigenvalues with the smallest total variance captured which is greater than p_variance

            greater_than_p_variance_idx = (eigenvalues_norm_cumsum > self.p_variance).nonzero()[0]

            idx_gt_p_variance = np.argmin(eigenvalues_norm_cumsum[greater_than_p_variance_idx])

            idx = greater_than_p_variance_idx[idx_gt_p_variance]

            assert eigenvalues_norm_cumsum[idx] > self.p_variance, 'component cutoff idx not chosen correctly: variance captured less than p_variance'
            assert eigenvalues_norm[:idx+1].sum() > self.p_variance, 'component cutoff idx not chosen correctly: variance captured less than p_variance'

            self.components = eigenvectors[:, :idx+1] # +1 so that the eigenvector at the bound is included

        # components are eigenvectors and should be orthogonal
        assert np.isclose(self.components.T @ self.components, np.eye(self.components.shape[1])).all(), 'components not orthogonal'

    def transform(self, X):
        """Transforms input data into principal component space

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)

        Raises:
            ValueError: if pca not yet fitted

        Returns:
            X_transformed (np.ndarray): transformed data of shape (num_samples, num_components)
        """
        if self.components is None:
            raise ValueError('pca not yet fitted')
            
        self._validate_X(X)

        # input validation; doesn't make sense to have it in _validate_X because of inverse_transform
        if self.components is not None:
            if X.shape[1] != self.components.shape[0]:
                raise ValueError(f'mismatch between number of features in X ({X.shape[1]}) and number of components ({self.components.shape[0]})')
        
        # center X and project onto principal components
        X_transformed = (X - self.mean) @ self.components

        return X_transformed

    def fit_transform(self, X):
        """Fit PCA model and transform input data

        Args:
            X (np.ndarray): input data of shape (num_samples, num_features)

        Returns:
            X_transformed (np.ndarray): transformed data of shape (num_samples, num_components)
        """
        self.fit(X)

        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Transforms principal component space back to original space

        Args:
            X_transformed (np.ndarray): transformed data of shape (num_samples, num_components)

        Raises:
            ValueError: if pca not yet fitted

        Returns:
            X_original (np.ndarray): original data of shape (num_samples, num_features)
        """
        if self.components is None:
            raise ValueError('pca not yet fitted')
        
        self._validate_X(X_transformed)
            
        # project back onto original space
        X_original = X_transformed @ self.components.T + self.mean
        # we can do this because the components are orthogonal, so components @ components.T = I (identity matrix)

        return X_original
