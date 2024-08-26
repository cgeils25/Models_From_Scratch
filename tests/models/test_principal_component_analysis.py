import numpy as np
from models import PCA

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# create redundant features
X_and_redundant_features = [X, X * 2, X * 3, X-1, X-5, X]

# concatenate features with X
X_redundant = np.concatenate(X_and_redundant_features, axis=1)

all_X = [X, X_redundant]

def test_PCA_num_components():
    """Test PCA with num_components argument"""
    for X in all_X:
        pca = PCA(num_components=1)

        pca.fit(X)
        
        assert pca.components.shape == (X.shape[1], 1)

        X_transformed = pca.transform(X)

        assert X_transformed.shape == (X.shape[0], 1)
        
        # make sure fit_transform is the same as calling fit and then transform
        pca2 = PCA(num_components=1)
        
        X_transformed2 = pca2.fit_transform(X)

        assert np.allclose(X_transformed, X_transformed2)

        # make sure inverse_transform recovers X

        X_reconstructed = pca.inverse_transform(X_transformed)

        assert np.allclose(X, X_reconstructed)

def test_PCA_p_variance():
    """Test PCA with p_variance argument"""
    for X in all_X:
        pca = PCA(p_variance=0.95)

        pca.fit(X)
        
        assert pca.components.shape == (X.shape[1], 1)

        X_transformed = pca.transform(X)

        assert X_transformed.shape == (X.shape[0], 1)
        
        # make sure fit_transform is the same as calling fit and then transform
        pca2 = PCA(num_components=1)
        X_transformed2 = pca2.fit_transform(X)

        assert np.allclose(X_transformed, X_transformed2)

        X_reconstructed = pca.inverse_transform(X_transformed)

        assert np.allclose(X, X_reconstructed)
