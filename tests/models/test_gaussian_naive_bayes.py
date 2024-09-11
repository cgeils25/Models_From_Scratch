from models import GaussianNaiveBayesClassifer
import numpy as np

def test_fit_binary():
    # create toy binary classification data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)
    
    # check means
    assert np.all(model.means[0] == X[y == 0].mean(axis=0)) and np.all(model.means[1] == X[y == 1].mean(axis=0)), 'feature means for each target not calculated correctly'
    
    # check standard deviations
    assert np.all(model.stds[0] == X[y == 0].std(axis=0)) and np.all(model.stds[1] == X[y == 1].std(axis=0)), 'feature standard deviations for each target not calculated correctly'
    
    # check unique y values extracted correctly
    assert np.all(model.y_vals == np.unique(y)), 'unique target (y) values not stored correctly'
    
    # check prior probabilities aka proportion of data corresponding to each target
    assert np.all(model.prior_probabilities == np.array([0.5, 0.5])), 'prior probabilities not calculated correctly'

def test_fit_multiclass():
    # create toy multiclass classification data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 1, 2, 0, 1, 2])

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # check means
    assert np.all(model.means[0] == X[y == 0].mean(axis=0)) and \
    np.all(model.means[1] == X[y == 1].mean(axis=0)) and \
    np.all(model.means[2] == X[y == 2].mean(axis=0)), 'feature means for each target not calculated correctly'

    # check standard deviations
    assert np.all(model.stds[0] == X[y == 0].std(axis=0)) and \
    np.all(model.stds[1] == X[y == 1].std(axis=0)) and \
    np.all(model.stds[2] == X[y == 2].std(axis=0)), 'feature standard deviations for each target not calculated correctly'

    # check unique y values extracted correctly
    assert np.all(model.y_vals == np.unique(y)), 'unique target (y) values not stored correctly'

    # check prior probabilities aka proportion of data corresponding to each target
    assert np.all(model.prior_probabilities == np.array([1/3, 1/3, 1/3])), 'prior probabilities not calculated correctly'

def test_calculate_posterior_probability_one_class():
    # create toy single class data drawn from normal distribution
    np.random.seed(0)
    X = np.random.normal(0, 1, 1000).reshape(-1, 1)
    dummy_y = np.zeros(shape=1000)

    model = GaussianNaiveBayesClassifer()

    model.fit(X, dummy_y)

    posterior_probabilities = model.calculate_posterior_probability(X)

    # posterior probabilities should be 1 for all samples because there is only one class
    assert np.all(posterior_probabilities == np.ones(shape=(1000, 1))), 'posterior probabilities not calculated correctly for one class'

def test_calculate_posterior_probability_binary():
    # create toy binary classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1], size=1000)

    # shift samples of class 1 by 100. Because we're moving it 100 standard deviations away from class 0 model should be able to classify it easily
    X[y==1] += 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    posterior_probabilities = model.calculate_posterior_probability(X)

    # check that posterior probabilities are higher for the correct class
    assert np.all(posterior_probabilities[y == 0][:, 0] > posterior_probabilities[y == 0][:, 1]) \
    and np.all(posterior_probabilities[y == 1][:, 1] > posterior_probabilities[y == 1][:, 0]), 'posterior probabilities not calculated correctly for binary classification'


def test_call_one_class():
    # create toy single class data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    dummy_y = np.zeros(shape=1000)

    model = GaussianNaiveBayesClassifer()

    model.fit(X, dummy_y)

    # predict
    y_hat = model(X)

    # predictions should be the same as the dummy y values
    assert np.all(y_hat == dummy_y), 'predictions not calculated correctly for one class'

def test_call_binary():
    # create toy binary classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1], size=1000)

    # shift samples of class 1 by 100. Because we're moving it 100 standard deviations away from class 0 model should be able to classify it easily
    X[y==1] += 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # predict
    y_hat = model(X)

    # predictions should be the same as the true y values due to extreme separation between classes
    assert np.all(y_hat == y), 'predictions not calculated correctly for binary classification'

def test_call_multiclass():
    # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # predict
    y_hat = model(X)

    # predictions should be the same as the true y values due to extreme separation between classes
    assert np.all(y_hat == y), 'predictions not calculated correctly for binary classification'

def test_accuracy_one_class():
    # create toy single class data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    dummy_y = np.zeros(shape=1000)

    model = GaussianNaiveBayesClassifer()

    model.fit(X, dummy_y)

    # predict
    y_hat = model(X)

    # calculate accuracy
    accuracy = model.accuracy(X, dummy_y)

    # ensure accuracy calculated correctly
    assert accuracy == np.mean(y_hat == dummy_y), 'accuracy not calculated correctly for one class'

def test_accuracy_binary():
    # create toy binary classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1], size=1000)

    # shift samples of class 1 by 100. Because we're moving it 100 standard deviations away from class 0 model should be able to classify it easily
    X[y==1] += 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # predict
    y_hat = model(X)

    # calculate accuracy
    accuracy = model.accuracy(X, y)

    # ensure accuracy calculated correctly
    assert accuracy == np.mean(y_hat == y), 'accuracy not calculated correctly for binary classification'

def test_accuracy_multiclass():
    # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # predict
    y_hat = model(X)

    # calculate accuracy
    accuracy = model.accuracy(X, y)

    # ensure accuracy calculated correctly
    assert accuracy == np.mean(y_hat == y), 'accuracy not calculated correctly for binary classification'

def test_model_multiclass_multiple_features():
    # create toy multiclass classification data with each feature drawn from normal distribution
    rng = np.random.default_rng(seed = 0)

    num_features = 10
    num_samples = 1000

    X = np.empty(shape=(num_samples, num_features))

    # create one feature at a time and sub into empty X
    for feature_idx in range(num_features):
        X[:, feature_idx] = rng.normal(0, 1, num_samples)

    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # predict
    y_hat = model(X)

    # calculate accuracy
    accuracy = model.accuracy(X, y)

    # ensure accuracy calculated correctly
    assert np.all(y_hat == y), 'predictions not calculated correctly for multiclass classification'
    assert accuracy == np.mean(y_hat == y), 'accuracy not calculated correctly for binary classification'

def test_precision_one_class():
    # create toy single class data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    dummy_y = np.zeros(shape=1000)

    model = GaussianNaiveBayesClassifer()

    model.fit(X, dummy_y)

    # calculate precision
    precision = model.precision(X, dummy_y)

    # precision should 1 because there is only one class
    assert precision == 1, 'precision not calculated correctly for one class'

def test_precision_binary():
    # create toy binary classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1], size=1000)

    # shift samples of class 1 by 100. Because we're moving it 100 standard deviations away from class 0 model should be able to classify it easily
    X[y==1] += 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate precision
    precision = model.precision(X, y)

    # precision should be 1 because classes are perfectly separated
    assert precision == 1, 'precision not calculated correctly for binary classification'

def test_precision_multiclass_no_average():
    # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate precision, don't average precision for each class
    precision = model.precision(X, y, average = False)

    # precision should be 1 because classes are perfectly separated
    assert np.all(precision == np.array([1, 1, 1])), 'precision not calculated correctly for multiclass classification (no average)'

def test_precision_multiclass_average():
        # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate precision, don't average precision for each class
    precision = model.precision(X, y, average = True)

    # precision should be 1 because classes are perfectly separated
    assert precision == 1, 'precision not calculated correctly for multiclass classification (with average)'

def test_recall_one_class():
    # create toy single class data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    dummy_y = np.zeros(shape=1000)

    model = GaussianNaiveBayesClassifer()

    model.fit(X, dummy_y)

    # calculate recall
    recall = model.recall(X, dummy_y)

    # recall should 1 because there is only one class
    assert recall == 1, 'recall not calculated correctly for one class'

def test_recall_binary():
    # create toy binary classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1], size=1000)

    # shift samples of class 1 by 100. Because we're moving it 100 standard deviations away from class 0 model should be able to classify it easily
    X[y==1] += 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate recall
    recall = model.recall(X, y)

    # recall should be 1 because classes are perfectly separated
    assert recall == 1, 'recall not calculated correctly for binary classification'

def test_recall_multiclass_no_average():
    # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate recall, don't average recall for each class
    recall = model.recall(X, y, average = False)

    # recall should be 1 because classes are perfectly separated
    assert np.all(recall == np.array([1, 1, 1])), 'recall not calculated correctly for multiclass classification (no average)'

def test_recall_multiclass_average():
    # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate recall, don't average recall for each class
    recall = model.recall(X, y, average = True)

    # recall should be 1 because classes are perfectly separated
    assert recall == 1, 'recall not calculated correctly for multiclass classification (no average)'

def test_f1_score_one_class():
    # create toy single class data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    dummy_y = np.zeros(shape=1000)

    model = GaussianNaiveBayesClassifer()

    model.fit(X, dummy_y)

    # calculate f1 score
    f1_score = model.f1_score(X, dummy_y)

    precision = model.precision(X, dummy_y)
    recall = model.recall(X, dummy_y)

    # f1 score should 1 because there is only one class
    assert f1_score == 1 and f1_score == 2 * (precision * recall) / (precision + recall), 'f1 score not calculated correctly for one class'

def test_f1_score_binary():
    # create toy binary classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1], size=1000)

    # shift samples of class 1 by 100. Because we're moving it 100 standard deviations away from class 0 model should be able to classify it easily
    X[y==1] += 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate f1 score
    f1_score = model.f1_score(X, y)

    precision = model.precision(X, y)
    recall = model.recall(X, y)

    # f1 score should be 1 because classes are perfectly separated
    assert f1_score == 1 and f1_score == 2 * (precision * recall) / (precision + recall), 'f1 score not calculated correctly for binary classification'

def test_f1_score_multiclass_no_average():
    # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate f1 score, don't average f1 score for each class
    f1_score = model.f1_score(X, y)

    precision = model.precision(X, y, average = False)
    recall = model.recall(X, y, average = False)

    # f1 score should be 1 because classes are perfectly separated
    assert np.all(f1_score == 2 * (precision * recall) / (precision + recall)), 'f1 score not calculated correctly for multiclass classification (no average)'

def test_f1_score_multiclass_average():
    # create toy multiclass classification data drawn from normal distribution
    rng = np.random.default_rng(seed = 0)
    X = rng.normal(0, 1, 1000).reshape(-1, 1)
    y = rng.choice([0, 1, 2], size=1000)

    # shift samples of class 1 by 100 and samples of class 2 by -100. Because we're moving them 100 standard deviations away from class 0 model should be able to classify them easily
    X[y==1] += 100
    X[y==2] -= 100

    model = GaussianNaiveBayesClassifer()

    model.fit(X, y)

    # calculate f1 score, don't average f1 score for each class
    f1_score = model.f1_score(X, y)

    precision = model.precision(X, y, average = True)
    recall = model.recall(X, y, average = True)

    # f1 score should be 1 because classes are perfectly separated
    assert f1_score == 2 * (precision * recall) / (precision + recall), 'f1 score not calculated correctly for multiclass classification (no average)'
