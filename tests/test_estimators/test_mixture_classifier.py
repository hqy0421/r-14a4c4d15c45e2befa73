import numpy as np
import pytest
from scipy.special import logsumexp, softmax
from sklearn_compat.utils.estimator_checks import parametrize_with_checks

from sklego.mixture import BayesianGMMClassifier, GMMClassifier
from tests.conftest import (
    ARRAY_API_CHECK,
    GAUSSIAN_MIXTURE_ARRAY_API_REASON,
    GMM_CLASSIFIER_NAN_REASON,
    expect_failed_checks,
)

EXPECTED_FAILED_CHECKS = expect_failed_checks(
    (ARRAY_API_CHECK, GAUSSIAN_MIXTURE_ARRAY_API_REASON, (GMMClassifier,)),
    ("check_estimators_nan_inf", GMM_CLASSIFIER_NAN_REASON, (GMMClassifier,)),
)


@parametrize_with_checks(
    [GMMClassifier(), BayesianGMMClassifier()],
    expected_failed_checks=EXPECTED_FAILED_CHECKS,
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("clf", [GMMClassifier(max_iter=1000), BayesianGMMClassifier(max_iter=1000)])
def test_obvious_usecase(clf):
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (clf.fit(X, y).predict(X) == y).all()


COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]


@pytest.fixture
def correlated_dataset():
    rng = np.random.RandomState(42)
    X = np.vstack([rng.normal(0, 1, (300, 3)), rng.normal(4, 1.5, (300, 3))])
    # make the features correlated so that dropping them cannot be replaced by filling values
    X[:, 1] = 0.9 * X[:, 0] + 0.3 * rng.normal(size=X.shape[0])
    X[:, 2] = -0.6 * X[:, 0] + 0.5 * rng.normal(size=X.shape[0])
    y = np.repeat([0, 1], 300)
    return X, y


def _brute_force_log_likelihood(gmm, x):
    """Log marginal density of `x` under `gmm`, computed from the public `covariances_` attribute."""
    observed = ~np.isnan(x)
    n_observed = observed.sum()
    probabilities = []
    for mean, covariance, weight in zip(gmm.means_, _component_covariances(gmm), gmm.weights_):
        marginal_mean = mean[observed]
        marginal_covariance = covariance[np.ix_(observed, observed)]
        chol = np.linalg.cholesky(marginal_covariance)
        solved = np.linalg.solve(chol, x[observed] - marginal_mean)
        log_det = 2 * np.sum(np.log(np.diag(chol)))
        probabilities.append(-0.5 * (n_observed * np.log(2 * np.pi) + log_det + solved @ solved) + np.log(weight))
    return logsumexp(probabilities)


def _component_covariances(gmm):
    if gmm.covariance_type == "full":
        return gmm.covariances_
    if gmm.covariance_type == "tied":
        return np.repeat(gmm.covariances_[None, :, :], gmm.n_components, axis=0)
    if gmm.covariance_type == "diag":
        return np.stack([np.diag(covariance) for covariance in gmm.covariances_])
    return np.stack([np.eye(gmm.means_.shape[1]) * variance for variance in gmm.covariances_])


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_complete_rows_match_original_behaviour(correlated_dataset, covariance_type):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, covariance_type=covariance_type, random_state=0).fit(X, y)
    X_test = X[:20].copy()

    log_likelihoods = np.stack([clf.gmms_[c].score_samples(X_test) for c in clf.classes_], axis=1)

    np.testing.assert_allclose(clf.predict_proba(X_test), softmax(log_likelihoods, axis=1))
    np.testing.assert_array_equal(clf.predict(X_test), clf.classes_[log_likelihoods.argmax(axis=1)])


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
@pytest.mark.parametrize(
    "missing",
    [
        [0],
        [2],
        [0, 2],
        [0, 1, 2],
    ],
)
def test_missing_features_use_marginal_likelihood(correlated_dataset, covariance_type, missing):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, covariance_type=covariance_type, random_state=0).fit(X, y)
    x = X[7].copy()
    x[missing] = np.nan

    expected = np.exp(
        [
            _brute_force_log_likelihood(clf.gmms_[cls], x) for cls in clf.classes_
        ]
    )
    expected /= expected.sum()

    np.testing.assert_allclose(clf.predict_proba(x[None, :]), expected[None, :], rtol=1e-10)


def test_missing_features_keep_correlations(correlated_dataset):
    """Scoring the marginal must differ from imputing a feature, which loses the correlation."""
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    x = X[7].copy()
    x[1] = np.nan

    marginal = clf.predict_proba(x[None, :])

    imputed = x.copy()
    imputed[1] = X[:, 1].mean()
    imputed_prob = clf.predict_proba(imputed[None, :])

    assert not np.allclose(marginal, imputed_prob)


def test_all_missing_row_is_uniform(correlated_dataset):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    X_test = np.vstack([np.full((1, X.shape[1]), np.nan), X[:2]])

    probabilities = clf.predict_proba(X_test)

    np.testing.assert_allclose(probabilities[0], np.full(clf.classes_.shape[0], 1 / clf.classes_.shape[0]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_mixed_missing_patterns_are_scored_per_row(correlated_dataset):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    X_test = X[:5].copy()
    X_test[1, 0] = np.nan
    X_test[2, [0, 2]] = np.nan
    X_test[3, :] = np.nan
    # row 4 stays complete

    probabilities = clf.predict_proba(X_test)

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[4], clf.predict_proba(X[4:5])[0])
    np.testing.assert_allclose(probabilities[3], np.full(clf.classes_.shape[0], 1 / clf.classes_.shape[0]))
    for row, missing in [(0, []), (1, [0]), (2, [0, 2])]:
        x = X[row].copy()
        x[missing] = np.nan
        np.testing.assert_allclose(probabilities[row], clf.predict_proba(x[None, :])[0])


def test_predict_is_consistent_with_predict_proba(correlated_dataset):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    X_test = X[:10].copy()
    X_test[0, 0] = np.nan
    X_test[1, :] = np.nan

    probabilities = clf.predict_proba(X_test)

    np.testing.assert_array_equal(clf.predict(X_test), clf.classes_[probabilities.argmax(axis=1)])


def test_single_row_with_missing_feature(correlated_dataset):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    x = X[0].copy()
    x[1] = np.nan

    probabilities = clf.predict_proba(x[None, :])

    assert probabilities.shape == (1, clf.classes_.shape[0])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_predict_does_not_modify_input(correlated_dataset):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    X_test = X[:5].copy()
    X_test[1, 0] = np.nan
    X_test[2, [0, 1]] = np.nan
    original = X_test.copy()

    clf.predict_proba(X_test)
    clf.predict(X_test)

    np.testing.assert_array_equal(X_test, original)


def test_fit_still_rejects_missing_values(correlated_dataset):
    X, y = correlated_dataset
    X_train = X.copy()
    X_train[0, 1] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        GMMClassifier(n_components=2, random_state=0).fit(X_train, y)


def test_predict_still_rejects_infinite_values(correlated_dataset):
    X, y = correlated_dataset
    clf = GMMClassifier(n_components=2, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="infinity"):
        clf.predict_proba(np.array([[np.inf, 0.0, 0.0]]))


def test_refit_is_stable_with_missing_values(correlated_dataset):
    X, y = correlated_dataset
    X_test = X[:5].copy()
    X_test[1, 0] = np.nan
    X_test[2, :] = np.nan

    first = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    second = GMMClassifier(n_components=2, random_state=0).fit(X, y)

    np.testing.assert_allclose(first.predict_proba(X_test), second.predict_proba(X_test))
    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))
