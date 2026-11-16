import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data

_LOG_2PI = np.log(2.0 * np.pi)


def _log_component_marginal_prob(X, gmm, observed):
    """Log marginal density of every mixture component on a subset of observed features.

    The features that are not in `observed` are integrated out of each fitted component, keeping the
    correlations between the observed features intact. The component weights are included, so the result
    can be reduced with `logsumexp` to obtain the log likelihood of the mixture.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_observed)
        The observed feature values.
    gmm : GaussianMixture
        A fitted mixture model.
    observed : np.ndarray of shape (n_observed,)
        Indices of the features present in `X`.

    Returns
    -------
    np.ndarray of shape (n_samples, n_components)
        The log probability of the observed features under each component.
    """
    n_samples, n_observed = X.shape
    log_weights = np.log(gmm.weights_)
    if n_observed == 0:
        # Integrating out every feature leaves the mixture weights, which sum to one.
        return np.broadcast_to(log_weights, (n_samples, log_weights.shape[0]))

    means = gmm.means_[:, observed]
    covariance_type = gmm.covariance_type
    diff = X[:, None, :] - means[None, :, :]

    if covariance_type in ("full", "tied"):
        if covariance_type == "full":
            covariance = gmm.covariances_[:, observed][:, :, observed]
        else:
            covariance = gmm.covariances_[np.ix_(observed, observed)]
        chol = np.linalg.cholesky(covariance)
        solved = np.linalg.solve(chol, diff.transpose(1, 2, 0))
        quadratic = np.sum(solved**2, axis=1).T
        log_det = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    elif covariance_type in ("diag", "spherical"):
        if covariance_type == "diag":
            variances = gmm.covariances_[:, observed]
        else:
            variances = np.repeat(gmm.covariances_[:, None], n_observed, axis=1)
        quadratic = np.sum(diff**2 / variances[None, :, :], axis=2)
        log_det = np.sum(np.log(variances), axis=1)
    else:  # pragma: no cover - guarded by sklearn's GaussianMixture
        raise ValueError(f"Unknown covariance type: {covariance_type!r}")

    return -0.5 * (n_observed * _LOG_2PI + log_det + quadratic) + log_weights


def _log_mixture_prob(gmm, X):
    """Log likelihood of `X` under a fitted mixture, treating `NaN` as a missing feature.

    Rows without missing values are scored with the fitted mixture itself, so their result is identical to
    `GaussianMixture.score_samples`. For rows with missing values the likelihood is evaluated on the marginal
    distribution over the features that were actually observed.

    Parameters
    ----------
    gmm : GaussianMixture
        A fitted mixture model.
    X : np.ndarray of shape (n_samples, n_features)
        The data to score, possibly containing `NaN` values.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The log likelihood of each row.
    """
    scores = np.empty(X.shape[0])
    missing = np.isnan(X)
    complete = ~missing.any(axis=1)
    if complete.any():
        scores[complete] = gmm.score_samples(X[complete])

    incomplete = np.flatnonzero(~complete)
    if incomplete.size:
        X_incomplete = X[incomplete]
        patterns, pattern_index = np.unique(np.isnan(X_incomplete), axis=0, return_inverse=True)
        for idx, pattern in enumerate(patterns):
            rows = pattern_index == idx
            observed = np.flatnonzero(~pattern)
            component_scores = _log_component_marginal_prob(X_incomplete[rows][:, observed], gmm, observed)
            scores[incomplete[rows]] = logsumexp(component_scores, axis=1)

    return scores


class GMMClassifier(ClassifierMixin, BaseEstimator):
    """The `GMMClassifier` trains a Gaussian Mixture Model for each class in `y` on a dataset `X`. Once a density is
    trained for each class we can evaluate the likelihood scores to see which class is more likely.

    All parameters of the model are an exact copy of the parameters in scikit-learn.

    !!! note
        All the parameters are an exact copy of those of
        [sklearn.mixture.GaussianMixture]( https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

    !!! info
        Training data must be complete, but `predict` and `predict_proba` accept `np.nan` to mark features
        that were not observed. Each row is then scored with the marginal likelihood of its observed features
        (per mixture component, so correlations between observed features are preserved) instead of imputing
        values. A row without any observed feature gets a uniform probability over the classes.

    Attributes
    ----------
    gmms_ : dict[int, GaussianMixture]
        A dictionary of Gaussian Mixture Models, one for each class.
    classes_ : np.ndarray of shape (n_classes,)
        The classes seen during `fit`.

    Examples
    --------

    ```python
    import numpy as np
    from sklego.mixture import GMMClassifier

    # Generate dataset
    np.random.seed(1)
    group0 = np.random.normal(0, 3, (1000, 2))
    group1 = np.random.normal(2.5, 2, (500, 2))
    data = np.vstack([group0, group1])

    y = np.hstack([np.zeros((group0.shape[0],), dtype=int), np.ones((group1.shape[0],), dtype=int)])

    # Create and fit the GMMClassifier model
    gmm = GMMClassifier(n_components=2, random_state=1)
    gmm.fit(data, y)

    # Classify the train dataset into two clusters (n_components=2)
    labels = gmm.predict(data)

    # Classify a new point into one of two clusters
    p = np.array([[1.5, 0.5]])
    p_prob = gmm.predict_proba(p) # predict the probabilities p belongs to each cluster
    print(f'Probability point p belongs to group1 is {p_prob[0,0]:.2f}')
    ### Probability point p belongs to group1 is 0.41
    print(f'Probability point p belongs to group2 is {p_prob[0,1]:.2f}')
    ### Probability point p belongs to group2 is 0.59

    print(f'It is more probable that point p belongs to group{np.argmax(p_prob)}')
    ### It is more probable that point p belongs to group1
    ```
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMClassifier":
        """Fit the `GMMClassifier` model using `X`, `y` as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : GMMClassifier
            The fitted estimator.
        """
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            mixture = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                tol=self.tol,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                init_params=self.init_params,
                weights_init=self.weights_init,
                means_init=self.means_init,
                precisions_init=self.precisions_init,
                random_state=self.random_state,
                warm_start=self.warm_start,
                verbose=self.verbose,
                verbose_interval=self.verbose_interval,
            )
            self.gmms_[c] = mixture.fit(subset_x, subset_y)

        self.n_iter_ = sum(mixture.n_iter_ for mixture in self.gmms_.values())

        return self

    def predict(self, X):
        """Predict labels for `X` using fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict. Features that were not observed can be marked with `np.nan`; the
            label is then based on the marginal likelihood of the observed features.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        check_is_fitted(self, ["gmms_", "classes_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan", reset=False)

        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        """Predict probabilities for `X` using fitted estimator.

        Features that were not observed can be marked with `np.nan`. The probabilities are then computed
        from the marginal distribution of the observed features, integrating out the missing ones for each
        component of every class mixture while keeping the correlations between the observed features. Rows
        without any observed feature yield a uniform distribution over the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        check_is_fitted(self, ["gmms_", "classes_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan", reset=False)

        res = np.zeros((X.shape[0], self.classes_.shape[0]))
        for idx, c in enumerate(self.classes_):
            res[:, idx] = _log_mixture_prob(self.gmms_[c], X)
        return softmax(res, axis=1)
