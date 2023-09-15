import numpy as np
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression


class CorrectedPriorsModel(BaseEstimator):
    def __init__(self, model, **kwargs):
        """
        A sklearn model with corrected priors. To use when the model is fit on data that does not contain the
        true prior.
        """
        self.model = model(**kwargs)
        self.fake_prior_odds = None
        self.true_prior_odds = None

    def fit(self, X, y, **kwargs):
        if 'fake_prior_odds' in kwargs.keys() and 'true_prior_odds' in kwargs.keys():
            self.fake_prior_odds = kwargs['fake_prior_odds']
            self.true_prior_odds = kwargs['true_prior_odds']
        else:
            raise ValueError(
                "Cannot fit CorrectedPriorsModel without fake_prior_odds and true_prior_odds provided")

        return self.model.fit(X, y)

    def predict_proba(self, X):
        if self.fake_prior_odds is None or self.true_prior_odds is None:
            raise ValueError(
                "Cannot predict probabilities for CorrectedPriorsModel; priors not defined")

        # some sklearn models always return probabilities; for some we need to say so explicitly
        if isinstance(self.model, IsotonicRegression):
            p = self.model.predict(X)
        else:
            p = self.model.predict_proba(X)[:, 1]

        post_odds = p / (1 - p) * self.true_prior_odds / self.fake_prior_odds
        post_probs = post_odds / (1 + post_odds)

        return np.vstack([1 - post_probs, post_probs]).T
