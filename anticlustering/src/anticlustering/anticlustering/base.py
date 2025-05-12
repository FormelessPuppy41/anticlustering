"""
Abstract Solver interface with scikit-learn compatibility.
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class Solver(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for anticlustering solvers (Strategy pattern).

    Inherits scikit-learn BaseEstimator and TransformerMixin for
    seamless integration into pipelines and grid-search.
    """
    def __init__(self, n_groups: int, max_iter: int = 1, random_state: int = None):
        self.n_groups = n_groups
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.score_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Initialize labels and optimize objective.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self
        """
        n_samples = X.shape[0]
        if n_samples % self.n_groups != 0:
            raise ValueError("n_samples must be divisible by n_groups")
        self.labels_ = self._init_labels(n_samples)
        self.labels_ = self._optimize(X, self.labels_)
        self.score_ = self.objective(X, self.labels_)
        return self

    def transform(self, X: np.ndarray):
        """
        Return the fitted labels.
        """
        if self.labels_ is None:
            raise ValueError("Must fit before transform")
        return self.labels_

    def fit_transform(self, X: np.ndarray, y=None):
        """
        Fit to X, then transform it.
        """
        return self.fit(X).transform(X)

    def score(self, X: np.ndarray, y=None):
        """
        Returns the objective score of the fitted labels on X.
        """
        if self.score_ is None:
            raise ValueError("Must fit before scoring")
        return self.score_

    @abstractmethod
    def objective(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the objective to maximize for a given labeling.
        """
        pass  # implemented by subclasses

    @abstractmethod
    def _optimize(self, X: np.ndarray, initial_labels: np.ndarray) -> np.ndarray:
        """
        Optimize labeling starting from initial_labels and return improved labels.
        """
        pass  # implemented by subclasses

    def _init_labels(self, n_samples: int) -> np.ndarray:
        """
        Create a reproducible equal-sized random partition of 0..n_groups-1.
        """
        rng = np.random.default_rng(self.random_state)
        return rng.permutation(
            np.repeat(np.arange(self.n_groups), n_samples // self.n_groups)
        )
