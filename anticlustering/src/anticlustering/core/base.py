
from abc import ABC, abstractmethod
import numpy as np

class AntiCluster(ABC):
    """Common interface for all anticlustering solvers."""

    def __init__(self, n_clusters: int, *, random_state=None, **kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._labels = None
        self._score = None

    @abstractmethod
    def fit(self, X: np.ndarray, y=None):
        """Compute the anticlustering partition in-place."""
        ...

    def fit_predict(self, X: np.ndarray, y=None):
        self.fit(X, y)
        return self.labels_

    # ____________ Properties for easy access ____________
    @property
    def labels_(self):
        if self._labels is None:
            raise RuntimeError("Call `.fit()` first!")
        return self._labels

    @property
    def score_(self):
        if self._score is None:
            raise RuntimeError("Call `.fit()` first!")
        return self._score
    
    # ____________ Setters for internal use ____________
    def _set_labels(self, labels: np.ndarray):
        """Set the labels after fitting."""
        if len(labels) != self.n_clusters:
            raise ValueError(f"Expected {self.n_clusters} labels, got {len(labels)}.")
        self._labels = labels
        
    def _set_score(self, score: float):
        """Set the score after fitting."""
        if not isinstance(score, (int, float)):
            raise ValueError("Score must be a numeric value.")
        self._score = score