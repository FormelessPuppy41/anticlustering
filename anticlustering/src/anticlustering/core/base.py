from __future__ import annotations          # <- future-proof typing
from abc import ABC, abstractmethod
import logging
import numpy as np

from ._config import BaseConfig, Status

_LOG = logging.getLogger(__name__)


class AntiCluster(ABC):
    """Common interface for all anticlustering solvers."""

    def __init__(self, config: BaseConfig):
        self.config     : BaseConfig                   = config
        self._labels    : np.ndarray       | None      = None
        self._score     : float            | None      = None
        self._runtime   : float            | None      = None
        self._status    : Status           | None      = None  # e.g. "ok", "timeout", "error"
        self._gap       : float            | None      = None  # e.g. "gap" for ILP solvers

    @abstractmethod
    def fit(self, X: np.ndarray | None = None, *, D: np.ndarray | None = None):
        """
        Compute the partition in-place.  Either *X* **or** a
        pre-computed distance matrix *D* must be supplied.
        """
        ...

    def fit_predict(self, *args, **kwargs) -> np.ndarray:
        self.fit(*args, **kwargs)
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
    
    @property
    def runtime_(self):
        if self._runtime is None:
            raise RuntimeError("Call `.fit()` first!")
        return self._runtime
    
    @property
    def status_(self) -> str:
        """Status of the solver after fitting."""
        if self._status is None:
            raise RuntimeError("Call `.fit()` first!")
        return self._status.value if self._status else None
    
    @property
    def gap_(self) -> float | None:
        """Gap of the solver after fitting (if applicable). Gap may be None."""
        return self._gap
    
    # ------------ internal helpers (for subclasses) ------------------- #
    def _set_labels(self, labels: np.ndarray, *, allow_unassigned: bool = False):
        """Store a 1-D vector of length *N* with cluster indices 0…K-1."""
        _LOG.info("Labels set to %s", labels)

        if labels.ndim != 1:
            raise ValueError("labels must be a 1-D array")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("labels must contain integers")
        
        low, high = labels.min(initial=0), labels.max(initial=-1)
        if allow_unassigned: 
            if low < -1 or high >= self.config.n_clusters:
                raise ValueError("labels outside the expected 0…K-1 range or -1 sentinel range (unassigned)."
                                 f"Values range: {low}...{high}")
        else:
            if low < 0 or high >= self.config.n_clusters:
                raise ValueError("labels outside the expected 0…K-1 range. "
                                 f"Values range: {low}...{high}")
        
        self._labels = labels

    def _set_score(self, score: float):
        if not isinstance(score, (int, float)):
            raise TypeError("score must be numeric")
        self._score = float(score)

    def _set_runtime(self, runtime: float):
        if not isinstance(runtime, (int, float)):
            raise TypeError("runtime must be numeric")
        if runtime < 0:
            self._runtime = np.nan
            #raise ValueError("runtime must be non-negative")
        self._runtime = float(runtime)

    def _set_status(
            self, 
            status  : Status | str,
            gap     : float | None = None
            ):
        if not isinstance(status, Status):
            status = Status.from_string(status)
        self._status = status
        
        if gap is not None:
            if not isinstance(gap, (int, float)):
                raise TypeError("gap must be numeric")
            self._gap = float(gap)
        else:
            self._gap = None

    # ------------------------------------------------------------------ #
    # nice string representation                                         #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        lab = "unfitted" if self._labels is None else "fitted"
        return f"{cls}(K={self.config.n_clusters}, status={lab})"
    


__all__ = [
    "BaseConfig",
    "AntiCluster",
]
