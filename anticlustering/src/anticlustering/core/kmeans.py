import numpy as np

from .base import AntiCluster
from ._registry import register_solver

from ._config import KMeansConfig
from ..solvers.kmeans_heuristic import KMeansHeuristic
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix

from typing import Optional, Dict, Any
import logging
import time



_LOG = logging.getLogger(__name__)


@register_solver('kmeans')
class KMeansAntiCluster(AntiCluster):
    """Perfect‐matching based anticlustering (only for K=2)."""

    def __init__(self, config: KMeansConfig):
        super().__init__(config)
        self.cfg = config
        self._model: Optional[KMeansHeuristic] = None
    
    def fit(
            self,
            X: Optional[np.ndarray] = None,
            *,
            D: Optional[np.ndarray] = None
        ) -> "KMeansAntiCluster":
        
        # assume self._compute_dissimilarity_matrix() returns an (N×N) array
        if X is None and D is None:
            raise ValueError("Either X or D must be provided.")
        if D is None:
            D = get_dissimilarity_matrix(X)
            _LOG.debug("Dissimilarity matrix computed with shape %s", D.shape)
        else:
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise ValueError("D must be a square distance matrix.")
        N = D.shape[0]

        self._model = KMeansHeuristic(D=D, K=self.config.n_clusters, config=self.config)

        _LOG.info("Starting KMeans anticlustering: N=%d, K=%d", N, self.config.n_clusters)
        t0 = time.perf_counter()
        labels, score, status = self._model.solve(X, D=D)
        runtime = time.perf_counter() - t0

        self._set_labels(labels)
        self._set_score(score)
        self._set_status(status)
        self._set_runtime(runtime)
        return self
