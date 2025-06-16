import time
import numpy as np
from typing import Optional
from .base import AntiCluster
from ._registry import register_solver
from ._config import RandomConfig
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix, diversity_objective
from ..core._config import Status
import logging

_LOG = logging.getLogger(__name__)


@register_solver("random")
class RandomAntiCluster(AntiCluster):
    """
    Random equal‐size assignment baseline.
    """

    def __init__(self, config: RandomConfig):
        super().__init__(config)
        self.cfg = config

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        *,
        D: Optional[np.ndarray] = None
    ) -> "RandomAntiCluster":
        # 1) get or compute D
        if X is None and D is None:
            raise ValueError("Either X or D must be provided")
        if D is None:
            D = get_dissimilarity_matrix(X)
            _LOG.debug("Dissimilarity matrix shape %s", D.shape)
        else:
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise ValueError("D must be a square matrix")
        N = D.shape[0]
        K = self.config.n_clusters
        if N % K != 0:
            raise ValueError(f"Cannot split N={N} into {K} perfectly equal groups")

        #TODO: Should use different random states to avoid identical scores. Right? How to do this.
        rng = np.random.default_rng(self.cfg.random_state)

        # 2) assign evenly and shuffle
        start = time.perf_counter()
        labels = np.repeat(np.arange(K), N // K)
        rng.shuffle(labels)
        runtime = time.perf_counter() - start

        # 3) compute within‐cluster score
        score = diversity_objective(D, labels)
        
        # 4) store and return
        self._set_labels(labels)
        self._set_score(float(score))
        self._set_status(Status.heuristic)
        self._set_runtime(runtime)
        return self
