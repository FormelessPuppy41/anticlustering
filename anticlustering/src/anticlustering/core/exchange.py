"""
Exchange-based heuristic as a separate component (Composition over Inheritance).
"""
import numpy as np

from .base import AntiCluster
from ._registry import register_solver

from .config import ExchangeConfig
from ..solvers.exchange_heuristic import ExchangeHeuristic
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix

from typing import Optional, Dict, Any
import logging
import time


_LOG = logging.getLogger(__name__)


@register_solver('exchange')
class ExchangeAntiCluster(AntiCluster):


    def __init__(
            self, 
            config : ExchangeConfig
        ):
        super().__init__(config)
        self.cfg = config
        self._model: Optional[ExchangeHeuristic] = None
    
    def fit(
            self, 
            X: Optional[np.ndarray] = None,
            *,
            D: Optional[np.ndarray] = None
        ) -> "ExchangeAntiCluster":
        if X is None and D is None:
            raise ValueError("Either X or D must be provided.")

        if D is None:
            D = get_dissimilarity_matrix(X)
            _LOG.debug("Dissimilarity matrix computed with shape %s", D.shape)
        else:
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise ValueError("D must be a square distance matrix.")
        N = D.shape[0]


        # validate config
        #self.cfg.validate(n_items=N)

        # build model --------------------------------------------------------
        self._model = ExchangeHeuristic(D=D, K=self.cfg.n_clusters,config=self.cfg)

        # solve ------------------------------------------------------------
        _LOG.info("Starting ILP anticlustering: N=%d, K=%d", N, self.cfg.n_clusters)
        t0 = time.perf_counter()
        labels, score, status = self._model.solve(X)
        runtime = time.perf_counter() - t0

        # set labels and score
        self._set_labels(labels)
        self._set_score(score)
        self._set_status(status)
        self._set_runtime(runtime)
        return self