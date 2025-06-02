from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import logging
from typing import Optional, Sequence, Tuple, Union, List, Literal
import time

import numpy as np
import pyomo.environ as pyo

from .base import AntiCluster, BaseConfig, Status
from ._registry import register_solver
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix

from ..solvers.online import ModelAntiClusterOnline

from .config import OnlineConfig

_LOG = logging.getLogger(__name__)


from .base import AntiCluster, BaseConfig, Status
from ._registry import register_solver

@register_solver('online')
class OnlineAntiCluster(AntiCluster):
    def __init__(self, config: OnlineConfig):
        super().__init__(config)
        self._model: Optional[ModelAntiClusterOnline]
    

    def fit(
        self, 
        X: Optional[np.ndarray] = None,
        init_labels: Optional[np.ndarray] = None,
        *,
        D: Optional[np.ndarray] = None,
        ):
        if X is None and D is None:
            raise ValueError("Either X or D must be provided.")

        if D is None:
            D = get_dissimilarity_matrix(X)
            _LOG.debug("Dissimilarity matrix computed with shape %s", D.shape)
        else:
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise ValueError("D must be a square distance matrix.")
        N = D.shape[0]

        self._model = ModelAntiClusterOnline()

        # solve ------------------------------------------------------------
        _LOG.info("Starting ILP anticlustering: N=%d, K=%d", N, self.cfg.n_clusters)
        t0 = time.perf_counter()
        labels, score, status, gap = self._model.solve()
        runtime = time.perf_counter() - t0
    
        # set labels and score ----------------------------------------
        self._set_labels(labels)
        self._set_score(score)
        self._set_status(status, gap)
        self._set_runtime(runtime)

        return self
    


