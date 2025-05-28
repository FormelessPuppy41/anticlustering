from __future__ import annotations

"""Improved ILPAntiCluster solver.

This refactor introduces a typed, logger‑based, and configurable interface that
follows modern Python best‑practices while keeping the original 
mathematical formulation (pair‑wise transitivity ILP).  Users instantiate the
solver with an :class:`ILPConfig` and call :meth:`fit`.

Example
-------
>>> cfg = ILPConfig(n_clusters=3, time_limit=300, mip_gap=0.01)
>>> solver = ILPAntiCluster(cfg)
>>> labels = solver.fit_predict(X)  # X is (N, p) NumPy array
"""

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

from ..solvers.edge_ilp import ModelPreClusterILP, ModelAntiClusterILP


_LOG = logging.getLogger(__name__)

# ----------- Configuration dataclass ----------------------------------------------------

@dataclass(slots=True)
class ILPConfig(BaseConfig):
    """
    Tunable parameters for :class:`ILPAntiCluster`. Inherits from :class:`BaseConfig`.

    Notes
    -----
    * ``time_limit`` is interpreted in *seconds* and forwarded to the underlying
      MIP solver if supported (Gurobi, CBC, CPLEX, HiGHS).
    * ``warm_start`` may come from a fast heuristic such as the exchange
      algorithm; it should be a 1‑D integer array of length *N* containing group
      labels *(0 … K‑1)*.
    """

    n_clusters      : int
    solver_name     : str                   = "gurobi"
    max_n           : Optional[int]         = None  # max number of items (N) to solve
    time_limit      : Optional[int]         = None
    mip_gap         : Optional[float]       = None  # absolute or relative depending on solver
    warm_start      : Optional[np.ndarray]  = None
    preclustering   : bool                  = False  # whether to use preclustering
    verbose         : bool                  = False  # print solver output

    # future extensions ------------------------------------------------------
    categories  : Optional[np.ndarray]  = None  # 1‑D categorical strata

    # guard rails ------------------------------------------------------------
    max_items   : int                   = 30  # warn & abort above this threshold

    def validate(self, n_items: int) -> None:
        if self.n_clusters <= 1:
            raise ValueError("n_clusters must be >= 2")
        if n_items % self.n_clusters:
            raise ValueError(
                f"Number of items {n_items} not divisible by K={self.n_clusters}."
            )
        if n_items > self.max_items:
            raise ValueError(
                f"Problem too large for the exact ILP (N={n_items} > {self.max_items})."
            )



# ---------- Public solver class -----------------------------------------------------------------

@register_solver("ilp")
class ILPAntiCluster(AntiCluster):
    """Exact anticlustering via Integer Linear Programming.

    Parameters
    ----------
    config : ILPConfig
        Solver and model hyper‑parameters.
    random_state : int | None, optional
        Only used when *warm_start* is **None** and a random feasible start is
        generated internally (rarely needed).
    """

    def __init__(self, config: ILPConfig):
        super().__init__(config=config)
        self.cfg = config
        self._model: Optional[ModelAntiClusterILP] = None

    # --------- Fit interface ------------------------------------------------------------
    def fit(
        self,
        X: Optional[np.ndarray] = None,
        *,
        D: Optional[np.ndarray] = None,
    ) -> "ModelAntiClusterILP":
        """Compute anticlusters from data matrix *X* or a pre‑computed *D*.

        Notes
        -----
        * ``X`` is expected to be feature matrix (N, p); the Euclidean distance
          matrix is computed on the fly.
        * Either ``X`` or ``D`` **must** be supplied.  If both are given, ``D``
          takes precedence.
        """
        if X is None and D is None:
            raise ValueError("Either X or D must be provided.")

        if D is None:
            D = get_dissimilarity_matrix(X)
            _LOG.debug("Dissimilarity matrix computed with shape %s", D.shape)
        else:
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise ValueError("D must be a square distance matrix.")
        N = D.shape[0]

        # Early exit if N exceeds max_n -------------------------------
        if self.cfg.max_n is not None and N > self.cfg.max_n:
            _LOG.warning(
                "ILP skipped: N=%d exceeds max_n=%d. - returning empty solution", 
                N, self.cfg.max_n,
            )
            self._set_labels(
                np.full(shape=N, fill_value=-1, dtype=int), 
                allow_unassigned=True
            )                               # empty solution
            self._set_score(float("nan"))   # no score due to size
            self._set_status("skipped")     # skipped due to size
            self._set_runtime(float("nan")) # no runtime due to size
            return self

        # validate & log ----------------------------------------------------
        self.cfg.validate(N)
    
        # build model ------------------------------------------------------
        self._model = ModelAntiClusterILP(D, self.cfg.n_clusters)

        # warm‑start (optional) -------------------------------------------
        if self.cfg.warm_start is not None:
            _LOG.debug("Applying warm‑start solution")
            self._model.apply_warm_start(self.cfg.warm_start)

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

    # alias required by *AntiCluster* --------------------------------------
    predict = AntiCluster.fit_predict


@register_solver("ilp/precluster")
class PreClusterILPAntiCluster(ILPAntiCluster):
    """ILPAntiCluster with preclustering enabled.

    This is a convenience subclass that sets the `preclustering` flag to True
    and inherits all other parameters from ILPAntiCluster.
    """
    def __init__(self, config: ILPConfig):
        super().__init__(config)
        self.cfg = config
        if not self.cfg.preclustering:
            _LOG.warning(
                "PreClusterILPAntiCluster should be used with preclustering enabled. "
                "Setting preclustering=True."
                )
            self.cfg.preclustering = True
        self._model: Optional[ModelAntiClusterILP] = None
    
    def fit(
            self, 
            X: Optional[np.ndarray] = None,
            *, 
            D: Optional[np.ndarray] = None
        ) -> "PreClusterILPAntiCluster":
        """Compute anticlusters from data matrix *X* or a pre‑computed *D*.

        Notes
        -----
        * ``X`` is expected to be feature matrix (N, p); the Euclidean distance
          matrix is computed on the fly.
        * Either ``X`` or ``D`` **must** be supplied.  If both are given, ``D``
          takes precedence.
        """
        if X is None and D is None:
            raise ValueError("Either X or D must be provided.")

        if D is None:
            D = get_dissimilarity_matrix(X)
            _LOG.debug("Dissimilarity matrix computed with shape %s", D.shape)
        else:
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise ValueError("D must be a square distance matrix.")
        N = D.shape[0]

        # Early exit if N exceeds max_n -------------------------------
        if self.cfg.max_n is not None and N > self.cfg.max_n:
            _LOG.warning(
                "ILP skipped: N=%d exceeds max_n=%d. - returning empty solution", 
                N, self.cfg.max_n,
            )
            self._set_labels(
                np.full(shape=N, fill_value=-1, dtype=int), 
                allow_unassigned=True
            )                               # empty solution
            self._set_score(float("nan"))   # no score due to size
            self._set_status(Status.skipped)     # skipped due to size
            self._set_runtime(float("nan")) # no runtime due to size
            return self

        # validate & log ----------------------------------------------------
        self.cfg.validate(N)

        # create preclustering labels and extract forbidden pairs (if needed) -------------------
        preclust = ModelPreClusterILP(D=D, K=self.cfg.n_clusters)
        preclust.solve()
        if preclust.status_ != "optimal":
            _LOG.error(
                "Preclustering failed to find an optimal solution: %s", preclust.status_
            )
            raise RuntimeError("Preclustering failed; cannot proceed with ILP.")

        
        anticlust = ModelAntiClusterILP(
            D=D,
            K=self.cfg.n_clusters,
            forbidden_pairs=preclust.extract_group_pairs()
        )
        
        # build model ------------------------------------------------------
        self._model = anticlust

        # warm‑start (optional) -------------------------------------------
        if self.cfg.warm_start is not None:
            _LOG.debug("Applying warm‑start solution")
            self._model.apply_warm_start(self.cfg.warm_start)

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


__all__ = ["ILPAntiCluster", "PreClusterILPAntiCluster", "ILPConfig"]