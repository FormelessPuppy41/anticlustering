# edge_ilp_gurobi.py
"""
Exact anticlustering / pre-clustering ILP formulated directly in Gurobi.

Drop-in replacement for the original Pyomo version (edge_ilp.py).
Author: <your-name>, 2025-06-09
"""
from __future__ import annotations
import logging
from functools import lru_cache
from itertools import combinations
from typing import List, Sequence, Tuple, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# keep using your existing helper enum / config
from ..core._config import Status, ILPConfig     # unchanged

_LOG = logging.getLogger(__name__)


class _ModelEdgeILPBase:
    # --------------------------- construction ----------------------------
    def __init__(
        self,
        D:          np.ndarray,           # (N×N) dissimilarity matrix
        K:          int,
        config:     ILPConfig,
        *,
        sense:      int,                  # GRB.MAXIMIZE | GRB.MINIMIZE
        target_degree: int,
        forbidden_pairs: Sequence[Tuple[int, int]] | None = None,
    ):
        if D.shape[0] != D.shape[1]:
            raise ValueError("D must be square")

        self.D               = D.astype(float, copy=False)
        self.N               = D.shape[0]
        self.K               = K
        self.sense           = sense
        self.target_degree   = target_degree
        self.forbidden_pairs = {(min(i, j), max(i, j))
                                for (i, j) in forbidden_pairs or []}
        self.cfg             = config

        if self.N % K != 0:
            raise ValueError(f"N={self.N} not divisible by K={K}")

        self._build_model()

    # --------------------------- helpers ---------------------------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _pairs(N: int) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

    @staticmethod
    @lru_cache(maxsize=None)
    def _triples(N: int) -> List[Tuple[int, int, int]]:
        return [(i, j, k)
                for i in range(N)
                for j in range(i + 1, N)
                for k in range(j + 1, N)]

    # --------------------------- model -----------------------------------
    def _build_model(self) -> None:
        m = gp.Model("edge_ilp")
        m.Params.OutputFlag = int(self.cfg.verbose)

        # decision vars x_{ij} (i<j)
        x = m.addVars(
            self._pairs(self.N),
            vtype=GRB.BINARY,
            name="x",
        )

        # fix forbidden pairs to 0
        for (i, j) in self.forbidden_pairs:
            x[i, j].ub = 0

        # objective
        obj = gp.quicksum(self.D[i, j] * x[i, j]
                          for (i, j) in self._pairs(self.N))
        m.setObjective(obj, self.sense)

        # ---------- triangle (transitivity) constraints ----------
        for (i, j, k) in self._triples(self.N):
            m.addConstr(x[i, j] + x[i, k] - x[j, k] <= 1)
            m.addConstr(x[i, j] - x[i, k] + x[j, k] <= 1)
            m.addConstr(-x[i, j] + x[i, k] + x[j, k] <= 1)

        # ---------- equal group-size (degree) constraints ----------
        for i in range(self.N):
            m.addConstr(gp.quicksum(
                x[min(i, j), max(i, j)]
                for j in range(self.N) if j != i
            ) == self.target_degree, name=f"deg[{i}]")

        self._m = m
        self._x = x              # keep a handle for decoding

    # ------------------------- warm start -------------------------
    def apply_warm_start(self, labels: np.ndarray) -> None:
        if len(labels) != self.N:
            raise ValueError("Warm-start labels length mismatch")
        for (i, j), var in self._x.items():
            var.start = 1.0 if labels[i] == labels[j] else 0.0

    # --------------------------- solve ----------------------------
    def _set_params(self) -> None:
        if self.cfg.time_limit is not None:
            self._m.Params.TimeLimit = int(self.cfg.time_limit)
        if self.cfg.mip_gap is not None:
            self._m.Params.MIPGap = float(self.cfg.mip_gap)

    def solve(self) -> Tuple[np.ndarray, float, Status | str, float | None]:
        self._set_params()
        self._m.optimize()

        status_map = {
            GRB.OPTIMAL:  "optimal",
            GRB.TIME_LIMIT: "timeout",
        }
        st_code   = self._m.Status
        status    = status_map.get(st_code, "error")

        gap = None
        if self._m.Params.MIPGap > 0:
            gap = self._m.MIPGap

        labels, obj_val = self._decode_solution()

        # public attributes (mirror old API)
        self.labels_   = labels
        self.variable_ = self._x
        self.status_   = status
        self.score_    = obj_val
        self.runtime_  = self._m.Runtime
        self.gap_      = gap
        return labels, obj_val, status, gap, self.runtime_

    # ----------------------- decode x-matrix ----------------------
    def _decode_solution(self) -> Tuple[np.ndarray, float]:
        # build adjacency list of x=1 edges
        adj = [set() for _ in range(self.N)]
        for (i, j), var in self._x.items():
            if var.X >= 0.5:
                adj[i].add(j)
                adj[j].add(i)

        labels = -np.ones(self.N, dtype=int)
        cid = 0
        for v in range(self.N):
            if labels[v] >= 0:
                continue
            stack = [v]
            while stack:
                u = stack.pop()
                if labels[u] == -1:
                    labels[u] = cid
                    stack.extend(adj[u])
            cid += 1

        return labels, self._m.ObjVal
    
    # ------------------------------------------------------------------
    # Pickling support: drop un-picklable Gurobi handles
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Return the state minus the un-picklable Gurobi objects."""
        state = self.__dict__.copy()
        for attr in ("_m", "_x", "variable_"):   # add/remove as needed
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        """Restore state. Optimiser handles stay None (not reloadable)."""
        self.__dict__.update(state)
        # Keep placeholders so that attribute access still works
        self._m = None
        self._x = None
        self.variable_ = None


# ------------------------------------------------------------------
# concrete subclasses (interface unchanged)
# ------------------------------------------------------------------
class ModelAntiClusterILP(_ModelEdgeILPBase):
    """Exact **max-diversity** anticlustering (cluster-editing formulation)."""
    def __init__(
        self, D: np.ndarray, K: int, config: ILPConfig,
        *, forbidden_pairs: Sequence[Tuple[int, int]] | None = None,
    ):
        group_size = D.shape[0] // K
        super().__init__(
            D, K, config,
            sense=GRB.MAXIMIZE,
            target_degree=group_size - 1,
            forbidden_pairs=forbidden_pairs,
        )


class ModelPreClusterILP(_ModelEdgeILPBase):
    """Exact **min-diversity** pre-clustering step."""
    def __init__(self, D: np.ndarray, K: int, config: ILPConfig):
        super().__init__(
            D, K, config,
            sense=GRB.MINIMIZE,
            target_degree=K - 1,
            forbidden_pairs=None,
        )

    def extract_group_pairs(self) -> Sequence[Tuple[int, int]]:
        return [(i, j) for (i, j), var in self._x.items() if var.X >= 0.5]
