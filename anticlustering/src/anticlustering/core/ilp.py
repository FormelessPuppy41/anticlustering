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

from .base import AntiCluster, BaseConfig
from ._registry import register_solver
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix


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
        self._model: Optional[_ILPAntiClusterModel] = None

    # --------- Fit interface ------------------------------------------------------------
    def fit(
        self,
        X: Optional[np.ndarray] = None,
        *,
        D: Optional[np.ndarray] = None,
    ) -> "ILPAntiCluster":
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
        self._model = _ILPAntiClusterModel(D, self.cfg)

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
        self._model: Optional[_ILPAntiClusterModel] = None
    
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
            self._set_status("skipped")     # skipped due to size
            self._set_runtime(float("nan")) # no runtime due to size
            return self

        # validate & log ----------------------------------------------------
        self.cfg.validate(N)

        # create preclustering labels and extract forbidden pairs (if needed) -------------------
        forbidden_pairs: Optional[Sequence[Tuple[int, int]]] = None
        if self.cfg.preclustering:
            _LOG.debug("Preclustering enabled; extracting forbidden pairs")
            forbidden_pairs = PreClustering(
                D, 
                n_clusters=self.config.n_clusters
            ).forbidden_pairs
            #TODO: Implement a preclustering step here, either a class or a function.
        
        # extract forbidden pairs (if needed) -------------------
        # build model ------------------------------------------------------
        self._model = _ILPAntiClusterModel(D, self.cfg)

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


# ---------------- PreClustering Class ----------------------------
class PreClustering:
    def __init__(
            self,
            D: np.ndarray,
            *,
            n_clusters: int = 3,
        ):
        self.D                  : np.ndarray                = D
        self.n_clusters         : int                       = n_clusters
        self._forbidden_pairs   : Sequence[Tuple[int, int]]
    
    @property
    def forbidden_pairs(self) -> Sequence[Tuple[int, int]]:
        """
        Extracts forbidden pairs from the dissimilarity matrix D based on the preclustering logic.

        Returns
        -------
        Sequence[Tuple[int, int]]
            A list of pairs (i, j) that should not be in the same group.
        """
        if not hasattr(self, "_forbidden_pairs"):
            self._forbidden_pairs = self._extract_forbidden_pairs()
        return self._forbidden_pairs
    
    def _extract_forbidden_pairs(self) -> Sequence[Tuple[int, int]]:
        """
        Extracts forbidden pairs from the dissimilarity matrix D based on the preclustering logic.

        This method identifies pairs of items that are too dissimilar to be in the same group
        based on the specified number of clusters.

        Returns
        -------
        Sequence[Tuple[int, int]]
            A list of pairs (i, j) that should not be in the same group.
        """
        #TODO: Implement a more sophisticated preclustering step here.
        N = self.D.shape[0]
        threshold = np.partition(self.D.flatten(), N * N // (2 * self.n_clusters))[
            N * N // (2 * self.n_clusters)
        ]
        forbidden_pairs = [
            (i, j) for i in range(N) for j in range(i + 1, N) if self.D[i, j] > threshold
        ]
        return forbidden_pairs
        


# ---------------- ILP Model Implementation --------------- #
class _ILPAntiClusterModel:
    """
    _ILPAntiClusterModel formulates and solves the anticluster editing problem using Pyomo.

    The anticlustering objective is to partition a set of N items into K groups of **equal size**,
    such that within-group dissimilarity is maximized (i.e., groups are internally diverse,
    and thus similar to each other in aggregate characteristics).

    This is formulated as an ILP with binary variables representing pairwise group co-membership,
    and solved using a chosen MIP solver (e.g., CBC, Gurobi, GLPK).

    Parameters
    ----------
    D : np.ndarray
        Symmetric pairwise dissimilarity matrix of shape (N, N).
    K : int
        Number of equally-sized groups to partition the data into.
        Must divide N exactly (N mod K == 0).
    solver : str, optional
        Name of a Pyomo-compatible MILP solver to use (default is "gurobi").

    Attributes
    ----------
    D : np.ndarray
        Original dissimilarity matrix.
    K : int
        Number of groups.
    N : int
        Number of items (inferred from D).
    group_size : int
        Number of elements per group (= N // K).
    solver_name : str
        Solver identifier used for `pyomo.SolverFactory`.
    m : pyo.ConcreteModel
        Pyomo model object representing the ILP.
    """

    def __init__(
            self, 
            D                   : np.ndarray, 
            cfg                 : ILPConfig,
            *,
            forbidden_pairs     : Optional[Sequence[Tuple[int, int]]] = None
        ):
        self.D = D
        self.cfg = cfg
        self.N = D.shape[0]
        self.K = cfg.n_clusters
        self.group_size = self.N // self.K
        self.forbidden_pairs = forbidden_pairs

        self.m = pyo.ConcreteModel()
        self._build_sets()
        self._build_params()
        self._build_vars()
        self._build_objective()
        self._build_constraints()

    # ------------------ Sets ------------------ #
    def _build_sets(self):
        """
        Define index sets used in the ILP model:
        - I: Item indices (0..N-1)
        - PAIRS: All unordered pairs (i, j) with i < j, used for binary co-membership variables
        - TRIPLES: All ordered triplets (i, j, k) with i < j < k, used for enforcing transitivity
        """
        self.m.I        = pyo.Set(
            initialize=range(self.N)
        )
        self.m.PAIRS    = pyo.Set(
            initialize=[(i, j) for i in range(self.N) for j in range(i + 1, self.N)]
        )
        self.m.TRIPLES  = pyo.Set(
            initialize=[(i, j, k) for i, j, k in combinations(range(self.N), 3)]
        )

    @property
    def items(self):
        """
        Set of item indices {0, ..., N-1}.
        Returns
        -------
        pyo.Set
            Pyomo set of item indices.
        """
        return self.m.I

    @property
    def pairs(self):
        """
        Set of item index pairs (i, j) with i < j.
        Used to define binary decision variables indicating co-group membership.
        
        Returns
        -------
        pyo.Set
            Pyomo set of unique item pairs.
        """
        return self.m.PAIRS

    @property
    def triples(self):
        """
        Set of item index triplets (i, j, k) with i < j < k.
        Used to enforce transitivity constraints on co-membership variables.
        
        Returns
        -------
        pyo.Set
            Pyomo set of ordered triplets.
        """
        return self.m.TRIPLES

    # ------------------ Parameters ------------------ #
    def _build_params(self):
        """
        Define the pairwise dissimilarity matrix as a Pyomo parameter.

        This is needed for use in the objective function and should match the original matrix D.
        The matrix is assumed symmetric and stored in full (including redundant entries).
        """
        self.m.D = pyo.Param(
            self.items, 
            self.items,
            initialize  =   {(i, j): self.D[i, j] for i in range(self.N) for j in range(self.N)},
            within      =   pyo.NonNegativeReals
        )

    def get_dissim(self, i, j):
        """
        Accessor for dissimilarity between items i and j.

        Parameters
        ----------
        i : int
            First item index.
        j : int
            Second item index.

        Returns
        -------
        float
            Dissimilarity value D[i, j] from the parameter table.
        """
        return self.m.D[i, j]


    # ------------------ Variables ------------------ #
    def _build_vars(self):
        """
        Define binary decision variables x[i, j] for all item pairs (i < j).

        x[i, j] = 1 if items i and j are assigned to the same anticluster, 0 otherwise.

        These are the core decision variables of the ILP and drive both the objective
        (which aims to maximize within-group dissimilarity) and the transitivity/group-size constraints.

        If `forbidden_pairs` is provided, constraints are added to ensure that
        certain pairs (i, j) cannot be in the same group (i.e., x[i, j] = 0).
        """
        self.m.x = pyo.Var(self.pairs, within=pyo.Binary)

        if self.forbidden_pairs is not None:
            # Add constraints to prevent certain pairs from being in the same group
            for i, j in self.forbidden_pairs:
                if (i, j) in self.pairs or (j, i) in self.pairs:
                    # Ensure that x[i, j] = 0 for forbidden pairs
                    self.m.x[(min(i, j), max(i, j))].fix(0)
                else:
                    _LOG.warning(
                        "Forbidden pair (%d, %d) not in pairs set; skipping.", i, j
                    )

    def x(self, i, j):
        """
        Accessor for the variable x[i, j], with automatic ordering of indices.

        Parameters
        ----------
        i : int
            First item index.
        j : int
            Second item index.

        Returns
        -------
        pyo.Var
            The Pyomo binary variable indicating whether items i and j are in the same group.
        """
        return self.m.x[(min(i, j), max(i, j))]

    # ------------------ Objective ------------------ #
    def _build_objective(self):
        """
        Define the ILP objective function: maximize total within-group dissimilarity.

        The objective sums D[i, j] * x[i, j] over all pairs (i < j), where x[i, j] = 1
        if items i and j are assigned to the same group.

        By maximizing this sum, the model encourages forming groups with large internal
        dissimilarity, thereby increasing between-group similarity.
        """
        def obj_rule(m):
            return sum(self.x(i, j) * self.get_dissim(i, j) for i, j in self.pairs)

        self.m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    def get_objective_value(self):
        """
        Get the current value of the objective function (after solving the model).

        Returns
        -------
        float
            The optimal value of the objective function (total within-group dissimilarity).
        """
        return pyo.value(self.m.OBJ)


    # ------------------ Constraints ------------------ #
    def _build_constraints(self):
        """
        Define and attach all constraints for the ILP model:
        - Transitivity constraints: ensure logical consistency of pairwise group memberships.
        - Group size constraints: enforce equal group sizes (N / K per group).
        """
        self._add_transitivity_constraints()
        self._add_group_size_constraints()

    def _add_transitivity_constraints(self):
        """
        Add transitivity constraints to ensure consistent group membership.

        For every triplet (i, j, k), the following must hold:
        If i is in the same group as j and i is in the same group as k,
        then j must be in the same group as k.

        Enforced using 3 inequalities for each (i, j, k):
            x_ij + x_ik - x_jk ≤ 1
            x_ij - x_ik + x_jk ≤ 1
            -x_ij + x_ik + x_jk ≤ 1
        These prevent cyclic inconsistencies in co-membership logic.
        """
        self.m.transitivity = pyo.ConstraintList()
        for i, j, k in self.triples:
            self.m.transitivity.add(self.x(i, j) + self.x(i, k) - self.x(j, k) <= 1)
            self.m.transitivity.add(self.x(i, j) - self.x(i, k) + self.x(j, k) <= 1)
            self.m.transitivity.add(-self.x(i, j) + self.x(i, k) + self.x(j, k) <= 1)

    def _add_group_size_constraints(self):
        """
        Add group size constraints to ensure all groups are of equal size (N/K).

        For each item i, enforce that it is in the same group as exactly (group_size - 1) others:
            ∑ x_ij (for j ≠ i) = group_size - 1

        This ensures that each group forms a clique of size group_size,
        and all N elements are exactly partitioned into K such groups.
        """
        self.m.group_size = pyo.ConstraintList()
        for i in self.items:
            neighbors = [
                self.x(i, j) if i < j else self.x(j, i)
                for j in self.items if i != j
            ]
            self.m.group_size.add(sum(neighbors) == self.group_size - 1)

    # ------------------------ warm‑start ----------------------------------
    def apply_warm_start(self, labels: np.ndarray) -> None:
        if len(labels) != self.N:
            raise ValueError("Warm‑start labels length mismatch.")
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.m.x[(i + 1, j + 1)].value = 1 if labels[i] == labels[j] else 0

    # ------------------ Settings ------------------ #
    def _set_mip_gap(self, solver, solver_name: str, gap: float):
        _GAP_OPTION = {
            "gurobi": {"MIPGap"},
            "highs":  {"mip_rel_gap"},
            "cbc":    {"ratio"},                       # CBC / Coin-OR
            "cplex":  {"mip tolerances mipgap"},       # note the space!
        }
        # normalise solver name once
        s = solver_name.lower()
        for key, opts in _GAP_OPTION.items():
            if s.startswith(key):
                for opt in opts:
                    solver.options[opt] = gap
                return
        # fall-back: try the most common spelling
        solver.options["MIPGap"] = gap

    # ------------------ Solve ------------------ #
    def solve(self) -> Tuple[np.ndarray, float, Literal['ok', 'timeout', 'error'], float | None]:
        """
        Solve the anticlustering ILP using the specified MILP solver.

        This method compiles the Pyomo model and passes it to a solver via `SolverFactory`.
        It checks for optimal termination and raises an error if the solution is infeasible
        or suboptimal.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints solver output to stdout. Useful for debugging or inspecting solver progress.

        Returns
        -------
        tuple of (np.ndarray, float, Literal['ok', 'timeout', 'error'])
            - group_assignments : np.ndarray of shape (N,)
                Integer group labels (0 to K-1) assigned to each item.
            - objective_value : float
                Optimal value of the objective function (total within-group dissimilarity).
            - status : Literal['ok', 'timeout', 'error']
                Status of the solver after solving:
                - 'ok' if an optimal solution was found,
                - 'timeout' if the solver stopped due to time limit,
                - 'error' if the solver failed to converge or encountered an error.
            - gap : float | None
                The MIP gap if applicable (e.g., for Gurobi, CBC, CPLEX). None if not applicable, however, could also just be None.

        Raises
        ------
        RuntimeError
            If the solver fails to find an optimal solution.
        """
        solver = pyo.SolverFactory(self.cfg.solver_name)

        # pass common options if supported --------------------------------
        if self.cfg.time_limit is not None:
            solver.options["TimeLimit"] = int(self.cfg.time_limit)
        if self.cfg.mip_gap is not None:
            # try both Gurobi/CBC style names; silently ignore if not supported
            try:
                self._set_mip_gap(solver, self.cfg.solver_name, self.cfg.mip_gap)
            except KeyError:
                pass
    
        results = solver.solve(self.m, tee=self.cfg.verbose)
        tc = results.solver.termination_condition
        good_tc = {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        }
        limit_tc = {
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.maxIterations,
            pyo.TerminationCondition.userInterrupt,
        }

        if tc in good_tc:
            status = "ok"
        elif tc in limit_tc:
            # make sure the model actually has values
            if not any(v.value is not None for v in self.m.component_data_objects(pyo.Var, active=True)):
                raise RuntimeError("Solver stopped at time-limit with no incumbent.")
            _LOG.warning(
                "Solver stopped (%s); using best incumbent found so far.", tc
            )
            status = "timeout"
        else:
            _LOG.error("Solver failed to converge: %s", tc)
            status = "error"

        gap = getattr(results.solver, "gap", None)
        assignment, obj_val = self._decode_solution()
        return assignment, obj_val, status, gap


    def _decode_solution(self) -> Tuple[np.ndarray, float]:
        """
        Extracts group assignments from the solved ILP model by analyzing pairwise group co-membership.

        This method:
        1. Builds an undirected adjacency graph where an edge exists between items (i, j)
           if x[i, j] = 1 in the solution (i.e., they are in the same group).
        2. Uses depth-first search to identify connected components (each component corresponds to one group).
        3. Assigns a unique group ID to each component.

        Returns
        -------
        tuple of (np.ndarray, float, Literal['ok', 'timeout', 'error'])
            - assignment : np.ndarray of shape (N,)
                An array of integers (0 to K-1) representing the group index for each item.
            - objective_value : float
                The optimal value of the objective function (i.e., total within-group dissimilarity).

        Notes
        -----
        This decoding assumes that each group forms a clique and that transitivity and group size
        constraints have been enforced. The connected components of the x[i,j]=1 adjacency graph
        correspond directly to groups.
        """
        # build adjacency list
        adj: List[set[int]] = [set() for _ in range(self.N)]
        for i, j in self.pairs:
            if pyo.value(self.x(i, j)) >= 0.5:
                adj[i - 1].add(j - 1)
                adj[j - 1].add(i - 1)

        # DFS to find connected components
        labels = -np.ones(self.N, dtype=int)
        current = 0
        for node in range(self.N):
            if labels[node] >= 0:
                continue
            stack = [node]
            while stack:
                v = stack.pop()
                if labels[v] == -1:
                    labels[v] = current
                    stack.extend(adj[v] - {w for w in stack if labels[w] == -1})
            current += 1

        return labels, float(pyo.value(self.m.OBJ))




__all__ = ["ILPAntiCluster", "ILPConfig"]