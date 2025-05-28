from __future__ import annotations

from itertools import combinations
from typing import List, Tuple, Sequence, Optional, Literal
import logging

import numpy as np
import pyomo.environ as pyo

from ..core.base import Status

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: shared implementation
# ---------------------------------------------------------------------------

class _ModelEdgeILPBase:
    """
    Protected base class that implements the edge‑formulation ILP.

    Parameters
    ----------
    D : np.ndarray (N × N)
        Symmetric non‑negative **dissimilarity** matrix.
    K : int
        Number of (anti)clusters.
    sense : pyomo.*
        ``pyo.maximize`` for the anticlustering problem, ``pyo.minimize``
        for the pre‑clustering (min‑diversity) problem.
    target_degree : int
        Each vertex must have exactly this many *intra*‑clique edges.
    forbidden_pairs : Sequence[tuple[int,int]] | None, default ``None``
        Pairs for which *x[i,j] = 0* is enforced (pre‑clustering hint).
    """

    # public (after solve) ---------------------------------------------------
    labels_     : Optional[np.ndarray]              = None
    variable_   : Optional[pyo.Var]                 = None
    status_     : Optional[Status]                  = None
    score_      : Optional[float]                   = None
    runtime_    : Optional[float]                   = None
    gap_        : Optional[float]                   = None

    # construction -----------------------------------------------------------
    def __init__(
        self,
        D: np.ndarray,
        K: int,
        *,
        sense,
        target_degree       : int,
        forbidden_pairs     : Sequence[Tuple[int, int]] | None = None,
    ) -> None:
        if D.shape[0] != D.shape[1]:
            raise ValueError("D must be a square matrix.")
        
        self.D                  = D.astype(float, copy=False)
        self.N                  = D.shape[0]
        self.K                  = K
        self.sense              = sense
        self.target_degree      = target_degree
        self.forbidden_pairs    = {(min(i,j), max(i,j)) for i,j in forbidden_pairs or []}

        if self.N % K != 0:
            raise ValueError(
                f"Number of items N={self.N} must be divisible by K={K}."
            )

        self._build_model()

    # model builder ----------------------------------------------------------
    def _build_model(self) -> None:
        self.m = pyo.ConcreteModel(name="edge_ilp")
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
        self.m.TRIPLES = pyo.Set(
            dimen=3, initialize=lambda *_:(
                (i,j,k) 
                for i in range(self.N)
                for j in range(i+1, self.N)
                for k in range(j+1, self.N)
            )
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
            initialize=lambda _m,i,j: float(self.D[i,j]),
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

        """
        def _init_x(m, i, j):
            """
            Initialize binary variable x[i, j] for pair (i, j) if it is forbidden.
            If (i, j) is in the forbidden pairs, return 0 to enforce that they are **not** in the same group.
            """
            return 0.0 if (min(i, j), max(i, j)) in self.forbidden_pairs else None
        
        def _fix_forbidden_pairs():
            """
            Fix the variable x[i, j] to 0 if (i, j) is in the forbidden pairs.
            This is used to enforce that certain pairs are not allowed to be in the same group.
            """
            for (i, j) in self.forbidden_pairs:
                self.x(i, j).fix(0)
            
        
        self.m.x = pyo.Var(
            self.pairs, 
            within          =   pyo.Binary, 
            initialize      =   _init_x
        )

        #_fix_forbidden_pairs()


    def x(self, i, j) -> pyo.Var:
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
        Define the ILP objective function: either minimize or maximize total within-group dissimilarity (based on self.sense).

        The objective sums D[i, j] * x[i, j] over all pairs (i < j), where x[i, j] = 1
        if items i and j are assigned to the same group.

        By maximizing this sum, the model encourages forming groups with large internal
        dissimilarity, thereby increasing **between**-group similarity.

        By minimizing, it encourages forming groups with small internal dissimilarity, 
        thereby increasing **within**-group homogeneity.
        """
        def obj_rule(m):
            return sum(self.x(i, j) * self.get_dissim(i, j) for i, j in self.pairs)

        self.m.OBJ = pyo.Objective(rule=obj_rule, sense=self.sense)

    def get_objective_value(self) -> float:
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
        Add group size constraints to ensure all groups are equal in size. Specifically, 
        based on the target_degree (N/K - 1 or group_size - 1), we enforce:
        
        For each item i, the sum of its connections to other items in the same group must equal
        the target degree.
            ∑ x_ij (for j ≠ i) = target_degree

        This ensures that each group forms a clique of size target_degree + 1,
        and all N elements are exactly partitioned into K such groups.
        """
        
        def size_rule(m, i) -> pyo.Expression:
            """
            Helper function to define the group size constraint for item i.
            It sums the binary variables x[i, j] for all j ≠ i and sets it equal to target_degree.
            """
            return sum(
                self.x(i,j) 
                for j in self.items 
                if i != j
            ) == self.target_degree
        
        self.m.group_size = pyo.Constraint(
            self.items,
            rule=size_rule
        )

    # ------------------------ warm‑start ----------------------------------
    def apply_warm_start(self, labels: np.ndarray) -> None:
        if len(labels) != self.N:
            raise ValueError("Warm‑start labels length mismatch.")
        for i, j in self.pairs:
            self.m.x[(i, j)].value = int(labels[i] == labels[j])

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
    def solve(self) -> Tuple[
            np.ndarray, 
            float, 
            Status | str, 
            float | None
        ]:
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
        tuple of (np.ndarray, float, Status | str, float | None)
            - group_assignments : np.ndarray of shape (N,)
                Integer group labels (0 to K-1) assigned to each item.
            - objective_value : float
                Optimal value of the objective function (total within-group dissimilarity).
            - status : Status | str
                Status of the solver after solving:
                - 'optimal' if an optimal solution was found,
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
            status = Status.optimal
        elif tc in limit_tc:
            # make sure the model actually has values
            if not any(v.value is not None for v in self.m.component_data_objects(pyo.Var, active=True)):
                raise RuntimeError("Solver stopped at time-limit with no incumbent.")
            _LOG.warning(
                "Solver stopped (%s); using best incumbent found so far.", tc
            )
            status = Status.timeout
        else:
            _LOG.error("Solver failed to converge: %s", tc)
            status = Status.error

        gap = getattr(results.solver, "gap", None)
        assignment, obj_val = self._decode_solution()
        
        # set the labels and other attributes
        self.labels_ = assignment
        self.variable_ = self.m.x
        self.status_ = status
        self.score_ = obj_val
        self.runtime_ = results.solver.time
        self.gap_ = gap
        
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
                adj[i].add(j)
                adj[j].add(i)

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


# ---------------------------------------------------------------------------
# Public subclasses
# ---------------------------------------------------------------------------

class ModelAntiClusterILP(_ModelEdgeILPBase):
    """Exact **max‑diversity** anticlustering ILP.

    Solves the cluster‑editing formulation (Eq. 11 in the paper) for a
    balanced design of *K* groups × *q = N/K* items.
    """

    def __init__(
        self,
        D: np.ndarray,
        K: int,
        *,
        forbidden_pairs: Sequence[Tuple[int, int]] | None = None,
    ) -> None:
        group_size = D.shape[0] // K
        super().__init__(
            D,
            K,
            sense=pyo.maximize,
            target_degree=group_size - 1,
            forbidden_pairs=forbidden_pairs,
        )


class ModelPreClusterILP(_ModelEdgeILPBase):
    """Exact **min‑diversity** ILP used as *pre‑clustering* step.

    Partitions the data into *r = K* very homogeneous mini‑clusters
    (pairs if K=2, triplets if K=3, …).  The resulting intra‑pairs are
    returned via :meth:`_EdgeILPBase.forbidden_pairs` and can be used to
    prune the search space of :class:`AntiClusterILP`.
    """

    def __init__(self, D: np.ndarray, K: int) -> None:
        super().__init__(
            D,
            K,
            sense=pyo.minimize,
            target_degree=K - 1,
            forbidden_pairs=None,
        )

    def extract_group_pairs(self) -> Sequence[Tuple[int, int]]:
        """
        Extracts the pairs of items that are in the same group from the solved model.

        Returns
        -------
        list of tuple[int, int]
            List of pairs (i, j) where items i and j are in the same group.
        """
        return [
            (i, j)
            for i, j in self.pairs
            if pyo.value(self.x(i, j)) >= 0.5
        ]