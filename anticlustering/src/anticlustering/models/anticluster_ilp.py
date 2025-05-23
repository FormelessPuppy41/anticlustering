import numpy as np
import pyomo.environ as pyo
from itertools import combinations


class AnticlusterILP:
    """
    AnticlusterILP formulates and solves the anticluster editing problem using Pyomo.

    The anticlustering objective is to partition a set of N items into K groups of equal size,
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

    def __init__(self, D: np.ndarray, K: int, solver: str = "gurobi"):
        assert D.shape[0] == D.shape[1], "D must be a square matrix"
        assert np.allclose(D, D.T), "D must be symmetric"
        self.D = D
        self.K = K
        self.N = D.shape[0]
        assert self.N % K == 0, "N must be divisible by K"

        self.group_size = self.N // K
        self.solver_name = solver
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
        self.m.I = pyo.Set(initialize=range(self.N))
        self.m.PAIRS = pyo.Set(
            initialize=[(i, j) for i in range(self.N) for j in range(i + 1, self.N)]
        )
        self.m.TRIPLES = pyo.Set(
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
            self.items, self.items,
            initialize={(i, j): self.D[i, j] for i in range(self.N) for j in range(self.N)},
            within=pyo.NonNegativeReals,
            mutable=False
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
        self.m.x = pyo.Var(self.pairs, within=pyo.Binary)

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
        self.m.transitivity = pyo.ConstraintList()
        self.m.group_size = pyo.ConstraintList()

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
        for i in self.items:
            neighbors = [
                self.x(i, j) if i < j else self.x(j, i)
                for j in self.items if i != j
            ]
            self.m.group_size.add(sum(neighbors) == self.group_size - 1)


        # ------------------ Solve ------------------ #
    def solve(self, verbose=False):
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
        tuple of (np.ndarray, float)
            - group_assignments : np.ndarray of shape (N,)
                Integer group labels (0 to K-1) assigned to each item.
            - objective_value : float
                Optimal value of the objective function (total within-group dissimilarity).

        Raises
        ------
        RuntimeError
            If the solver fails to find an optimal solution.
        """
        solver = pyo.SolverFactory(self.solver_name)
        results = solver.solve(self.m, tee=verbose)

        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("Solver did not find optimal solution.")

        return self._extract_assignments()


    def _extract_assignments(self):
        """
        Extracts group assignments from the solved ILP model by analyzing pairwise group co-membership.

        This method:
        1. Builds an undirected adjacency graph where an edge exists between items (i, j)
           if x[i, j] = 1 in the solution (i.e., they are in the same group).
        2. Uses depth-first search to identify connected components (each component corresponds to one group).
        3. Assigns a unique group ID to each component.

        Returns
        -------
        tuple of (np.ndarray, float)
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
        # Build adjacency graph based on x[i, j] = 1
        adj = {i: set() for i in self.items}
        for (i, j) in self.pairs:
            if pyo.value(self.x(i, j)) >= 0.5:
                adj[i].add(j)
                adj[j].add(i)

        # Identify connected components using DFS
        visited = set()
        groups = []
        for i in range(self.N):
            if i not in visited:
                group = set()
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        group.add(node)
                        stack.extend(adj[node] - visited)
                groups.append(group)

        # Convert group sets to assignment array
        assignment = np.empty(self.N, dtype=int)
        for group_id, group in enumerate(groups):
            for idx in group:
                assignment[idx] = group_id

        return assignment, self.get_objective_value()
