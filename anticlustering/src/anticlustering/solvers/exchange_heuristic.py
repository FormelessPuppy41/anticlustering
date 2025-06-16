import numpy as np
from typing import Callable, Literal, Tuple
from ..core._config import ExchangeConfig, Status
from ..metrics.dissimilarity_matrix import (
    sum_squared_to_centroids,
    diversity_objective,
    get_dissimilarity_matrix,
)


Objective = Literal["variance", "diversity"]


class ExchangeHeuristic:
    """
    Generic exchange heuristic for anticlustering, parameterized by objective.

    Algorithm (same for both objectives):
      1) Random equal‐size initialization.
      2) Repeatedly scan all cross‐cluster swaps (i,j) and compute Δ objective.
      3) Execute the single swap with the largest positive Δ.
      4) Stop when no swap yields Δ > tolerance.
      5) Repeat for n_restarts and keep the best solution.
    """

    def __init__(
        self,
        K: int,
        config: ExchangeConfig,
        objective: Objective = "diversity",
        tol: float = 1e-8,
        D: np.ndarray = None
    ):
        """
        Parameters
        ----------
        K : int
            Number of clusters (must exactly divide N).
        config : KMeansConfig
            - random_state : int, RNG seed
            - n_restarts   : int, number of random initializations
        objective : {"variance", "diversity"}
            "variance"  → maximize sum_squared_to_centroids(X, labels)
            "diversity"  → maximize within_group_distance(D, labels)
        tol : float
            Minimum positive gain to accept a swap.
        D : np.ndarray, shape (N, N), optional
            Pairwise squared‐Euclidean dissimilarities. Required if
            objective="diversity". If None, it will be computed from X.
        """
        self.K = K
        self.cfg = config
        self.objective = objective
        self.tol = tol
        self.D = D

        if objective == "variance":
            # k-means anticlustering objective
            self._obj_fn: Callable = sum_squared_to_centroids
        elif objective == "diversity":
            # anticluster‐editing (pairwise) objective
            self._obj_fn = diversity_objective
        else:
            raise ValueError(f"Unknown objective: {objective}")

    def solve(self, X: np.ndarray, D: np.ndarray = None) -> Tuple[np.ndarray, float, Status]:
        """
        Run the exchange heuristic under the chosen objective.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix.
        D : np.ndarray, shape (N, N), optional
            Pairwise squared‐Euclidean dissimilarities. Required if
            objective="diversity". If None, it will be computed.

        Returns
        -------
        labels : np.ndarray, shape (N,)
            Equal‐sized cluster assignments [0..K-1].
        score : float
            The maximized objective value.
        status : Status
            Always Status.heuristic.

        Raises
        ------
        ValueError
            If N % K != 0, or if D is needed but not provided.
        """
        N, _ = X.shape
        if N % self.K != 0:
            raise ValueError(f"N={N} not divisible by K={self.K}")

        # Prepare D if needed
        if self.objective == "diversity":
            if D is None and self.D is not None:
                D = self.D
            elif D is None:
                D = get_dissimilarity_matrix(X)
        else:
            D = None  # unused for variance objective

        size = N // self.K
        best_labels = None
        best_score = -np.inf
        rng = np.random.default_rng(self.cfg.random_state)

        for restart in range(self.cfg.n_restarts):
            # 1 random equal‐size init
            labels = np.repeat(np.arange(self.K), size)
            rng.shuffle(labels)

            # 2 initial score
            if self.objective == "variance":
                score = self._obj_fn(X, labels)  # sum_squared_to_centroids
            else:
                score = self._obj_fn(D, labels)  # within_group_distance

            # 3 exchange loop
            while True:
                best_delta = 0.0
                best_i = best_j = -1

                for i in range(N - 1):
                    for j in range(i + 1, N):
                        if labels[i] == labels[j]:
                            continue
                        # test swap
                        labels[i], labels[j] = labels[j], labels[i]
                        if self.objective == "variance":
                            new_score = self._obj_fn(X, labels)
                        else:
                            new_score = self._obj_fn(D, labels)
                        delta = new_score - score
                        # revert
                        labels[i], labels[j] = labels[j], labels[i]

                        if delta > best_delta:
                            best_delta = delta
                            best_i, best_j = i, j

                if best_delta > self.tol:
                    labels[best_i], labels[best_j] = labels[best_j], labels[best_i]
                    score += best_delta
                else:
                    break

            # 4 record best
            if score > best_score:
                best_score = score
                best_labels = labels.copy()

        return best_labels, float(best_score), Status.heuristic


""""

import numpy as np
from scipy.spatial.distance import squareform, pdist


from ..core._config import ExchangeConfig, Status

from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix

import logging


class ExchangeHeuristic:
    ""
    A class that implements the exchange heuristic for anticlustering.
    This heuristic iteratively exchanges elements between clusters to improve the objective function.
    ""

    def __init__(
            self, 
            D : np.ndarray,
            K : int,
            config: ExchangeConfig
            ):
        self.cfg = config
        self.D = D

        if self.cfg.random_state is not None:
            np.random.seed(self.cfg.random_state)
        else:
            np.random.seed(42)

        if self.cfg.verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    #  API
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    def solve(self, X: np.ndarray) -> np.ndarray:
        ""
        Main entry: returns cluster labels (shape (n,))
        ""
        if self.D is None:
            if X is None:
                raise ValueError("Either D or X must be provided.")
            self.D = get_dissimilarity_matrix(X)
            
        rng = np.random.default_rng(self.cfg.random_state)
        n, _ = X.shape

        # 2. balanced random initial assignment
        labels = self._balanced_initialisation(n, rng)

        # 3. pre-compute intra-cluster sums
        intra_sum = self._compute_all_intra(labels, self.D)

        # 4. build neighbour lists if needed
        if self.cfg.k_neighbours:
            self._build_neighbour_lists(self.D)

        # 4. local-search by swaps
        current_obj = intra_sum.sum()
        best_obj = current_obj
        no_imp = 0
        status = Status.solved

        for sweep in range(1, self.cfg.max_sweeps + 1):
            best_delta = 0.0
            best_pair = None

            for i in range(n):
                cand = (
                    self._neigh[i]
                    if self.cfg.k_neighbours
                    else range(i+1, n)
                )
                for j in cand:
                    if j <= i or labels[i] == labels[j]:
                        continue
                    delta = self._swap_gain(i, j, labels[i], labels[j], labels, self.D)
                    if delta > best_delta:
                        best_delta = delta
                        best_pair = (i, j)

            if best_delta > 1e-12:
                i, j = best_pair
                self._apply_swap(i, j, labels[i], labels[j], labels, intra_sum, self.D)
                current_obj += best_delta
            else:
                break
            improved = False

            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = labels[i], labels[j]
                    if ci == cj:
                        continue

                    delta = self._swap_gain(i, j, ci, cj, labels, self.D)
                    if delta > 0:             # accept best-improving
                        # update bookkeeping
                        self._apply_swap(
                            i, j, ci, cj, labels, intra_sum, self.D
                        )
                        current_obj += delta
                        improved = True

            if self.cfg.verbose:
                logging.info(
                    f"Sweep {sweep:3d} | "
                    f"Objective = {current_obj:,.4f} "
                    f"{'↑' if improved else ''}"
                )

            if improved:
                best_obj = current_obj
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.cfg.patience:
                    status = Status.stopped
                    if self.cfg.verbose:
                        logging.info("Early-stopping: no improvement.")
                    break
        
        return labels, best_obj, status

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    #  Helpers
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    def _balanced_initialisation(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ""
        Returns an array of length n with exactly ⌊n/k⌋ or ⌈n/k⌉ points per cluster.
        ""
        k = self.cfg.n_clusters
        base_size, remainder = divmod(n, k)
        sizes = np.array([base_size + 1] * remainder + [base_size] * (k - remainder))
        labels = np.hstack([np.full(sz, c, dtype=int) for c, sz in enumerate(sizes)])
        rng.shuffle(labels)
        return labels

    def _compute_all_intra(self, labels: np.ndarray, D: np.ndarray) -> np.ndarray:
        ""
        Intra-cluster sums for every cluster (upper-triangle, no double counting).
        ""
        k = self.cfg.n_clusters
        intra_sum = np.zeros(k, dtype=float)
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            intra = D[np.ix_(idx, idx)]
            intra_sum[c] = np.sum(np.triu(intra, 1))
        return intra_sum
    
    def _build_neighbour_lists(self, D):
        # called once before the first sweep
        idx = np.argsort(D, axis=1)[:, 1:self.cfg.k_neighbours + 1]
        self._neigh = [row.tolist() for row in idx]


    def _swap_gain(
        self,
        i: int,
        j: int,
        ci: int,
        cj: int,
        labels: np.ndarray,
        D: np.ndarray,
    ) -> float:
        ""
        Δ objective if we put i→cj, j→ci (they are currently in ci, cj).
        ""
        # Members currently in ci / cj (excluding i/j)
        members_ci = np.where(labels == ci)[0]
        members_cj = np.where(labels == cj)[0]

        # remove i/j themselves for "old" contribution
        members_ci = members_ci[members_ci != i]
        members_cj = members_cj[members_cj != j]

        # OLD sums: i with its own cluster + j with its cluster
        old_sum = D[i, members_ci].sum() + D[j, members_cj].sum()

        # NEW sums after swap
        new_sum = D[i, members_cj].sum() + D[j, members_ci].sum()

        return new_sum - old_sum

    def _apply_swap(
        self,
        i: int,
        j: int,
        ci: int,
        cj: int,
        labels: np.ndarray,
        intra_sum: np.ndarray,
        D: np.ndarray,
    ):
        ""
        Execute swap i↔j and update intra_sum in **O(n)**.
        ""
        # contribution of i & j to their current clusters
        members_ci = np.where(labels == ci)[0]
        members_cj = np.where(labels == cj)[0]

        # remove i and j themselves
        members_ci = members_ci[members_ci != i]
        members_cj = members_cj[members_cj != j]

        # old contributions
        old_ci = D[i, members_ci].sum()
        old_cj = D[j, members_cj].sum()

        # new contributions
        new_ci = D[j, members_ci].sum()
        new_cj = D[i, members_cj].sum()

        # update sums
        intra_sum[ci] += (new_ci - old_ci)
        intra_sum[cj] += (new_cj - old_cj)

        # finally swap labels
        labels[i], labels[j] = cj, ci
        """