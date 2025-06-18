import numpy as np
from typing import Callable, Literal, Tuple
from ..core._config import ExchangeConfig, Status
from ..metrics.dissimilarity_matrix import (
    variance_objective,
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
            self._obj_fn: Callable = variance_objective
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
        if D is None and self.D is not None:
            D = self.D
        elif D is None:
            D = get_dissimilarity_matrix(X)

        size = N // self.K
        best_labels = None
        best_score = -np.inf
        rng = np.random.default_rng(self.cfg.random_state)

        for restart in range(self.cfg.n_restarts):
            # 1 random equal‐size init
            labels = np.repeat(np.arange(self.K), size)
            rng.shuffle(labels)

            # 2 initial score
            score = (
                self._obj_fn(X, labels)
                if self.objective == 'variance'
                else self._obj_fn(D, labels)
            )

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
                        new_score = (
                            self._obj_fn(X, labels)
                            if self.objective == "variance"
                            else self._obj_fn(D, labels)
                        )
    
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

        final_diversity = diversity_objective(D, best_labels) 

        return best_labels, float(final_diversity), Status.heuristic
