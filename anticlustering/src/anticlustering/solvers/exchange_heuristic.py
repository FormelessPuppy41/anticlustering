import numpy as np
from typing import Callable, Literal, Tuple, Optional
from ..core.offline._config import ExchangeConfig, Status
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
      1) Random equal‑size initialization.
      2) Repeatedly scan all cross‑cluster swaps (i,j) and compute Δ objective.
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
        config : ExchangeConfig
            - random_state : int, RNG seed
            - n_restarts   : int, number of random initializations
        objective : {"variance", "diversity"}
            Objective to optimize.
        tol : float
            Minimum positive gain to accept a swap.
        D : np.ndarray, optional
            Precomputed dissimilarity matrix (for diversity).
        """
        self.K = K
        self.cfg = config
        self.objective = objective.lower()
        self.tol = tol
        self.D = D

        if self.objective == "variance":
            self._obj_fn: Callable = variance_objective
        elif self.objective == "diversity":
            self._obj_fn = diversity_objective
        else:
            raise ValueError(f"Unknown objective: {objective}")

    def solve(
        self,
        X: np.ndarray,
        D: np.ndarray = None
    ) -> Tuple[np.ndarray, float, Status]:
        """
        Run the exchange heuristic under the chosen objective, for equally sized clusters.

        Parameters
        ----------
        X : array (N, D) feature matrix
        D : array (N, N) dissimilarity matrix (optional)

        Returns
        -------
        labels, score, Status.heuristic
        """
        N, _ = X.shape
        if N % self.K != 0:
            raise ValueError(f"N={N} not divisible by K={self.K}")

        # prepare dissimilarity for diversity
        if self.objective == "diversity":
            if D is None:
                D = get_dissimilarity_matrix(X)
        else:
            D = None

        best_labels = None
        best_score = -np.inf
        rng = np.random.default_rng(self.cfg.random_state)

        for _ in range(self.cfg.n_restarts):
            labels = np.repeat(np.arange(self.K), N // self.K)
            rng.shuffle(labels)
            score = self._score(X, D, labels)
            labels, score = self._exchange_loop(X, D, labels, score)
            if score > best_score:
                best_labels = labels.copy()
                best_score = score

        return best_labels, best_score, Status.heuristic

    def _score(self, X, D, labels):
        return self._obj_fn(X, labels) if self.objective == 'variance' else self._obj_fn(D, labels)

    def _exchange_loop(self, X, D, labels, score):
        N = X.shape[0]
        while True:
            best_delta = 0.0
            best_i = best_j = -1
            for i in range(N-1):
                for j in range(i+1, N):
                    if labels[i] == labels[j]:
                        continue
                    labels[i], labels[j] = labels[j], labels[i]
                    new_score = self._score(X, D, labels)
                    delta = new_score - score
                    labels[i], labels[j] = labels[j], labels[i]
                    if delta > best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta > self.tol:
                labels[best_i], labels[best_j] = labels[best_j], labels[best_i]
                score += best_delta
            else:
                break
        return labels, score


class GreedyExchangeHeuristic(ExchangeHeuristic):
    """
    Offline exchange heuristic seeded with a greedy balanced assignment.
    Uses the same swap loop as ExchangeHeuristic, but starts from
    a plain greedy allocation instead of a random init.
    """
    def solve(
        self,
        X: np.ndarray,
        D: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, Status]:
        N, _ = X.shape
        # prepare dissimilarity
        if self.objective == "diversity" and D is None:
            D = get_dissimilarity_matrix(X)

        # 1) greedy initial allocation
        labels = self._greedy_initial(X)
        score = self._obj_fn(X, labels) if self.objective == 'variance' else self._obj_fn(D, labels)

        # 2) apply exchange swaps on that seed
        labels, score = self._exchange_loop(X, D, labels, score)
        return labels, score, Status.heuristic

    def _greedy_initial(self, X: np.ndarray) -> np.ndarray:
        """
        Build a balanced greedy assignment for X:
          - Fill empty clusters first,
          - Otherwise assign each point to the cluster with max incremental gain.
        Returns a label array of length N.
        """
        N, F = X.shape
        clusters: dict[int, list[int]] = {j: [] for j in range(self.K)}
        sizes = [0] * self.K
        labels = np.empty(N, dtype=int)

        for i in range(N):
            xi = X[i]
            # empty cluster prioritization
            empty = [j for j, sz in enumerate(sizes) if sz == 0]
            if empty:
                best_j = empty[0]
            else:
                best_j = None
                best_gain = -np.inf
                for j in range(self.K):
                    idxs = clusters[j]
                    n_j = len(idxs)
                    # centroid
                    mu = X[idxs].mean(axis=0) if n_j > 0 else np.zeros(F)
                    # compute gain
                    if self.objective == 'variance':
                        diff2 = float(((xi - mu) ** 2).sum())
                        gain = (n_j / (n_j + 1.0)) * diff2
                    else:
                        block = X[idxs]
                        gain = np.linalg.norm(block - xi, axis=1).sum() / n_j
                    if gain > best_gain:
                        best_gain, best_j = gain, j
            # assign
            labels[i] = best_j
            clusters[best_j].append(i)
            sizes[best_j] += 1

        return labels
