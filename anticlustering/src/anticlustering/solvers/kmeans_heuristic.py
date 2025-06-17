import numpy as np
from typing import Tuple
from ..core._config import KMeansConfig, Status

from ..metrics.dissimilarity_matrix import variance_objective, diversity_objective

from .exchange_heuristic import ExchangeHeuristic

class KMeansHeuristic:
    """
    Exchange heuristic to maximize within-cluster variance:
      1. Start with a random balanced assignment.
      2. Compute current variance‐objective V.
      3. Repeat up to max_sweeps:
         • For each i<j with labels[i] != labels[j]:
             – Swap them; compute V_new.
             – If V_new > V + tol, keep swap and update V.
             – Else revert.
         • If no swap improved V by tol, break.
    """

    def __init__(self, K: int, config: KMeansConfig):
        self.K = K
        self.cfg = config
        self.max_sweeps = getattr(config, "max_sweeps", 20)
        self.tol        = getattr(config, "tol", 0.0)
        self.rng        = np.random.default_rng(getattr(config, "random_state", None))

    def solve(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, float, Status]:
        ex_h = ExchangeHeuristic(
            D=None,  # Not used in this heuristic
            K=self.K,
            config=self.cfg,
            objective='variance'
        )

        return ex_h.solve(X=X, D=None)


        N = X.shape[0]
        if N % self.K != 0:
            raise ValueError(f"Cannot split N={N} into {self.K} equal groups")

        # 1) initial balanced random labels
        labels = np.repeat(np.arange(self.K), N // self.K)
        self.rng.shuffle(labels)

        # 2) initial objective
        best_score = variance_objective(X, labels)

        # 3) greedy exchange sweeps
        for sweep in range(self.max_sweeps):
            improved = False
            # try all cross‐cluster pairs
            for i in range(N - 1):
                for j in range(i + 1, N):
                    if labels[i] == labels[j]:
                        continue
                    # swap
                    labels[i], labels[j] = labels[j], labels[i]
                    score = variance_objective(X, labels)
                    if score > best_score + self.tol:
                        best_score = score
                        improved = True
                    else:
                        # revert
                        labels[i], labels[j] = labels[j], labels[i]
            if not improved:
                break

        best_score = diversity_objective(X, labels)

        return labels, best_score, Status.heuristic
