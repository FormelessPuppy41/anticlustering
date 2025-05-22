"""
Reverse k-means anticlustering solver.
"""
import numpy as np
from ..base import Solver
from ..optimizers import optimize_with_exchange
from ..solver_factory import register_solver

@register_solver('kmeans')
class KMeansSolver(Solver):
    """
    Anticlustering via maximizing within-group variance (reverse k-means).
    """
    def objective(self, X: np.ndarray, labels: np.ndarray) -> float:
        K = self.n_groups
        centroids = np.vstack([
            X[labels == k].mean(axis=0) for k in range(K)
        ])
        return sum(((X[labels == k] - centroids[k]) ** 2).sum() for k in range(K))

    def _optimize(self, X: np.ndarray, initial_labels: np.ndarray) -> np.ndarray:
        return optimize_with_exchange(X, initial_labels, self.objective, self.max_iter)
