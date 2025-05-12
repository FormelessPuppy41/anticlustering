"""
Cluster editing anticlustering solver.
"""
import numpy as np
from ..base import Solver
from ..optimizers import optimize_with_exchange
from ..utils import compute_euclidean_distances
from ..factory import register_solver

@register_solver('cluster_editing')
class ClusterEditingSolver(Solver):
    """
    Anticlustering by maximizing sum of within-group pairwise distances.
    """
    def __init__(
        self,
        n_groups: int,
        max_iter: int = 1,
        random_state: int = None
    ):
        super().__init__(n_groups, max_iter, random_state)
        self._dissimilarity = None

    def objective(self, X: np.ndarray, labels: np.ndarray) -> float:
        if self._dissimilarity is None:
            self._dissimilarity = compute_euclidean_distances(X)
        total = 0.0
        for k in range(self.n_groups):
            idx = np.where(labels == k)[0]
            total += self._dissimilarity[np.ix_(idx, idx)].sum()
        return total

    def _optimize(self, X: np.ndarray, initial_labels: np.ndarray) -> np.ndarray:
        return optimize_with_exchange(X, initial_labels, self.objective, self.max_iter)
