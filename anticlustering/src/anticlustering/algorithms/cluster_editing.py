"""
Cluster editing anticlustering solver.
"""
import numpy as np
from ..solvers.base_solver import Solver
from ..anticlustering.solvers import PairwiseCacheMixin
from ..anticlustering.optimizers import optimize_with_exchange
from ..solvers.solver_factory import register_solver

@register_solver('cluster_editing')
class ClusterEditingSolver(PairwiseCacheMixin, Solver):
    """
    Anticlustering by maximizing sum of within-group pairwise distances.
    """
    def objective(
            self, 
            X: np.ndarray, 
            labels: np.ndarray
        ) -> float:
        dissim = self._get_dissim(X)
        total = 0.0
        for k in range(self.n_groups):
            idx = np.where(labels == k)[0]
            total += dissim[np.ix_(idx, idx)].sum()
        return total

    def _optimize(
            self, 
            X: np.ndarray, 
            initial_labels: np.ndarray
        ) -> np.ndarray:
        return optimize_with_exchange(
            X, 
            initial_labels, 
            self.objective, 
            self.max_iter
        )
