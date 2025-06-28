from typing import Dict, List
import numpy as np

from .online_base import BaseOnlineSolver  
from ..offline.greedy_exchange import GreedyExchangeAntiCluster, ExchangeConfig  # Greedy exchange implementation
from anticlustering.metrics.dissimilarity_matrix import variance_objective, diversity_objective


class OfflineExchangeSolver(BaseOnlineSolver):
    """
    Offline baseline solver: recomputes the full anticlusters at each update
    using the ExchangeAntiCluster class (offline exchange heuristic).
    """
    def __init__(self, config: ExchangeConfig) -> None:
        super().__init__(config)
        self.config = config
        # instantiate the offline model once
        self._model = GreedyExchangeAntiCluster(config)

    def assign_new(
        self,
        data,
        prev_assignments: Dict[str, int],
        new_ids: List[str]
    ) -> Dict[str, int]:
        """
        Upon new arrivals, ignore previous assignments and recompute
        a fresh partition from scratch on all current items.
        """
        # features matrix (N × D)
        X = data.features
        # fit offline exchange model (computes D internally)
        self._model.fit(X=X)
        # retrieve labels (array of length N)
        labels = getattr(self._model, 'labels_', None)
        if labels is None:
            raise RuntimeError("OfflineExchangeSolver: no labels found after fit()")
        # map each ID to its cluster label
        ids = data.ids
        return {lid: int(lbl) for lid, lbl in zip(ids, labels)}

    def remove_old(
        self,
        data,
        assignments: Dict[str, int],
        old_ids: List[str]
    ) -> Dict[str, int]:
        """
        After removals, recompute clustering from scratch.
        """
        return self.assign_new(data, assignments, [])

    def rebalance(
        self,
        data,
        assignments: Dict[str, int]
    ) -> Dict[str, int]:
        """
        No separate rebalance step needed; assign_new enforces balance.
        """
        return assignments.copy()

    def objective_value(
        self,
        data,
        assignments: Dict[str, int]
    ) -> float:
        """
        Compute the anticlustering objective (variance or diversity)
        on the current assignment.
        """
        ids = data.ids
        labels = np.array([assignments[lid] for lid in ids], dtype=int)
        X = data.features
        # choose metric based on config if available
        if hasattr(self.config, 'objective') and self.config.objective.lower() == 'variance':
            return variance_objective(X, labels)
        else:
            return diversity_objective(X, labels)

    def finalize(self) -> None:
        """
        No cleanup required for offline solver.
        """
        pass