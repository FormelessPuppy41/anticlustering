# anticlustering/solvers/online/exchange_all.py

import logging
import heapq
from typing import Dict, List, Tuple
import numpy as np

from .online_greedy import OnlineGreedySolver        # your existing greedy solver
from .online_base import BaseOnlineSolver
from ...streaming.data_store import StreamingDataStore
from ._registry import register_online_solver
from ._config import OnlineExchangeConfig

from ...metrics.dissimilarity_matrix import variance_objective, diversity_objective

_LOG = logging.getLogger(__name__)

@register_online_solver("online_exchange")
class OnlineExchangeSolver(OnlineGreedySolver):
    """
    Online anticlustering solver that first applies the greedy one-pass
    assignment, then runs a full 2-exchange local-search over the entire
    stream to swap pairs of items between clusters until no positive-gain
    swap remains.
    """
    def __init__(self, config: OnlineExchangeConfig) -> None:
        """
        Initialize with configuration parameters.
        """
        super().__init__(config)
        self.config = config
        self.K = config.n_clusters
        self.obj = config.objective

        self.obj_f = (
            variance_objective if self.obj == "variance" else
            diversity_objective
        )
    
    def rebalance(
            self, 
            data: StreamingDataStore, 
            assignments: Dict[str, int]
        ) -> Dict[str, int]:
        assignments = super().rebalance(data, assignments)

        # After rebalancing, run the global 2-exchange pass again
        assignments = self._exchange_all(data, assignments)

        return assignments

    def _exchange_all(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Fast 2‐exchange via centroids + distance matrix.
        - Build centroids and N×K distance matrix.
        - For each cluster‐pair (a,b), pick:
            i* = argmax_{i in a} [d(i,μ_b)-d(i,μ_a)]
            j* = argmax_{j in b} [d(j,μ_a)-d(j,μ_b)]
        and score gain = Δ_i + Δ_j.
        - Swap the best pair, update only centroids[a], centroids[b],
        and update distances to those two centroids (two columns of Dmat).
        - Repeat until no positive‐gain swap or max_swaps reached.
        - Log number of swaps, initial score, final score, and Δ.
        """
        X = data.features            # shape (N, D)
        ids = data.ids
        N, D = X.shape
        K = self.K
        obj_f = self.obj_f

        # Create an array of current cluster assignments (shape N,)
        assign_arr = np.array([assignments[lid] for lid in ids], dtype=int)

        # Compute initial objective score
        initial_score = 0.0
        for k in range(K):
            idxs = np.where(assign_arr == k)[0]
            if idxs.size > 0:
                block = X[idxs]
                zeros = np.zeros(len(idxs), dtype=int)
                initial_score += obj_f(block, zeros)

        # Build initial centroids
        centroids = np.zeros((K, D), dtype=float)
        for k in range(K):
            mask = (assign_arr == k)
            if mask.any():
                centroids[k] = X[mask].mean(axis=0)

        # Build full distance matrix: Dmat[i,k] = ||X[i] - centroids[k]||
        Dmat = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

        max_swaps = getattr(self.config, "max_swaps_per_exchange", 100)
    
        _LOG.info("Starting fast exchange: N=%d, K=%d, max_swaps=%d", N, K, max_swaps)
        swaps_done = 0
        for _ in range(max_swaps):
            best_gain = 0.0
            best_move = None  # tuple (i_idx, j_idx, a, b)

            # scan over unordered pairs a<b
            for a in range(K):
                idxs_a = np.where(assign_arr == a)[0]
                if idxs_a.size == 0:
                    continue
                for b in range(a+1, K):
                    idxs_b = np.where(assign_arr == b)[0]
                    if idxs_b.size == 0:
                        continue
                    
                    # Calculate the gain for swapping items between clusters a and b:
                    
                    # Δ_i for each i in a: Dmat[i,b] - Dmat[i,a]
                    diffs_a = Dmat[idxs_a, b] - Dmat[idxs_a, a]
                    i_loc = diffs_a.argmax()
                    gain_i = diffs_a[i_loc]

                    # Δ_j for each j in b: Dmat[j,a] - Dmat[j,b]
                    diffs_b = Dmat[idxs_b, a] - Dmat[idxs_b, b]
                    j_loc = diffs_b.argmax()
                    gain_j = diffs_b[j_loc]

                    total_gain = gain_i + gain_j
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_move = (idxs_a[i_loc], idxs_b[j_loc], a, b)

            # no positive‐gain swap?
            if best_move is None or best_gain <= 0:
                break

            # perform the best swap
            i_idx, j_idx, a, b = best_move
            swaps_done += 1
            lid_i, lid_j = ids[i_idx], ids[j_idx]
            _LOG.debug(
                "Swap #%d: %s↔%s between clusters %d↔%d, gain=%.6f",
                swaps_done, lid_i, lid_j, a, b, best_gain
            )

            # update assignment array and dict
            assign_arr[i_idx], assign_arr[j_idx] = b, a
            assignments[lid_i], assignments[lid_j] = b, a

            # update centroids for a and b in O(D)
            mask_a = (assign_arr == a)
            mask_b = (assign_arr == b)
            centroids[a] = X[mask_a].mean(axis=0) if mask_a.any() else np.zeros(D)
            centroids[b] = X[mask_b].mean(axis=0) if mask_b.any() else np.zeros(D)

            # update only two columns of Dmat in O(N·D)
            Dmat[:, a] = np.linalg.norm(X - centroids[a], axis=1)
            Dmat[:, b] = np.linalg.norm(X - centroids[b], axis=1)

        # Compute final objective score
        final_score = 0.0
        for k in range(K):
            idxs = np.where(assign_arr == k)[0]
            if idxs.size > 0:
                block = X[idxs]
                zeros = np.zeros(len(idxs), dtype=int)
                final_score += obj_f(block, zeros)

        _LOG.info(
            "Finished exchange: swaps=%d, score: %.4f → %.4f (Δ=%.4f)",
            swaps_done, initial_score, final_score, final_score - initial_score
        )

        return assignments
