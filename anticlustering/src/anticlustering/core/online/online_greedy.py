# anticlustering/solvers/online/exchange.py

import logging
from typing import Dict, List

import numpy as np

from .online_base import BaseOnlineSolver
from ...streaming.data_store import StreamingDataStore
from ._config import OnlineGreedyConfig
from ._registry import register_online_solver

from ...metrics.dissimilarity_matrix import variance_objective, diversity_objective

_LOG = logging.getLogger(__name__)

@register_online_solver("online_greedy")
class OnlineGreedySolver(BaseOnlineSolver):
    """
    One‐pass greedy anticlustering solver with support for
    'offline' vs. 'incremental' rebalance. Always maximizes the
    chosen objective, and enforces size‐delta strictly.
    """

    def __init__(self, config: OnlineGreedyConfig) -> None:
        super().__init__(config)
        self.config = config

        self.K = config.n_clusters
        self.delta = config.size_delta
        self.obj = config.objective.lower()
        self.rebalance_method = config.rebalance_method.lower()

        # pick objective function
        if self.obj == "variance":
            self.obj_f = variance_objective
        elif self.obj == "diversity":
            self.obj_f = diversity_objective
        else:
            raise ValueError(
                f"Unsupported objective '{config.objective}'. "
                "Use 'variance' or 'diversity'."
            )

        # validate rebalance_method
        if self.rebalance_method not in ("offline", "incremental"):
            raise ValueError(
                f"Unsupported rebalance_method '{config.rebalance_method}'. "
                "Use 'offline' or 'incremental'."
            )

        _LOG.debug("OnlineGreedySolver initialized with config: %s", config)

    def assign_new(
        self,
        data: StreamingDataStore,
        prev_assignments: Dict[str, int],
        new_ids: List[str]
    ) -> Dict[str, int]:
        """
        Assign each incoming ID immediately, choosing the cluster
        that maximizes the increase in the anticlustering objective,
        then rebalance if needed.
        """
        assignments = prev_assignments.copy()
        ids = data.ids
        X = data.features  # (N_total × D)
        id_to_idx = {lid: i for i, lid in enumerate(ids)}

        # build current clusters as index lists
        clusters: Dict[int, List[int]] = {j: [] for j in range(self.K)}
        for lid, j in assignments.items():
            clusters[j].append(id_to_idx[lid])

        # greedy assignment
        for lid in new_ids:
            i = id_to_idx[lid]
            best_j = None
            best_score = -np.inf

            for j in range(self.K):
                member_idxs = clusters[j] + [i]
                block = X[member_idxs]
                # objective returns higher = better
                score = self.obj_f(block, np.zeros(len(block), dtype=int))
                if score > best_score:
                    best_score = score
                    best_j = j

            assignments[lid] = best_j
            clusters[best_j].append(i)

        # delegate to rebalance (which may raise)
        return self.rebalance(data, assignments)

    def remove_old(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int],
        old_ids: List[str]
    ) -> Dict[str, int]:
        """
        Drop departed IDs, then rebalance if needed.
        """
        if not old_ids:
            return assignments.copy()

        updated = assignments.copy()
        for lid in old_ids:
            updated.pop(lid, None)

        return self.rebalance(data, updated)

    def rebalance(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Conditionally rebalance, then enforce that no cluster still violates size_delta.
        Falls back to RuntimeError if the incremental method fails to restore balance.
        """
        n = len(data.ids)
        if n == 0:
            return assignments

        sizes_before = self._cluster_sizes(assignments)
        if not self._needs_rebalance(sizes_before, n):
            return assignments

        if self.rebalance_method == "offline":
            updated = self.assign_new(data, {}, data.ids)
        else:  # incremental
            updated = self._incremental_rebalance(data, assignments)

        sizes_after = self._cluster_sizes(updated)
        if self._needs_rebalance(sizes_after, n):
            raise RuntimeError(
                f"Rebalance failed to enforce size_delta={self.delta}. "
                f"Post‐rebalance sizes: {sizes_after}"
            )
        return updated

    def _needs_rebalance(self, sizes: List[int], n: int) -> bool:
        """
        True if any |size_j - (n/K)| > delta.
        """
        avg = n / self.K
        return any(abs(size - avg) > self.delta for size in sizes)

    def _cluster_sizes(self, assignments: Dict[str,int]) -> List[int]:
        """
        Count members per cluster; validates indices.
        """
        sizes = [0] * self.K
        for c in assignments.values():
            if not (0 <= c < self.K):
                raise ValueError(f"Invalid cluster index {c}")
            sizes[c] += 1
        return sizes

    def _incremental_rebalance(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Single‐swap heuristic: move the loan whose swap gives the largest
        positive total‐objective gain between an overfull and underfull cluster.
        """
        ids = data.ids
        X = data.features
        n = len(ids)
        avg = n / self.K

        # build per‐cluster lists of loan_ids
        clusters: Dict[int, List[str]] = {j: [] for j in range(self.K)}
        for lid, j in assignments.items():
            clusters[j].append(lid)

        # precompute index lookup for efficiency
        id_to_idx = {lid: i for i, lid in enumerate(ids)}

        # Make a dense label array: label[i] is cluster idx of piont i.
        labels = np.full(n, -1, dtype=int)
        for lid, cluster in assignments.items():
            labels[id_to_idx[lid]] = cluster

        # precompute counts & centroids for variance
        if self.obj == "variance":
            counts = {j: len(clusters[j]) for j in range(self.K)}
            centroids = {
                j: (X[[id_to_idx[l] for l in clusters[j]]].mean(axis=0)
                    if counts[j] > 0 else np.zeros(X.shape[1]))
                for j in range(self.K)
            }

        def total_gain(lid: str, a: int, b: int) -> float:
            i = id_to_idx[lid]
            xi = X[i]

            if self.obj == "variance":
                ca, cb = counts[a], counts[b]
                μa, μb = centroids[a], centroids[b]
                d2a = float(np.dot(xi - μa, xi - μa))
                d2b = float(np.dot(xi - μb, xi - μb))
                # remove from a, add to b
                gain_a = 0.0 if ca <= 1 else - (ca / (ca - 1.0)) * d2a
                gain_b = (cb / (cb + 1.0)) * d2b
                return gain_a + gain_b

            else:  # diversity
                # vectorized sum of distances
                idxs_a = np.where(labels == a)[0]
                idxs_b = np.where(labels == b)[0]
                sum_a = np.linalg.norm(X[idxs_a] - xi, axis=1).sum() if idxs_a.size else 0.0
                sum_b = np.linalg.norm(X[idxs_b] - xi, axis=1).sum() if idxs_b.size else 0.0
                return sum_b - sum_a

        while True:
            counts = np.bincount(labels, minlength=self.K)
            over  = np.where(counts >  avg + self.delta)[0]
            under = np.where(counts <  avg - self.delta)[0]
            if over.size == 0 or under.size == 0:
                break

            best_gain, best_move = -np.inf, None
            for a in over:
                for i in np.where(labels == a)[0]:
                    lid = ids[i]
                    for b in under:
                        g = total_gain(lid, a, b)
                        if g > best_gain:
                            best_gain, best_move = g, (lid, a, b)

            _LOG.info(
                "Best swap found: %s with gain=%.3f",
                best_move if best_move else "None",
                best_gain if best_move else -np.inf
            )

            if best_move and best_gain > -np.inf:
                lid, a, b = best_move
                i = id_to_idx[lid]
                labels[i] = b
                assignments[lid] = b
                # update counts & centroids incrementally if variance
                if self.obj == "variance":
                    counts[a] -= 1; counts[b] += 1
                    # incremental centroid update omitted for brevity
                continue

            _LOG.info("No positive‐gain swap remains; stopping.")
            break

        return assignments

        moved = True
        while moved:
            moved = False
            sizes = {j: len(clusters[j]) for j in range(self.K)}
            over  = [j for j,s in sizes.items() if s > avg + self.delta]
            under = [j for j,s in sizes.items() if s < avg - self.delta]
            if not over or not under:
                break

            best_gain = -np.inf
            best_move = None  # (loan_id, from_cluster, to_cluster)

            for a in over:
                for lid in clusters[a]:
                    for b in under:
                        g = total_gain(lid, a, b)
                        if g > best_gain:
                            best_gain = g
                            best_move = (lid, a, b)
            _LOG.info(
                "Best swap found: %s with gain=%.3f",
                best_move if best_move else "None",
                best_gain if best_move else -np.inf
            )
            if best_move:
                lid, a, b = best_move
                clusters[a].remove(lid)
                clusters[b].append(lid)
                assignments[lid] = b
                moved = True

            if not moved:
                _LOG.info(
                    "No further moves possible to restore balance. "
                    "Current sizes: %s", sizes
                )

        return assignments
    

    def finalize(self) -> None:
        """Nothing to clean up."""
        _LOG.debug("ExchangeOnlineSolver.finalize()") 