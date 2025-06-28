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

    #FIXME: Is it logical for both assign and remove to rebalance? wouldn't it be better to only rebalance once? 
    # Either do this in a 'pipeline' or choose to first remove and then assign, or vice versa and lastly rebalance.

    def __init__(self, config: OnlineGreedyConfig) -> None:
        super().__init__(config)
        self.config = config

        self.K = config.n_clusters
        self.delta = config.size_delta
        self.obj = config.objective.lower()
        self.rebalance_method = config.rebalance_method.lower()
        self.size_balance_all_assignments = config.size_balance_all_assignments
        self.obj_f = None

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


    def objective_value(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int]
    ) -> float:
        """
        Compute the objective value for the current assignments.
        """
        if not data.ids:
            return 0.0

        ids = data.ids
        X = data.features
        id_to_idx = {lid: i for i, lid in enumerate(ids)}
        # Build current clusters
        clusters: Dict[int, List[int]] = {j: [] for j in range(self.K)}
        for lid, k in assignments.items():
            idx = id_to_idx[lid]
            clusters[k].append(idx)

        labels = np.array(
                [assignments.get(lid, -1) for lid in ids], dtype=int
            )
        
        # Compute the objective value
        if self.obj == "variance":
            # Determine the np.ndarray of cluster assignments
            
            # Compute the variance objective
            return self.obj_f(X, labels)
        elif self.obj == "diversity":
            # For diversity, we can directly use the assignments
            return self.obj_f(X, labels)

    def _greedy_assignment(
        self,
        data: StreamingDataStore,
        prev_assignments: Dict[str, int],
        new_ids: List[str],
        enforce_size: bool = False
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

        # Build current clusters & sizes
        clusters: Dict[int, List[int]] = {j: [] for j in range(self.K)}
        sizes = [0] * self.K
        for lid, k in assignments.items():
            idx = id_to_idx[lid]
            clusters[k].append(idx)
            sizes[k] += 1

        # Track which clusters are still empty
        empty = [j for j, sz in enumerate(sizes) if sz == 0]

        new_alloc = []
        # greedy assignment
        for lid in new_ids:
            i = id_to_idx[lid]
            xi = X[i]

            # 1) If any empty cluster remains, fill it immediately
            if empty:
                best_j = empty.pop(0)
                _LOG.info("_greedy_assignment: Assigning %s to empty cluster %d", lid, best_j)

            else:
                best_j    = None
                best_gain = -np.inf

                if enforce_size:
                    N_before = sum(sizes)
                    N_after  = N_before + 1
                    avg_size_after = N_after / self.K

                    feasible_clusters = [
                        j for j in range(self.K)
                        if abs(sizes[j] + 1 - avg_size_after) <= self.delta
                    ]

                    if not feasible_clusters:
                        raise RuntimeError(
                            f"_greedy_assignment: No feasible cluster for new id {lid} "
                            f"with size_delta={self.delta}. The current sizes are: {sizes}. "
                            f"Total N_before={N_before}, N_after={N_after}, "
                            f"avg_size_after={avg_size_after}. Allowed sizes: [{avg_size_after - self.delta}, {avg_size_after + self.delta}] "
                            "This should not happen."
                        )
                    
                else:
                    feasible_clusters = list(range(self.K))

                for j in feasible_clusters:
                    member_idxs = clusters[j]          # **without** i

                    mu_j  = X[member_idxs].mean(axis=0)
                    n_j  = len(member_idxs)
                    # compute incremental gain = sum_{m in cluster j} d(x_i, x_m)
                    # for the diversity objective:
                    if self.obj == "diversity":
                        # farthest point from centroid
                        block = X[member_idxs]

                        gain = np.linalg.norm(block - xi, axis=1).sum()
                        
                        # normalize by cluster size
                        gain /= n_j

                    # for the variance objective:
                    else:  # "variance"
                        # increase in sum‐of‐squared‐distances to centroid
                        diff2 = float(((xi - mu_j)**2).sum())
                        gain  = (n_j / (n_j + 1.0)) * diff2

                    _LOG.debug(
                        "_greedy_assignment: Gain for id %s in cluster %d (size: %d): %.3f",
                        lid, j, n_j, gain
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_j    = j

                if best_gain == - np.inf:
                    raise RuntimeError(
                        f"_greedy_assignment: No valid cluster found for new id {lid}. "
                        "This should not happen."
                    )
                
                _LOG.debug(
                    "_greedy_assignment: Assigning new id %s to cluster %d with gain=%.3f",
                    lid, best_j, best_gain
                )
        
            # now assign to the cluster with the largest **gain**
            assignments[lid] = best_j
            clusters[best_j].append(i)
            sizes[best_j] += 1
            new_alloc.append(best_j)

        _LOG.debug(
            "_greedy_assignment: Assigned the new ids to: %s", 
            new_alloc
        )

        return assignments
    
    def assign_new(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int],
        new_ids: List[str]
    ) -> Dict[str, int]:
        assignments = self._greedy_assignment(
            data, assignments, new_ids, enforce_size=self.size_balance_all_assignments
        )

        # If size_balance_all_assignments is True, we rebalance immediately
        if not self.size_balance_all_assignments:
            assignments = self.rebalance(data, assignments)
        
        return assignments


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

        # If size_balance_all_assignments is True, we rebalance immediately
        if self.size_balance_all_assignments:
            assignments = self.rebalance(data, updated)
        else:
            assignments = updated

        return assignments

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
        
        _LOG.info(
            "rebalance: Rebalancing needed: sizes before=%s, n=%d, K=%d, delta=%d",
            sizes_before, n, self.K, self.delta
        )

        if self.rebalance_method == "offline":
            updated = self._greedy_assignment(
                data, {}, data.ids, enforce_size=True
            )
        else:  # incremental
            updated = self._incremental_rebalance(data, assignments)

        sizes_after = self._cluster_sizes(updated)
        if self._needs_rebalance(sizes_after, n):
            raise RuntimeError(
                f"Rebalance failed to enforce size_delta={self.delta}. "
                f"Post‐rebalance sizes: {sizes_after}"
            )
        else:
            _LOG.info(
                "rebalance: Rebalance successful: sizes before=%s, after=%s",
                sizes_before, sizes_after
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
                "_incremental_rebalance: Best swap found: %s with gain=%.3f",
                best_move if best_move else "None",
                best_gain if best_move else -np.inf
            )

            if best_move and best_gain > -np.inf:
                lid, a, b = best_move
                i = id_to_idx[lid]
                x_i = X[i]
                labels[i] = b
                assignments[lid] = b
                # update counts & centroids incrementally if variance
                if self.obj == "variance":
                    # old counts
                    ca = counts[a]
                    cb = counts[b]
                    # update counts
                    counts[a] = ca - 1
                    counts[b] = cb + 1

                    # remove x_i from cluster a
                    if counts[a] > 0:
                        centroids[a] = (ca * centroids[a] - x_i) / counts[a]
                    else:
                        centroids[a] = np.zeros_like(centroids[a])

                    # add x_i to cluster b
                    centroids[b] = (cb * centroids[b] + x_i) / counts[b]

                continue

            _LOG.info("_incremental_rebalance: No positive‐gain swap remains; stopping.")
            break

        return assignments


    def finalize(self) -> None:
        """Nothing to clean up."""
        _LOG.debug("ExchangeOnlineSolver.finalize()") 