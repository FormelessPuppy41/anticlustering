# anticlustering/solvers/online/exchange_all.py

import logging
import heapq
from typing import Dict, List, Tuple, Optional
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
        assignments = self._exchange_all(data, assignments)
        assignments = super().rebalance(data, assignments)

        # After rebalancing, run the global 2-exchange pass again
        

        return assignments
    
    def _build_centroids(
        self, X: np.ndarray, assign_arr: np.ndarray
    ) -> np.ndarray:
        """Compute centroids for each cluster."""
        K, D = self.K, X.shape[1]
        centroids = np.zeros((K, D), dtype=float)
        for j in range(K):
            mask = (assign_arr == j)
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
        return centroids

    def _build_distance_matrix(
        self, X: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Compute distance matrix D[i,k] = ||X[i] - centroids[k]||."""
        return np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2
        )

    def _proxy_gain_variance(
        self,
        D2: np.ndarray,
        counts: Dict[int,int],
        Ca: List[int],
        Cb: List[int],
        a: int,
        b: int
    ) -> Optional[Tuple[int,int,float]]:
        """
        Compute surrogate gain for variance between clusters a and b.
        Returns (i_idx, j_idx, proxy_gain) or None if invalid.
        """
        ca = counts[a]
        cb = counts[b]
        if ca <= 1 or cb <= 1:
            return None
        # candidate from a->b
        gi = (cb/(cb+1.0)) * D2[Ca, b] - (ca/(ca-1.0)) * D2[Ca, a]
        i_loc    = int(gi.argmax()); gain_i = gi[i_loc]
        i_idx    = Ca[i_loc]
        # candidate from b->a
        gj = (ca/(ca+1.0)) * D2[Cb, a] - (cb/(cb-1.0)) * D2[Cb, b]
        j_loc    = int(gj.argmax()); gain_j = gj[j_loc]
        j_idx    = Cb[j_loc]
        return (i_idx, j_idx, gain_i + gain_j)

    def _proxy_gain_diversity(
        self,
        Dmat: np.ndarray,
        Ca: List[int],
        Cb: List[int],
        a: int,
        b: int
    ) -> Tuple[int,int,float]:
        """
        Compute surrogate gain for diversity between clusters a and b.
        Returns (i_idx, j_idx, proxy_gain).
        """
        diffs_a = Dmat[Ca, b] - Dmat[Ca, a]
        i_loc   = int(diffs_a.argmax()); gain_i = diffs_a[i_loc]
        i_idx   = Ca[i_loc]
        diffs_b = Dmat[Cb, a] - Dmat[Cb, b]
        j_loc   = int(diffs_b.argmax()); gain_j = diffs_b[j_loc]
        j_idx   = Cb[j_loc]
        return (i_idx, j_idx, gain_i + gain_j)

    def _find_best_proxy_swap(
        self,
        Dmat: np.ndarray,
        clusters: Dict[int, List[int]]
    ) -> Optional[Tuple[int,int,int,int,float]]:
        """
        Scan all unordered cluster-pairs (a<b) and propose the best swap
        under the chosen surrogate (variance or diversity).
        """
        best: Tuple[Optional[int],Optional[int],Optional[int],Optional[int],float]
        best = (None, None, None, None, 0.0)
        K = self.K
        # precompute squared-distances and counts if variance
        D2 = Dmat**2 if self.obj == "variance" else None
        counts = {j: len(clusters[j]) for j in range(K)} if self.obj == "variance" else None

        for a in range(K):
            Ca = clusters[a]
            if not Ca:
                continue
            for b in range(a+1, K):
                Cb = clusters[b]
                if not Cb:
                    continue

                if self.obj == "variance":
                    result = self._proxy_gain_variance(D2, counts, Ca, Cb, a, b)
                else:
                    result = self._proxy_gain_diversity(Dmat, Ca, Cb, a, b)

                if result is None:
                    continue
                i_idx, j_idx, proxy_gain = result
                if proxy_gain > best[4]:
                    best = (i_idx, j_idx, a, b, proxy_gain)

        if best[0] is None:
            return None
        return best  # type: ignore

    def _compute_true_gain(
        self,
        X: np.ndarray,
        Ca: List[int],
        Cb: List[int],
        i_idx: int,
        j_idx: int
    ) -> float:
        """
        Compute actual objective gain from swapping i_idx in Ca with j_idx in Cb.
        """
        obj_f = self.obj_f
        # before
        A = X[Ca]; B = X[Cb]
        zerosA = np.zeros(len(Ca), dtype=int)
        zerosB = np.zeros(len(Cb), dtype=int)
        obj_before = obj_f(A, zerosA) + obj_f(B, zerosB)
        # after indices
        A2_idx = [u for u in Ca if u != i_idx] + [j_idx]
        B2_idx = [u for u in Cb if u != j_idx] + [i_idx]
        A2 = X[A2_idx]; B2 = X[B2_idx]
        zerosA2 = np.zeros(len(A2_idx), dtype=int)
        zerosB2 = np.zeros(len(B2_idx), dtype=int)
        obj_after = obj_f(A2, zerosA2) + obj_f(B2, zerosB2)
        return obj_after - obj_before

    def _apply_swap(
        self,
        X: np.ndarray,
        assign_arr: np.ndarray,
        assignments: Dict[str,int],
        clusters: Dict[int, List[int]],
        centroids: np.ndarray,
        Dmat: np.ndarray,
        ids: List[str],
        swap: Tuple[int,int,int,int]
    ) -> None:
        """
        Execute swap (i_idx, j_idx, a, b), update all structures in place.
        """
        i_idx, j_idx, a, b = swap
        lid_i, lid_j = ids[i_idx], ids[j_idx]
        # update assignment arrays
        assign_arr[i_idx], assign_arr[j_idx] = b, a
        assignments[lid_i], assignments[lid_j] = b, a
        # update clusters
        clusters[a].remove(i_idx); clusters[a].append(j_idx)
        clusters[b].remove(j_idx); clusters[b].append(i_idx)
        # update centroids for a and b
        for j in (a, b):
            idxs = clusters[j]
            if idxs:
                centroids[j] = X[idxs].mean(axis=0)
            else:
                centroids[j].fill(0)
        # update distance matrix columns for a and b
        Dmat[:, a] = np.linalg.norm(X - centroids[a], axis=1)
        Dmat[:, b] = np.linalg.norm(X - centroids[b], axis=1)

    def _exchange_all(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int]
    ) -> Dict[str, int]:
        """
        One‐pass largest‐gain exchange on the active items.
        """
        X   = data.features               # shape (N, M)
        ids = data.ids                    # list of length N
        N   = X.shape[0]

        # Build label array and cluster lists
        labels = np.array([assignments[lid] for lid in ids], dtype=int)
        clusters = {k: np.where(labels == k)[0].tolist() for k in range(self.K)}

        # Precompute full pairwise distance matrix once
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        _LOG.debug("Distance matrix computed with shape %s. And input %s", D.shape, D)
        # For each item i, search for best single swap
        for i in range(N):
            a = labels[i]
            # Precompute sums of distances from i into its own and each other cluster
            sum_i_a = D[i, clusters[a]].sum()
            best_delta = 0.0
            best_j = None
            best_b = None

            for b, members_b in clusters.items():
                if b == a or len(members_b) == 0:
                    continue

                # Sum of distances from i into cluster b
                sum_i_b = D[i, members_b].sum()

                # For each candidate j in cluster b, compute swap delta
                # delta = (sum of j->a) - (sum of i->a) + (sum of i->b) - (sum of j->b)
                # Precompute sum_j_a and sum_j_b arrays
                sum_j_a = D[np.ix_(members_b, clusters[a])].sum(axis=1)
                sum_j_b = D[np.ix_(members_b, members_b)].sum(axis=1)

                # Compute deltas for all j in one go
                deltas = sum_j_a - sum_i_a + sum_i_b - sum_j_b

                # Find best j in this cluster
                idx_local = np.argmax(deltas)
                delta_j = deltas[idx_local]
                if delta_j > best_delta:
                    best_delta = delta_j
                    best_j = members_b[idx_local]
                    best_b = b

            # If a positive‐gain swap was found, execute it
            if best_delta > 0 and best_j is not None:
                # swap i <-> best_j between clusters a and best_b
                j = best_j
                labels[i], labels[j] = best_b, a

                # update clusters lists
                clusters[a].remove(i);   clusters[a].append(j)
                clusters[best_b].remove(j); clusters[best_b].append(i)

        # write back into assignments dict
        for idx, lid in enumerate(ids):
            assignments[lid] = int(labels[idx])

        return assignments
    
    def exchange_all(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int]
    ) -> Dict[str, int]:
        X   = data.features
        ids = data.ids
        assign_arr = np.array([assignments[lid] for lid in ids], dtype=int)
        clusters   = {j: np.where(assign_arr == j)[0].tolist() for j in range(self.K)}

        centroids = self._build_centroids(X, assign_arr)
        Dmat      = self._build_distance_matrix(X, centroids)

        swaps_done = 0
        max_swaps  = getattr(self.config, "max_swaps_per_exchange", 100)

        initial_score = self.obj_f(X, assign_arr)

        for _ in range(max_swaps):
            best = self._find_best_proxy_swap(Dmat, clusters)
            if not best or best[4] <= 0: # no positive gain found
                break
            i_idx, j_idx, a, b, proxy_gain = best

            true_gain = self._compute_true_gain(X, clusters[a], clusters[b], i_idx, j_idx)
            if true_gain <= 0:
                _LOG.debug(
                    "exchange: proxy=%.4f but true_gain=%.4f ≤ 0, stopping",
                    proxy_gain, true_gain
                )
                continue

            self._apply_swap(
                X, assign_arr, assignments,
                clusters, centroids, Dmat, ids,
                (i_idx, j_idx, a, b)
            )
            swaps_done += 1

        final_score = self.obj_f(X, assign_arr)
        diff = final_score - initial_score

        _LOG.debug("exchange complete: %d swaps applied. Score increase: %.4f (from %.4f to %.4f)", swaps_done, diff, initial_score, final_score)
        return assignments
