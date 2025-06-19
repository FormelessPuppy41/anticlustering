# anticlustering/solvers/online/exchange.py

import logging
from typing import Dict, List

import numpy as np

from .online_base import BaseOnlineSolver
from ...streaming.data_store import StreamingDataStore
from ._config import OnlineExchangeConfig
from ..offline._config import ExchangeConfig
from anticlustering.solvers.exchange_heuristic import ExchangeHeuristic
from ..online._registry import register_online_solver

logger = logging.getLogger(__name__)

@register_online_solver("online_exchange")
class ExchangeOnlineSolver(BaseOnlineSolver):
    """
    Protocol‐compliant solver that takes full snapshots from the data store.
    No internal state beyond configuration.
    """
    def __init__(self, config: OnlineExchangeConfig) -> None:
        super().__init__(config)
        self.config = config

        self.K = config.n_clusters
        self.delta = config.size_delta
        self.m = config.k_neighbours
        self.R = config.n_restarts
        self.obj = config.objective

        self.off_cfg = ExchangeConfig(
            n_clusters=self.K,
            k_neighbours=self.m,
            n_restarts=self.R,
            objective=self.obj,
            metric=config.metric,
        )
        logger.debug("ExchangeOnlineSolver params: %s", config)
    
    def assign_new(
        self,
        data: StreamingDataStore,
        prev: Dict[str,int],
        new_ids: List[str]
    ) -> Dict[str,int]:
        assignments = prev.copy()
        if not new_ids:
            return assignments

        D = data.distances
        ids = data.ids
        for loan_id in new_ids:
            idx = data.index_of(loan_id)
            # consider distances to all existing points < idx
            d = D[idx, :idx]
            m = min(self.m, len(d))
            nn = np.argpartition(-d, m-1)[:m]
            # score clusters
            scores = [0.0]*self.K
            for i in nn:
                lbl = assignments[ids[i]]
                scores[lbl] += float(d[i])
            # pick best, tie‐break on size
            best = [j for j,s in enumerate(scores) if s == max(scores)]
            if len(best)>1:
                sizes = [list(assignments.values()).count(j) for j in best]
                chosen = best[sizes.index(min(sizes))]
            else:
                chosen = best[0]
            assignments[loan_id] = chosen

        return self.rebalance(data, assignments)

    
    def remove_old(
        self,
        data: StreamingDataStore,
        prev: Dict[str,int],
        old_ids: List[str]
    ) -> Dict[str,int]:
        """
        Drop old_ids from the assignment map, then optionally rebalance.
        """
        if not old_ids:
            return prev.copy()

        # filter out removed loans
        assignments = {lid:lbl for lid, lbl in prev.items() if lid not in set(old_ids)}
        missing = set(old_ids) - set(prev)
        if missing:
            logger.warning("remove_old: these IDs not found in prev assignments: %s", missing)

        # rebalance on the reduced set
        return self.rebalance(data, assignments)

    def rebalance(
        self,
        data: StreamingDataStore,
        assignments: Dict[str,int]
    ) -> Dict[str,int]:
        """
        Iteratively swap pairs of loans across clusters whenever
        the swap increases the total within-cluster distance sum.
        Stops when no improving swap exists or size drift ≤ delta.
        """
        ids = data.ids
        D   = data.distances
        n   = len(ids)
        if n == 0:
            return assignments

        # Check size drift
        sizes = [list(assignments.values()).count(c) for c in range(self.K)]
        if max(sizes) - min(sizes) <= self.delta:
            return assignments

        # Build cluster → member indices
        clusters: Dict[int, List[int]] = {
            c: [i for i, lid in enumerate(ids) if assignments[lid] == c]
            for c in range(self.K)
        }

        def swap_gain(i: int, j: int, ca: int, cb: int) -> float:
            """
            Gain = (sum of distances of i with cluster cb minus cluster ca)
                 + (sum of distances of j with cluster ca minus cluster cb)
            """
            # i currently in ca, j in cb
            members_a = [k for k in clusters[ca] if k != i]
            members_b = [k for k in clusters[cb] if k != j]
            gain_i = sum(D[i, k] for k in members_b) - sum(D[i, k] for k in members_a)
            gain_j = sum(D[j, k] for k in members_a) - sum(D[j, k] for k in members_b)
            return gain_i + gain_j

        improved = True
        while improved:
            best_gain = 0.0
            best_pair = None  # (i, j, ca, cb)
            # search all cluster‐pairs
            for ca in range(self.K):
                for cb in range(ca+1, self.K):
                    for i in clusters[ca]:
                        for j in clusters[cb]:
                            g = swap_gain(i, j, ca, cb)
                            if g > best_gain:
                                best_gain = g
                                best_pair = (i, j, ca, cb)
            if best_pair and best_gain > 0:
                i, j, ca, cb = best_pair
                # perform swap in assignments and clusters
                id_i, id_j = ids[i], ids[j]
                assignments[id_i], assignments[id_j] = cb, ca
                clusters[ca].remove(i); clusters[ca].append(j)
                clusters[cb].remove(j); clusters[cb].append(i)
                improved = True
                logger.debug(
                    "Swapped %s(idx=%d, from=%d→%d) with %s(idx=%d, from=%d→%d), gain=%.3f",
                    ids[i], i, ca, cb, ids[j], j, cb, ca, best_gain
                )
            else:
                improved = False

        logger.info(
            "Rebalance done via %s swaps; final size drift %d",
            "local", max(list(assignments.values()).count(c) for c in range(self.K)) 
            - min(list(assignments.values()).count(c) for c in range(self.K)))
        return assignments

    def finalize(self) -> None:
        logger.debug("ExchangeOnlineSolver.finalize()")
