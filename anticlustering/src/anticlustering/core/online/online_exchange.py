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

from ...metrics.dissimilarity_matrix import variance_objective, diversity_objective

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
        
        self.obj_f = (
            variance_objective if self.obj == "variance" else
            diversity_objective if self.obj == "diversity" else
            None
        )
        if self.obj_f is None:
            raise ValueError(f"Unsupported objective: {self.obj}. Use 'variance' or 'diversity'.")

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
                scores[lbl] += float(d[i]) #FIXME: Shouldn't this also be based on the objective?
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
        X   = data.features
        n   = len(ids)
        if n == 0:
            return assignments

        #FIXME: Fully rewrite this bcs it does not work. 
        avg_size = n / self.K
        sizes = [list(assignments.values()).count(c) for c in range(self.K)]
        if len(sizes) != self.K:
            raise ValueError(
                f"Assignments must contain {self.K} clusters, but found {len(sizes)}: {sizes}"
            )
        
        if abs(max(sizes) - avg_size) <= self.delta:
            logger.info(
                "Rebalance not needed: size drift %d is within delta %d",
                abs(max(sizes) - avg_size), self.delta
            )
            return assignments
        
        labels = np.array([assignments[lid] for lid in ids])

        # compute starting objective
        if self.obj == "variance":
            current_obj = self.obj_f(X, labels)
        else:  # diversity
            current_obj = self.obj_f(D, labels)
        logger.debug(
            "Initial objective: %.4f, size drift=%.4f",
            current_obj, abs(max(sizes) - avg_size)
        )

        def swap_gain(i: int, j: int, ca: int, cb: int) -> float:
            """
            Gain = (sum of distances of i with cluster cb minus cluster ca)
                 + (sum of distances of j with cluster ca minus cluster cb)
            """
            # i currently in ca, j in cb
            members_a = [k for k in range(n) if labels[k] == ca and k != i]
            members_b = [k for k in range(n) if labels[k] == cb and k != j]
            gain_i = sum(D[i, k] for k in members_b) - sum(D[i, k] for k in members_a)
            gain_j = sum(D[j, k] for k in members_a) - sum(D[j, k] for k in members_b)
            return gain_i + gain_j

        #FIXME: We do the wrong algorithm here! We should not be swapping, but moving a single item!
        while abs(max(sizes) - avg_size) > self.delta:
            best_gain = -float("inf")
            best_pair = None  # (i, j, ca, cb)
            for i in ids:
                for j in ids:
                    if j <= i:
                        continue
                    ca = assignments[i]
                    cb = assignments[j]
                    if ca == cb:
                        continue

                    # compute swap gain
                    g = swap_gain(ids.index(i), ids.index(j), ca, cb)
                    if g > best_gain:
                        best_gain = g
                        best_pair = (i, j, ca, cb)
            
            if best_pair and best_gain > 0:
                i, j, ca, cb = best_pair
                # perform swap in assignments
                assignments[i], assignments[j] = cb, ca
                logger.debug(
                    "Swapped %s(idx=%d, from=%d→%d) with %s(idx=%d, from=%d→%d), gain=%.3f",
                    i, ids.index(i), ca, cb, j, ids.index(j), cb, ca, best_gain
                )
                # update sizes
                sizes[ca] -= 1; sizes[cb] += 1


            pass


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
            best_gain = 0
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
