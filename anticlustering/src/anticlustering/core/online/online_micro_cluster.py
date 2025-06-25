# anticlustering/solvers/online/anticlustream.py

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .online_base import BaseOnlineSolver
from ._config import OnlineDenStreamConfig  # you’ll need to add this config class
from ._registry import register_online_solver
from ...streaming.data_store import StreamingDataStore
from ...metrics.dissimilarity_matrix import diversity_objective, variance_objective

_LOG = logging.getLogger(__name__)

class MicroCluster:
    """
    Exponentially‐decaying micro‐cluster summary.

    Attributes
    ----------
    cf1 : np.ndarray[D]
        Linear sum of (decayed) vectors.
    cf2 : np.ndarray[D]
        Sum of elementwise squares (for possible future variance use).
    weight : float
        Total (decayed) mass.
    t_last : float
        Last timestamp at which decay was applied.
    members : List[str]
        List of member IDs (for back‐mapping).
    """
    def __init__(self, vec: np.ndarray, loan_id: str, t: float, lambda_decay: float):
        self.cf1 = vec.copy()
        self.cf2 = vec * vec
        self.weight = 1.0
        self.t_last = t
        self.lambda_decay = lambda_decay
        self.members = [loan_id]

    def decay(self, t: float) -> None:
        """Apply exponential decay from t_last to t."""
        Δ = t - self.t_last
        if Δ <= 0:
            return
        factor = np.exp(-self.lambda_decay * Δ)
        self.cf1 *= factor
        self.cf2 *= factor
        self.weight *= factor
        self.t_last = t

    @property
    def centroid(self) -> np.ndarray:
        return self.cf1 / self.weight

    def add(self, vec: np.ndarray, loan_id: str, t: float) -> None:
        """Decay, then absorb a new point."""
        self.decay(t)
        self.cf1 += vec
        self.cf2 += vec * vec
        self.weight += 1.0
        self.members.append(loan_id)

    def remove(self, vec: np.ndarray, loan_id: str, t: float) -> None:
        """Decay, then remove a departing point."""
        self.decay(t)
        self.cf1 -= vec
        self.cf2 -= vec * vec
        self.weight -= 1.0
        self.members.remove(loan_id)
        # Note: if weight falls ≤ 0, caller should prune this micro‐cluster.

@register_online_solver("online_anticlustream")
class OnlineDenStreamSolver(BaseOnlineSolver):
    """
    Anti-CluStream: DenStream‐style micro-clusters with decay,
    plus offline greedy anticlustering on micro-cluster centroids.
    """

    def __init__(self, config: OnlineDenStreamConfig) -> None:
        """
        Parameters
        ----------
        config.n_microclusters : int
            Number of micro-clusters to maintain (m).
        config.lambda_decay : float
            Exponential decay rate λ.
        config.n_clusters : int
            Final anticluster count (K).
        config.random_state : int
            RNG seed for tie‐breaking.
        """
        super().__init__(config)
        self.config = config
        self.m = config.n_microclusters
        self.K = config.n_clusters
        self.lambda_decay = config.lambda_decay
        self._rng = np.random.RandomState(config.random_state)

        self._time: float = 0.0
        self.microclusters: List[MicroCluster] = []
        # loan_id → microcluster index
        self._id_to_mc: Dict[str,int] = {}

    def assign_new(
        self,
        data: StreamingDataStore,
        prev_assignments: Dict[str,int],
        new_ids: List[str]
    ) -> Dict[str,int]:
        """
        1) Advance time by 1 unit
        2) For each new loan:
             – decays all micro-clusters to current time
             – if < m microclusters: create a new one
             – else: assign to the microcluster whose centroid is farthest
               (maximizing diversity), then update that microcluster
        3) Recompute final anticluster labels via offline rebalance
        """
        # advance logical clock
        self._time += 1.0

        X = data.features           # (N_total × D)
        id2idx = {lid:i for i,lid in enumerate(data.ids)}

        # 1) Update microclusters with each new point
        for lid in new_ids:
            vec = X[id2idx[lid]]
            # decay all
            for mc in self.microclusters:
                mc.decay(self._time)

            if len(self.microclusters) < self.m:
                # warm-up: spawn a new microcluster
                mc = MicroCluster(vec, lid, self._time, self.lambda_decay)
                self.microclusters.append(mc)
                self._id_to_mc[lid] = len(self.microclusters) - 1

            else:
                # find the farthest‐centroid microcluster
                centroids = np.vstack([mc.centroid for mc in self.microclusters])
                dists = np.linalg.norm(centroids - vec, axis=1)
                j = int(dists.argmax())
                self.microclusters[j].add(vec, lid, self._time)
                self._id_to_mc[lid] = j

        # 2) Delegate final assignment to offline rebalance
        #    (will cluster microclusters → K anticlusters)
        return self.rebalance(data, prev_assignments)

    def remove_old(
        self,
        data: StreamingDataStore,
        assignments: Dict[str,int],
        old_ids: List[str]
    ) -> Dict[str,int]:
        """
        Remove departed IDs from their micro-clusters, then rebalance.
        """
        if not old_ids:
            return assignments.copy()

        X = data.features
        id2idx = {lid:i for i,lid in enumerate(data.ids)}
        self._time += 1.0

        for lid in old_ids:
            mc_idx = self._id_to_mc.pop(lid, None)
            if mc_idx is None:
                continue
            vec = X[id2idx[lid]]
            mc = self.microclusters[mc_idx]
            mc.remove(vec, lid, self._time)

        # Optionally prune empty / near-zero microclusters
        self.microclusters = [
            mc for mc in self.microclusters if mc.weight > 1e-6
        ]
        # rebuild id→mc indices
        self._id_to_mc = {
            lid: j for j, mc in enumerate(self.microclusters)
            for lid in mc.members
        }

        return self.rebalance(data, assignments)

    def rebalance(
        self,
        data: StreamingDataStore,
        assignments: Dict[str,int]
    ) -> Dict[str,int]:
        """
        Offline: cluster the m microclusters into K anticlusters
        via one‐pass greedy assignment (maximizing centroid distances),
        then propagate labels to all loan_ids.
        """
        if not self.microclusters:
            return {}

        # build centroid array & membership lists
        centroids = [mc.centroid for mc in self.microclusters]
        m = len(centroids)
        C = np.vstack(centroids)    # (m × D)

        # 1-pass greedy on microclusters
        mc_labels = [-1]*m
        counts = [0]*self.K
        means = np.zeros((self.K, C.shape[1]))

        order = list(range(m))
        self._rng.shuffle(order)
        for j_mc in order:
            vec = C[j_mc]
            best_k = None
            best_score = None
            for k in range(self.K):
                if counts[k] == 0:
                    score = 0.0  # seed empty cluster
                else:
                    diff = vec - means[k]
                    score = float(np.linalg.norm(diff))
                if best_k is None or score > best_score:
                    best_k, best_score = k, score
            # assign
            mc_labels[j_mc] = best_k
            # update cluster mean
            c = counts[best_k]
            means[best_k] = (means[best_k]*c + vec) / (c+1)
            counts[best_k] += 1

        # propagate to loans
        final = {}
        for j_mc, mc in enumerate(self.microclusters):
            k = mc_labels[j_mc]
            for lid in mc.members:
                final[lid] = k

        return final

    def finalize(self) -> None:
        """Nothing to clean up for DenStream solver."""
        _LOG.debug("OnlineDenStreamSolver.finalize()")
