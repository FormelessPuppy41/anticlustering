# anticlustering/solvers/online/anticlustream.py

import logging
from typing import Dict, List

import numpy as np

from .online_base import BaseOnlineSolver
from ._config import OnlineDenStreamConfig
from ._registry import register_online_solver
from ...streaming.data_store import StreamingDataStore

_LOG = logging.getLogger(__name__)


class MicroCluster:
    """
    Exponentially‐decaying micro‐cluster summary that retains per‐member vectors.
    """

    def __init__(self, vec: np.ndarray, loan_id: str, t: float, lambda_decay: float):
        self.cf1 = vec.copy()
        self.cf2 = vec * vec
        self.weight = 1.0
        self.t_last = t
        self.lambda_decay = lambda_decay
        self.member_vecs: Dict[str, np.ndarray] = {loan_id: vec.copy()}

    def decay(self, t: float) -> None:
        delta = t - self.t_last
        if delta <= 0:
            return
        factor = np.exp(-self.lambda_decay * delta)
        self.cf1 *= factor
        self.cf2 *= factor
        self.weight *= factor
        self.t_last = t

    @property
    def centroid(self) -> np.ndarray:
        return self.cf1 / self.weight

    def add(self, vec: np.ndarray, loan_id: str, t: float) -> None:
        self.decay(t)
        self.cf1 += vec
        self.cf2 += vec * vec
        self.weight += 1.0
        self.member_vecs[loan_id] = vec.copy()

    def remove(self, loan_id: str, t: float) -> None:
        self.decay(t)
        vec = self.member_vecs.pop(loan_id, None)
        if vec is None:
            _LOG.warning("MicroCluster.remove: loan_id %s not found", loan_id)
            return
        self.cf1 -= vec
        self.cf2 -= vec * vec
        self.weight -= 1.0


@register_online_solver("online_anticlustream")
class OnlineDenStreamSolver(BaseOnlineSolver):
    """
    Anti‐CluStream solver: maintains decaying micro‐clusters,
    then does one‐pass greedy anticlustering on their centroids.
    """

    def __init__(self, config: OnlineDenStreamConfig) -> None:
        super().__init__(config)
        self.config = config
        self.m = config.n_microclusters
        self.K = config.n_clusters
        self.lambda_decay = config.lambda_decay
        self._rng = np.random.RandomState(config.random_state)

        self._time = 0.0
        self.microclusters: List[MicroCluster] = []
        self._id_to_mc: Dict[str, int] = {}

    def assign_new(
        self,
        data: StreamingDataStore,
        prev_assignments: Dict[str, int],
        new_ids: List[str]
    ) -> Dict[str, int]:
        self._time += 1.0
        X = data.features
        id2idx = {lid: i for i, lid in enumerate(data.ids)}

        for lid in new_ids:
            vec = X[id2idx[lid]]
            for mc in self.microclusters:
                mc.decay(self._time)

            if len(self.microclusters) < self.m:
                mc = MicroCluster(vec, lid, self._time, self.lambda_decay)
                self.microclusters.append(mc)
                self._id_to_mc[lid] = len(self.microclusters) - 1
            else:
                centroids = np.vstack([mc.centroid for mc in self.microclusters])
                scores = np.linalg.norm(centroids - vec, axis=1)
                j = int(scores.argmax())
                self.microclusters[j].add(vec, lid, self._time)
                self._id_to_mc[lid] = j

        return self.rebalance(data, prev_assignments)

    def remove_old(
        self,
        data: StreamingDataStore,
        assignments: Dict[str, int],
        old_ids: List[str]
    ) -> Dict[str, int]:
        if not old_ids:
            return assignments.copy()

        self._time += 1.0
        for lid in old_ids:
            mc_idx = self._id_to_mc.pop(lid, None)
            if mc_idx is not None:
                self.microclusters[mc_idx].remove(lid, self._time)

        # prune any empty microclusters
        self.microclusters = [mc for mc in self.microclusters if mc.weight > 1e-6]
        # rebuild the id→mc map
        self._id_to_mc = {
            lid: idx
            for idx, mc in enumerate(self.microclusters)
            for lid in mc.member_vecs.keys()
        }

        return self.rebalance(data, assignments)

    def rebalance(
        self,
        data: StreamingDataStore,
        prev_assignments: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Recluster micro‐clusters into K anticlusters with exact balancing,
        then ensure every loan_id in data.ids is assigned.
        """
        # 1) If no micro‐clusters yet, just preserve existing labels for everyone.
        if not self.microclusters:
            # fallback: assign everyone to 0 if even prev_assignments is empty
            return {lid: prev_assignments.get(lid, 0) for lid in data.ids}

        # 2) Build weight and centroid lists
        mcs = self.microclusters
        m = len(mcs)
        D = mcs[0].centroid.size
        weights = [len(mc.member_vecs) for mc in mcs]
        centroids = [mc.centroid for mc in mcs]

        # 3) Sort micro‐clusters descending by weight
        order = sorted(range(m), key=lambda j: weights[j], reverse=True)

        # 4) Compute exact target sizes
        n = len(data.ids)
        base = n // self.K
        rem = n % self.K
        target = [base + (1 if k < rem else 0) for k in range(self.K)]

        # 5) Prepare accumulators
        cluster_weights = [0] * self.K
        cluster_centroids = [np.zeros(D) for _ in range(self.K)]
        mc_labels = [-1] * m

        # 6) Seed the top‐K largest micro‐clusters one to each cluster
        for seed_k, j_mc in enumerate(order[: self.K]):
            mc_labels[j_mc] = seed_k
            w = weights[j_mc]
            cluster_weights[seed_k] = w
            cluster_centroids[seed_k] = centroids[j_mc].copy()

        # 7) Greedy‐diversity assign the rest with capacity constraints
        for j_mc in order[self.K :]:
            w = weights[j_mc]
            vec = centroids[j_mc]

            best_k = None
            best_score = -np.inf

            for k in range(self.K):
                if cluster_weights[k] + w > target[k]:
                    continue
                diff = vec - cluster_centroids[k]
                score = float(np.linalg.norm(diff))
                if best_k is None or score > best_score:
                    best_k, best_score = k, score

            # if _no_ valid place (shouldn’t happen), pick the emptiest
            if best_k is None:
                # choose cluster with minimum weight
                best_k = int(min(range(self.K), key=lambda x: cluster_weights[x]))

            # assign and update
            mc_labels[j_mc] = best_k
            cw = cluster_weights[best_k]
            cluster_centroids[best_k] = (cluster_centroids[best_k] * cw + vec * w) / (cw + w)
            cluster_weights[best_k] += w

        # 8) Build the new assignments map
        new_assignments: Dict[str,int] = {}

        # 8a) First, inherit any prev_assignments for IDs *not* in any microcluster
        for lid in data.ids:
            if lid not in self._id_to_mc:
                new_assignments[lid] = prev_assignments.get(lid, 0)

        # 8b) Then, overlay microcluster‐based labels
        for j_mc, mc in enumerate(mcs):
            lbl = mc_labels[j_mc]
            for lid in mc.member_vecs.keys():
                new_assignments[lid] = lbl

        # 9) Final sanity: ensure completeness
        missing = set(data.ids) - new_assignments.keys()
        for lid in missing:
            # this really shouldn't happen, but just in case
            new_assignments[lid] = int(self._rng.randint(self.K))

        return new_assignments

    def finalize(self) -> None:
        _LOG.debug("OnlineDenStreamSolver.finalize()")
