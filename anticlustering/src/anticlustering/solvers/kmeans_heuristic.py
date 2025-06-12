import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from ..core._config import KMeansConfig, Status
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix, within_group_distance


class KMeansHeuristic:
    """
    Simple k‐means “baseline” for anticlustering (ignores equal‐size constraint).
    """

    def __init__(self, D: np.ndarray, K: int, config: KMeansConfig):
        self.cfg = config
        self.K = K
        self.D = D.copy() if D is not None else None

    def solve(self, X: np.ndarray = None, *, D: np.ndarray) -> Tuple[np.ndarray, float, Status]:
        """
        Simple k‐means heuristic for anticlustering.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Data points (not used in this heuristic).
        D : np.ndarray, shape (N, N)
            Dissimilarity matrix (N×N), where N is the number of data points.
        If D is None, it will be computed from X.
        
        Returns
        -------
        labels : np.ndarray, shape (N,)
        score  : float
            Sum of within‐cluster dissimilarities.
        status : Status
            Always Status.heuristic here.

        Raises
        ------
        ValueError
            If D is not provided and X is None.
        """
        # 1) obtain or compute full distance matrix
        if self.D is None:
            if X is None:
                raise ValueError("Need either X or precomputed D")
            self.D = get_dissimilarity_matrix(X)
        D = self.D
        N = D.shape[0]
        if N % self.K != 0:
            raise ValueError(f"N={N} not divisible by K={self.K}")

        # 2) random equal-size start
        size = N // self.K
        labels = np.repeat(np.arange(self.K), size)
        rng = np.random.default_rng(self.cfg.random_state)
        rng.shuffle(labels)

        best_score = within_group_distance(D, labels)
        improved = True
        while improved:
            improved = False

            # 3) compute centroids in feature space
            if X is None:
                raise ValueError("KMeansAntiCluster requires the original X")
            centroids = np.vstack([
                X[labels == k].mean(axis=0) for k in range(self.K)
            ])  # shape (K, n_features)

            # 4) build list of all (i,k,d^2(i,mu_k)) and sort descending
            #    we use D squared to approximate the k-means objective
            #    but final scoring uses D itself.
            d2 = np.square(
                np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            )  # shape (N, K)
            idxs, ks = np.unravel_index(
                np.argsort(d2.ravel())[::-1],
                (N, self.K)
            )

            # 5) greedy reassign under equal-size constraint
            new_labels = -np.ones(N, dtype=int)
            counts = {k: 0 for k in range(self.K)}
            for i, k in zip(idxs, ks):
                if new_labels[i] < 0 and counts[k] < size:
                    new_labels[i] = k
                    counts[k] += 1
                if all(c == size for c in counts.values()):
                    break

            # 6) measure true distance‐based score
            new_score = within_group_distance(D, new_labels)
            if new_score > best_score + 1e-8:
                labels, best_score = new_labels, new_score
                improved = True

        return labels, float(best_score), Status.heuristic