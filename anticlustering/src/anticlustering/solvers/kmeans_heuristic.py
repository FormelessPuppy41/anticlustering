import numpy as np
from typing import Tuple
from ..core._config import KMeansConfig, Status
from ..metrics.dissimilarity_matrix import sum_squared_to_centroids


class KMeansHeuristic:
    """
    K-Means anticlustering heuristic with strict equal‐size groups,
    random restarts, greedy assignment, and local swap optimization.

    Based on Papenberg & Klau’s “K-Means Clustering” criterion: we
    maximize the sum of squared Euclidean distances of points to
    their cluster centroids, subject to |c_j| = N/K for all clusters.
    """

    def __init__(self, K: int, config: KMeansConfig):
        """
        Initialize the anticlustering solver.

        Parameters
        ----------
        K : int
            Number of clusters (must divide N).
        config : KMeansConfig
            Configuration with attributes:
              - random_state: int, seed for reproducibility
              - n_restarts: int, number of random initializations
        """
        self.K = K
        self.cfg = config

    def solve(self, X: np.ndarray) -> Tuple[np.ndarray, float, Status]:
        """
        Partition X into K equally‐sized clusters that maximize
        within‐cluster variance (anticlustering).

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix.

        Returns
        -------
        labels : np.ndarray, shape (N,)
            Cluster assignments in 0,...,K-1.
        score : float
            Sum of squared distances to centroids (to maximize).
        status : Status
            Always Status.heuristic.

        Raises
        ------
        ValueError
            If X is None or N not divisible by K.
        """
        if X is None:
            raise ValueError("Data matrix X must be provided")
        N = X.shape[0]
        if N % self.K != 0:
            raise ValueError(f"N={N} not divisible by K={self.K}")

        best_labels = None
        best_score = -np.inf

        # Multiple random restarts to avoid poor local optima
        for _ in range(self.cfg.n_restarts):
            labels, score = self._single_run(X)
            if score > best_score:
                best_labels, best_score = labels, score

        return best_labels, float(best_score), Status.heuristic

    def _single_run(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        One run of the heuristic: random init, greedy assignment,
        then local pair-swap improvements.
        """
        rng = np.random.default_rng(self.cfg.random_state)
        N = X.shape[0]
        size = N // self.K

        # 1) Random equal‐size initialization
        labels = np.repeat(np.arange(self.K), size)
        rng.shuffle(labels)

        # 2) Greedy centroid‐based reassignments until no gain
        while True:
            centroids = np.vstack([X[labels == k].mean(axis=0)
                                   for k in range(self.K)])  # (K, D)
            # squared distances from each point to each centroid
            d2 = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2
            # flatten and sort descending: largest d2 first
            idxs, ks = np.unravel_index(
                np.argsort(d2.ravel())[::-1], (N, self.K)
            )
            # fill new_labels greedily under equal‐size constraint
            new_labels = -np.ones(N, dtype=int)
            counts = {k: 0 for k in range(self.K)}
            for i, k in zip(idxs, ks):
                if new_labels[i] < 0 and counts[k] < size:
                    new_labels[i] = k
                    counts[k] += 1
                if all(c == size for c in counts.values()):
                    break

            old_score = sum_squared_to_centroids(X, labels)
            new_score = sum_squared_to_centroids(X, new_labels)
            if new_score > old_score + 1e-8:
                labels = new_labels
            else:
                break

        # 3) Local pair‐swap optimization to further increase score
        current_score = sum_squared_to_centroids(X, labels)
        while True:
            improved = False
            for i in range(N - 1):
                for j in range(i + 1, N):
                    if labels[i] == labels[j]:
                        continue
                    # try swapping i<->j
                    swapped = labels.copy()
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    score_swapped = sum_squared_to_centroids(X, swapped)
                    if score_swapped > current_score + 1e-8:
                        labels = swapped
                        current_score = score_swapped
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return labels, current_score
