import numpy as np
from typing import Tuple
from ..core._config import KMeansConfig, Status

from ..metrics.dissimilarity_matrix import within_group_distance

class KMeansHeuristic:
    """
    A k-means clustering heuristic mirroring R's stats::kmeans(), with
    k-means++ init, multiple restarts (n_init), convergence tolerance,
    empty-cluster handling, and reproducible RNG. Returns (labels, inertia, status).
    """

    def __init__(self, K: int, config: KMeansConfig):
        self.K = K
        self.cfg = config
        # Map your config fields to kmeans params (adjust names if your config differs)
        self.n_init = getattr(config, "n_restarts", 10)
        self.max_iter = getattr(config, "max_sweeps", 300)
        self.tol = getattr(config, "tol", 1e-4)
        self.random_state = getattr(config, "random_state", None)
        # optional: allow "random" vs "kmeans++"
        self.init_method = getattr(config, "init", "kmeans++")

    def solve(
        self, 
        X: np.ndarray, 
        *, 
        D: np.ndarray = None  # ignored for pure clustering
    ) -> Tuple[np.ndarray, float, Status]:
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        def _init_centroids() -> np.ndarray:
            # random initialization
            if self.init_method == "random":
                seeds = rng.choice(n_samples, size=self.K, replace=False)
                return X[seeds].copy()

            # kmeans++ initialization
            centroids = np.empty((self.K, n_features), dtype=X.dtype)
            # first center uniformly
            idx = rng.integers(n_samples)
            centroids[0] = X[idx]
            # distances to nearest existing centroid
            closest_dist_sq = np.full(n_samples, np.inf)
            for i in range(1, self.K):
                dist_to_new = np.sum((X - centroids[i-1]) ** 2, axis=1)
                closest_dist_sq = np.minimum(closest_dist_sq, dist_to_new)
                probs = closest_dist_sq / closest_dist_sq.sum()
                cumprobs = np.cumsum(probs)
                r = rng.random()
                centroids[i] = X[np.searchsorted(cumprobs, r)]
            return centroids

        best_inertia = np.inf
        best_labels = None

        # --- multiple restarts
        for _ in range(self.n_init):
            centroids = _init_centroids()

            for _ in range(self.max_iter):
                # assign points to nearest centroid
                dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                labels = np.argmin(dists, axis=1)
                inertia = float(np.sum(dists[np.arange(n_samples), labels]))

                # recompute centroids
                new_centroids = np.zeros_like(centroids)
                for k in range(self.K):
                    members = X[labels == k]
                    if members.shape[0] == 0:
                        # empty cluster → reseed
                        new_centroids[k] = X[rng.integers(n_samples)]
                    else:
                        new_centroids[k] = members.mean(axis=0)

                # check convergence by center shift
                shift = np.linalg.norm(centroids - new_centroids)
                centroids = new_centroids
                if shift**2 <= self.tol * inertia:
                    break

            # keep best restart
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()

        score = within_group_distance(D=D, labels=best_labels)

        return best_labels, score, Status.heuristic

















        N, _ = X.shape
        if N % self.K != 0:
            raise ValueError(f"N={N} not divisible by K={self.K}")

        best_labels = None
        best_score = -np.inf
        size = N // self.K
        rng = np.random.default_rng(self.cfg.random_state)

        for restart in range(self.cfg.n_restarts):
            # 1) random equal‐size init
            labels = np.repeat(np.arange(self.K), size)
            rng.shuffle(labels)
            score = sum_squared_to_centroids(X, labels)

            # 2) Exchange loop
            improved = True
            while improved:
                improved = False
                best_delta = 0.0
                best_i = best_j = -1

                # scan all cross‐cluster pairs
                for i in range(N - 1):
                    for j in range(i + 1, N):
                        if labels[i] == labels[j]:
                            continue
                        # compute objective change if we swap i↔j
                        labels[i], labels[j] = labels[j], labels[i]
                        new_score = sum_squared_to_centroids(X, labels)
                        delta = new_score - score
                        # revert
                        labels[i], labels[j] = labels[j], labels[i]

                        if delta > best_delta:
                            best_delta = delta
                            best_i, best_j = i, j

                # if we found a profitable swap, apply it
                if best_delta > 1e-8:
                    labels[best_i], labels[best_j] = labels[best_j], labels[best_i]
                    score += best_delta
                    improved = True

            # record best over restarts
            if score > best_score:
                best_score = score
                best_labels = labels.copy()

        best_score = within_group_distance(D=D, labels=best_labels)

        return best_labels, float(best_score), Status.heuristic