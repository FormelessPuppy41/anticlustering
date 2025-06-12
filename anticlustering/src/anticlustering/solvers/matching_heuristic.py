import numpy as np
from typing import Tuple
from ..core._config import MatchingConfig, Status
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix


class MatchingHeuristic:
    """
    Greedy “max‐distance” matching for K=2 anticlustering.
    """

    def __init__(self, D: np.ndarray, K: int, config: MatchingConfig):
        if K != 2:
            raise ValueError(f"MatchingHeuristic only supports K=2, got K={K}")
        self.cfg = config
        self.K = K
        self.D = D.copy() if D is not None else None

    def solve(self, X: np.ndarray = None, *, D: np.ndarray) -> Tuple[np.ndarray, float, Status]:
        """
        Greedy max‐distance matching for K=2 anticlustering.

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
            Cluster labels 0 or 1.
        score : float
            Sum of within‐cluster dissimilarities.
        status : Status
            Always Status.heuristic here.
        
        Raises
        ------
        ValueError
            If D is not provided and X is None, or if N is odd.
        ValueError
            If K is not 2.
        """
        # 1) get or compute D
        if self.D is None:
            if X is None:
                raise ValueError("Either D or X must be provided")
            self.D = get_dissimilarity_matrix(X)
        else:
            self.D = D
            if D.shape[0] != D.shape[1] or D.ndim != 2:
                raise ValueError("D must be a square distance matrix")

        N = self.D.shape[0]
        if N % 2 != 0:
            raise ValueError(f"Cannot perfectly match odd N={N}")

        # 2) initialize labels and a working copy of the matrix
        M = self.D.copy()
        np.fill_diagonal(M, -np.inf)
        labels = np.empty(N, dtype=int)

        # 3) greedy remove max‐distance pair at each step
        for _ in range(N // 2):
            idx = np.argmax(M)
            i, j = np.unravel_index(idx, M.shape)
            labels[i] = 0
            labels[j] = 1
            # remove both from further consideration
            M[i, :] = -np.inf
            M[:, i] = -np.inf
            M[j, :] = -np.inf
            M[:, j] = -np.inf

        # 4) compute within‐cluster score
        score = 0.0
        for k in (0, 1):
            members = np.where(labels == k)[0]
            if members.size > 1:
                sub = self.D[np.ix_(members, members)]
                # each pair appears twice in the full submatrix
                score += sub.sum() / 2.0

        return labels, float(score), Status.heuristic
