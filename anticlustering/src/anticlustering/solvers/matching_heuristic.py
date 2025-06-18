import numpy as np
from typing import Tuple, Optional
from ..core._config import MatchingConfig, Status
from ..metrics.dissimilarity_matrix import get_dissimilarity_matrix, diversity_objective


class MatchingHeuristic:
    """
    Greedy “matching” anticlustering for K=2 using the minimum‐distance heuristic.

    Implements the procedure from Papenberg & Klau (p. 164):
      1. Compute all pairwise Euclidean distances.
      2. Find the two most similar items (smallest distance).
      3. Randomly assign one to each of the two groups.
      4. Remove these two from consideration.
      5. Repeat until every item is assigned.
    """

    def __init__(
        self,
        D: Optional[np.ndarray],
        K: int,
        config: MatchingConfig,
    ):
        """
        Parameters
        ----------
        D : np.ndarray or None
            Precomputed dissimilarity matrix (N×N). If None,
            `solve` will compute it from X.
        K : int
            Number of clusters. Must be 2 for this heuristic.
        config : MatchingConfig
            Configuration object (e.g., seed, verbosity).
        """
        if K != 2:
            raise ValueError(f"MatchingHeuristic only supports K=2, got K={K}")
        self.K = K
        self.cfg = config
        self.rng = np.random.default_rng(config.random_state)
        # Store a copy of the provided distance matrix (or None)
        self.D = D.copy() if D is not None else None

    def solve(
        self,
        X: Optional[np.ndarray] = None,
        *,
        D: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Status]:
        """
        Perform greedy matching‐based anticlustering for K=2.

        Parameters
        ----------
        X : np.ndarray, shape (N, d), optional
            Data matrix. Used to compute D if no matrix is provided.
        D : np.ndarray, shape (N, N), optional
            Dissimilarity matrix. Overrides any stored self.D.

        Returns
        -------
        labels : np.ndarray, shape (N,)
            Binary cluster labels (0 or 1) for each item.
        score : float
            Sum of within‐cluster dissimilarities.
        status : Status
            Always Status.heuristic on success.

        Raises
        ------
        ValueError
            If neither X nor D is given, if D is not square,
            or if number of items N is odd.
        """
        # 1) Obtain or compute the dissimilarity matrix
        if D is not None:
            D_mat = D
        elif self.D is None:
            if X is None:
                raise ValueError("Either X or a precomputed D must be provided")
            D_mat = get_dissimilarity_matrix(X)
        # Validate matrix
        if D_mat.ndim != 2 or D_mat.shape[0] != D_mat.shape[1]:
            raise ValueError("D must be a square (N×N) dissimilarity matrix")

        N = D_mat.shape[0]
        if N % 2 != 0:
            raise ValueError(f"Cannot match an odd number of items: N={N}")

        # 2) Prepare a working copy and label array
        M = D_mat.copy()
        # Prevent self‐matching by setting diagonal to +∞
        np.fill_diagonal(M, np.inf)
        # Initialize labels to -1 (unassigned)
        labels = -1 * np.ones(N, dtype=int)

        # 3) Iteratively pick and assign the most similar pair
        for _ in range(N // 2):
            # Find the closest pair (i, j)
            flat_idx = np.argmin(M)
            i, j = np.unravel_index(flat_idx, M.shape)

            # Randomly assign one to cluster 0 and the other to cluster 1
            if self.rng.random() < 0.5:
                labels[i] = 0
                labels[j] = 1
            else:
                labels[i] = 1
                labels[j] = 0

            # Exclude these two from further matching
            M[i, :] = np.inf
            M[:, i] = np.inf
            M[j, :] = np.inf
            M[:, j] = np.inf

        # 4) Compute the within‐cluster total dissimilarity
        score = diversity_objective(self.D, labels)

        return labels, float(score), Status.heuristic
