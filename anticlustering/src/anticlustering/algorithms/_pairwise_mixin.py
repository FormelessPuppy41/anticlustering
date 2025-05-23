# src/anticlustering/solvers/_pairwise_mixin.py
import numpy as np
from ..utils.distance_metrics import compute_euclidean_distances

class PairwiseCacheMixin:
    """Add lazy pair-wise distance caching to a solver."""
    _dissimilarity: np.ndarray | None = None

    def _get_dissim(self, X: np.ndarray) -> np.ndarray:
        # build once per X
        if (
            self._dissimilarity is None
            or self._dissimilarity.shape[0] != X.shape[0]
        ):
            self._dissimilarity = compute_euclidean_distances(X)
        return self._dissimilarity
