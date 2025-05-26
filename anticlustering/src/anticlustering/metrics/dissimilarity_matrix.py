# src/anticlustering/solvers/_pairwise_mixin.py
import numpy as np
from .distance_metrics import compute_euclidean_distances

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


def get_dissimilarity_matrix(X: np.ndarray, distance_measure: str = 'euclidean') -> np.ndarray:
    """
    Compute the pairwise dissimilarity matrix for the given data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    distance_measure : str, optional
        The distance measure to use. Currently only 'euclidean' is supported.

    Returns
    -------
    dissimilarity_matrix : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarity matrix computed using the specified distance measure.
    """
    if distance_measure == 'euclidean':
        from .distance_metrics import compute_euclidean_distances
        return compute_euclidean_distances(X)
    if distance_measure != 'euclidean':
        raise ValueError(f"Unsupported distance measure: {distance_measure}. Only 'euclidean' is supported.")
    # For now, only Euclidean distance is implemented
    