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
    
def within_group_distance(
        D: np.ndarray,
        labels: np.ndarray,
    ) -> float:
    """
    Total within-group dissimilarity for a given partition.

    Parameters
    ----------
    D :
        (N×N) **symmetric** dissimilarity matrix with zeros on the diagonal.
    labels :
        1-D iterable of length *N* assigning each observation to a group.

    Returns
    -------
    float
        \\( \sum_{g \\in G} \sum_{i<j \\in g} D_{ij} \\) – i.e. the sum of all
        pairwise distances **inside** every group.  Divisor 2 is applied so
        each unordered pair contributes only once.

    Notes
    -----
    *Time complexity*: \\(O(N²)\\) in the worst case; in practice dominated by
    the size of each group’s sub-matrix.
    """
    if D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")
    labels = np.asarray(labels)
    if len(labels) != D.shape[0]:
        raise ValueError("labels length mismatch")

    total = 0.0
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if idx.size < 2:           # singleton → zero contribution
            continue
        sub = D[np.ix_(idx, idx)]
        total += sub.sum() * 0.5   # divide by 2 to avoid double-count

    return float(total)



def sum_squared_to_centroids(X, labels) -> float:
    """
    Total sum of squared distances to the centroids of each group.
    This is a measure of within-group variance.
    
    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Data points.
    labels : np.ndarray, shape (N,)
        Group labels for each data point.

    Returns
    -------
    float
        Total sum of squared distances to the centroids of each group.

    Notes
    -----
    This function computes the sum of squared distances of each point in a group
    to the mean (centroid) of that group. It is a common measure of variance
    within groups, often used in clustering contexts.
    The function iterates over each unique label, computes the mean of the points
    in that group, and then sums the squared distances of those points to the mean.
    This is useful for evaluating the compactness of clusters in clustering algorithms.
    """
    total = 0.0
    for j in np.unique(labels):
        idx = np.where(labels == j)[0]
        mu = X[idx].mean(axis=0)
        total += ((X[idx] - mu)**2).sum()
    return total