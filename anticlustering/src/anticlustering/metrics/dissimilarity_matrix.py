# src/anticlustering/solvers/_pairwise_mixin.py
import numpy as np
from .distance_metrics import compute_squared_euclidean_distances
from scipy.spatial.distance import cdist

class PairwiseCacheMixin:
    """Add lazy pair-wise distance caching to a solver."""
    _dissimilarity: np.ndarray | None = None

    def _get_dissim(self, X: np.ndarray) -> np.ndarray:
        # build once per X
        if (
            self._dissimilarity is None
            or self._dissimilarity.shape[0] != X.shape[0]
        ):
            self._dissimilarity = compute_squared_euclidean_distances(X)
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
        return np.sqrt(compute_squared_euclidean_distances(X))
    if distance_measure != 'euclidean':
        raise ValueError(f"Unsupported distance measure: {distance_measure}. Only 'euclidean' is supported.")
    # For now, only Euclidean distance is implemented


def diversity_objective(
    data: np.ndarray,
    clusters: np.ndarray
) -> float:
    """
    Compute the diversity (cluster‐editing) objective:
    the sum of pairwise Euclidean distances within each cluster.

    Args:
        data: either an (N x F) feature matrix or an (N x N) dissimilarity matrix.
        clusters: 1d array of length N with cluster labels (0,..,K-1 or any ints).
    Returns:
        Total within-cluster diversity (higher = more diverse).
    """
    # Determine if `data` is already a full dissimilarity matrix
    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        dissim = data
    else:
        # compute pairwise Euclidean distances (not squared)
        sq_d = compute_squared_euclidean_distances(data)
        dissim = np.sqrt(sq_d)

    total = 0.0
    for lbl in np.unique(clusters):
        idx = np.where(clusters == lbl)[0]
        # extract the submatrix for this cluster
        sub = dissim[np.ix_(idx, idx)]
        # sum over upper triangle to avoid double-counting
        triu = np.triu_indices_from(sub, k=1)
        total += sub[triu].sum()
    return total


def cluster_centers(
    data: np.ndarray,
    clusters: np.ndarray
) -> np.ndarray:
    """
    Compute cluster centroids.

    Args:
        data: (N x F) feature matrix.
        clusters: length-N vector of integer cluster labels.

    Returns:
        (K x F) array of cluster centers, 
        in the order of unique labels.
    """
    unique = np.unique(clusters)
    centers = np.vstack([data[clusters == lbl].mean(axis=0) for lbl in unique])
    return centers

def dist_from_centers(
    data: np.ndarray,
    centers: np.ndarray,
    squared: bool = False
) -> np.ndarray:
    """
    Compute distances from each point to each cluster center.

    Args:
        data: (N x F) feature matrix.
        centers: (K x F) centroids.
        squared: if True, return squared Euclidean distances.

    Returns:
        (N x K) distance matrix.
    """
    metric = "sqeuclidean" if squared else "euclidean"
    D = cdist(data, centers, metric=metric)
    return D

def variance_objective(
    data: np.ndarray,
    clusters: np.ndarray
) -> float:
    """
    Compute the k-means within-cluster variance objective:
    sum of squared distances of points to their assigned cluster center.

    Args:
        data: (N x F) feature matrix.
        clusters: length-N array of integer labels in [0..K-1].

    Returns:
        Scalar total within-cluster variance.
    """
    centers = cluster_centers(data, clusters)
    D = dist_from_centers(data, centers, squared=True)
    # select each point's distance to its own center
    idx = np.arange(data.shape[0])
    return D[idx, clusters].sum()




def within_group_distance(
    data: np.ndarray,
    clusters: np.ndarray
) -> float:
    """
    Compute the within-group distance objective:
    the sum of pairwise Euclidean distances within each cluster.

    Args:
        data: (N x F) feature matrix.
        clusters: length-N array of integer labels in [0..K-1].

    Returns:
        Total within-cluster distance (higher = more diverse).
    """
    return diversity_objective(data, clusters)