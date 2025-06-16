# src/anticlustering/solvers/_pairwise_mixin.py
import numpy as np
from .distance_metrics import compute_euclidean_distances
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
        sq_d = compute_euclidean_distances(data)
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


# def weighted_diversity_objective(
#     data: np.ndarray,
#     clusters: np.ndarray,
#     frequencies: np.ndarray
# ) -> float:
#     """
#     Compute the weighted diversity objective:
#     sum over clusters of (within-cluster diversity / frequency).

#     Args:
#         data: (N x F) features or (N x N) dissimilarity.
#         clusters: cluster labels length N.
#         frequencies: 1d array of length K with frequencies for each cluster in label order.
#     Returns:
#         Weighted diversity score.
#     """
#     # Precompute full dissimilarity if needed
#     if data.ndim == 2 and data.shape[0] == data.shape[1]:
#         dissim = data
#     else:
#         sq_d = compute_euclidean_distances(data)
#         dissim = np.sqrt(sq_d)

#     total = 0.0
#     unique = np.unique(clusters)
#     for i, lbl in enumerate(unique):
#         idx = np.where(clusters == lbl)[0]
#         sub = dissim[np.ix_(idx, idx)]
#         triu = np.triu_indices_from(sub, k=1)
#         group_div = sub[triu].sum()
#         total += group_div / frequencies[i]
#     return total


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




#TODO: Legacy code...
    
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