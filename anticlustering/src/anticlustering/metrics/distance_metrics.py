"""
Utility functions and caching for heavy computations.
"""
import numpy as np
from scipy.spatial.distance import cdist

def compute_euclidean_distances(data: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Caches result internally on first call for each solver.
    """
    return cdist(data, data, metric='euclidean') ** 2

def compute_manhattan_distances(data: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Manhattan distances.

    Caches result internally on first call for each solver.
    """
    return cdist(data, data, metric='cityblock')

def compute_mahalanobis_distances(data: np.ndarray, VI: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Mahalanobis distances.

    Caches result internally on first call for each solver.
    
    Args:
        data (np.ndarray): Input data array.
        VI (np.ndarray): Inverse covariance matrix for Mahalanobis distance.

    Returns:
        np.ndarray: Pairwise Mahalanobis distances.
    """
    return cdist(data, data, metric='mahalanobis', VI=VI)
