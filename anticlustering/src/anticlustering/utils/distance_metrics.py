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
    return cdist(data, data, metric='euclidean')
