"""
Reverse k-means anticlustering solver.
"""
import numpy as np

from .base import AntiCluster
from ._registry import register_solver

@register_solver('kmeans')
class KMeansAntiCluster(AntiCluster):
    """
    Anticlustering via maximizing within-group variance (reverse k-means).
    """

    def fit(self, X):
        raise NotImplementedError("KMeansAntiCluster is not implemented yet.")
