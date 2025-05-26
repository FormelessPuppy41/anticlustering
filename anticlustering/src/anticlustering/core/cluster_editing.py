"""
Cluster editing anticlustering solver.
"""
import numpy as np

from .base import AntiCluster
from ..metrics.dissimilarity_matrix import PairwiseCacheMixin, get_dissimilarity_matrix
from ._registry import register_solver

@register_solver('cluster_editing')
class ClusterEditingAntiCluster(AntiCluster, PairwiseCacheMixin):
    """
    Anticlustering by maximizing sum of within-group pairwise distances.
    """
    def fit(self, X):
        raise NotImplementedError("ClusterEditingAntiCluster is not implemented yet.")


