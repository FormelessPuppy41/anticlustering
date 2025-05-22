"""
Load registered solvers into the registry.
"""
# Ensure dynamic registration occurs by importing modules
from ..solver_factory import register_solver

from ._pairwise_mixin import PairwiseCacheMixin
from .kmeans import KMeansSolver
from .cluster_editing import ClusterEditingSolver
