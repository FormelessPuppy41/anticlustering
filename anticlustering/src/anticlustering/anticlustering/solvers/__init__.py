"""
Load registered solvers into the registry.
"""
# Ensure dynamic registration occurs by importing modules
from ..factory import register_solver
from .kmeans import KMeansSolver
from .cluster_editing import ClusterEditingSolver