"""
anticlustering
"""
from importlib.metadata import version

from .models.anticluster_ilp import AnticlusterILP
from .algorithms.exchange import ExchangeAnticluster
from .algorithms.kmeans import KMeansAnticluster
from .solvers.solver_factory import get_solver

__all__ = [
    "AnticlusterILP",
    "ExchangeAnticluster",
    "KMeansAnticluster",
    "get_solver",
    "__version__",
]

__version__ = version(__name__)
