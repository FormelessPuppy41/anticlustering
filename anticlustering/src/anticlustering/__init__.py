"""
anticlustering – public API
"""
from importlib.metadata import version as _pkg_version

from .core.base import AntiCluster
from .core.kmeans import KMeansAntiCluster
from .core.cluster_editing import ClusterEditingAntiCluster
from .core.ilp import ILPAntiCluster
from .core._registry import get_solver

__all__ = [
    "AntiCluster",
    "KMeansAntiCluster",
    "ClusterEditingAntiCluster",
    "ILPAntiCluster",
    "get_solver",
    "__version__",
]

#__version__ = _pkg_version(__name__)
