"""
anticlustering – public API
"""
from importlib.metadata import version as _pkg_version

from .core.offline.base import AntiCluster
from .core.offline.ilp import ILPAntiCluster
from .core.offline._registry import get_solver

__all__ = [
    "AntiCluster",
    "ILPAntiCluster",
    "get_solver",
    "__version__",
]

#__version__ = _pkg_version(__name__)
