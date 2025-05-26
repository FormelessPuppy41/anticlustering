"""
Exchange-based heuristic as a separate component (Composition over Inheritance).
"""
import numpy as np

from .base import AntiCluster
from ._registry import register_solver

@register_solver('exchange')
class ExchangeAntiCluster(AntiCluster):
    
    def fit(self, X):
        raise NotImplementedError("ExchangeAntiCluster is not implemented yet.")