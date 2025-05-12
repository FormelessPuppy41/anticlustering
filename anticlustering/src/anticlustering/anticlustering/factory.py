"""
Factory and dynamic registration for Solver subclasses.
"""
from typing import Type, Dict, Callable
from .base import Solver

_registry: Dict[str, Type[Solver]] = {}

def register_solver(name: str) -> Callable[[Type[Solver]], Type[Solver]]:
    """
    Decorator to register a Solver subclass under a key.

    Example:
    @register_solver('kmeans')
    class KMeansSolver(Solver): ...
    """
    def decorator(cls: Type[Solver]) -> Type[Solver]:
        _registry[name.lower()] = cls
        return cls
    return decorator


def get_solver(
    name: str,
    n_groups: int,
    max_iter: int = 1,
    random_state: int = None
) -> Solver:
    """
    Instantiate a registered Solver by name.
    """
    key = name.lower()
    if key not in _registry:
        raise ValueError(f"Unknown solver '{name}'. Available: {list(_registry.keys())}")
    SolverCls = _registry[key]
    return SolverCls(n_groups=n_groups, max_iter=max_iter, random_state=random_state)
