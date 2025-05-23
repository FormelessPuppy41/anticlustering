"""
Factory and dynamic registration for Solver subclasses.
"""
from typing import Type, Dict, Callable
from .base_solver import Solver

from pathlib import Path
import pkgutil
from importlib import import_module

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

def _populate_registry_once() -> None:
    """Import every sub-module of anticlustering.solvers exactly once."""
    if _registry:         # already populated
        return
    import anticlustering.solvers as _pkg
    package_path = Path(_pkg.__file__).parent
    for mod_info in pkgutil.iter_modules([str(package_path)]):
        import_module(f"{_pkg.__name__}.{mod_info.name}")


def get_solver(
    name: str,
    n_groups: int,
    max_iter: int = 1,
    random_state: int = None
) -> Solver:
    """
    Instantiate a registered Solver by name.
    """
    if not _registry:  # populate the registry if empty
        _populate_registry_once()
        #print(f"Solver registry populated with: {list(_registry.keys())}")

    key = name.lower()
    if key not in _registry:
        raise ValueError(f"Unknown solver '{name}'. Available: {list(_registry.keys())}")
    SolverCls = _registry[key]
    return SolverCls(n_groups=n_groups, max_iter=max_iter, random_state=random_state)
