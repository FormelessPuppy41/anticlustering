
from .base import AntiCluster

_SOLVERS = {}

def register_solver(name: str):
    def _decorator(cls):
        _SOLVERS[name.lower()] = cls
        return cls
    return _decorator


def get_solver(name: str, /, *args, **kwargs) -> 'AntiCluster':
    try:
        cls = _SOLVERS[name.lower()]
    except KeyError:  # graceful error
        raise ValueError(f"Unknown solver '{name}'. "
                         f"Available: {list(_SOLVERS)}") from None
    return cls(*args, **kwargs)
