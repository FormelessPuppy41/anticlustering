# _registry.py
from __future__ import annotations
from typing import Any

from .online_base import OnlineBaseConfig, BaseOnlineSolver

_SOLVERS: dict[str, type[BaseOnlineSolver]] = {}


def register_online_solver(name: str):
    def _decorator(cls: type[BaseOnlineSolver]):
        _SOLVERS[name.lower()] = cls
        return cls
    return _decorator


def get_online_solver(name: str, /, *args: Any, **kwargs: Any) -> BaseOnlineSolver:
    """
    Factory that instantiates a registered online solver.

    Parameters
    ----------
    name : str
        The key used in ``@register_solver`` (case-insensitive).
    config : OnlineBaseConfig
        **Required** keyword-only argument.  Contains all hyper-parameters
        for the solver (e.g. ``OnlineExchangeConfig`` or ``OnlineBaseConfig``).
    *args, **kwargs
        Any additional positional / keyword arguments are passed straight
        into the solver’s ``__init__`` *after* the config.

    Returns
    -------
    AntiCluster
        A *fitted* or *unfitted* solver instance, depending on your call.
    """
    try:
        cls = _SOLVERS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown solver '{name}'. Available: {list(_SOLVERS)}") from exc

    # ------------------------------------------------------------------ #
    # pull config out of kwargs or the first positional arg ------------- #
    if "config" in kwargs:
        config = kwargs.pop("config")
    elif args:
        config, *args = args
    else:
        raise TypeError(
            "get_online_solver() missing required argument 'config'. "
            "Call it like get_online_solver('exchange', config=OnlineExchangeConfig(...))."
        )

    if not isinstance(config, (OnlineBaseConfig)):
        raise TypeError(
            f"'config' must be a OnlineBaseConfig (got {type(config).__name__})."
        )

    # ------------------------------------------------------------------ #
    # instantiate and return ------------------------------------------- #
    return cls(config, *args, **kwargs)
