
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence
import numpy as np

@dataclass(slots=True)
class OnlineBaseConfig:
    """Generic knobs that *any* anticlustering solver may use.

    Concrete subclasses can extend this dataclass in their own modules
    (e.g.  see :class:`ILPConfig` in *ilp_anticluster_improved.py*).
    """
    n_clusters: int
    random_state: Optional[int] = None
    time_limit: Optional[int] = None  # seconds

@dataclass(slots=True)
class OnlineExchangeConfig(OnlineBaseConfig):
    runtime: int = 10
    random_state: int = 42
    time_limit: int = 1000

    size_delta: int = 10
    k_neighbours: int = 10
    n_restarts: int = 1
    objective: str = "diversity"
    metric: str = "euclidean"
