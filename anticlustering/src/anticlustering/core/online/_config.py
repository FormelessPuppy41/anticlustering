
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
class OnlineGreedyConfig(OnlineBaseConfig):
    runtime: int = 10
    random_state: int = 42
    time_limit: int = 1000

    size_delta: int = 5
    objective: str = "diversity" # 'diversity' or 'variance'
    rebalance_method: str = "offline" # 'offline' or 'incremental
    size_balance_all_assignments: bool = False # 'True' > maintain size balance at each assignment step, 'False' > only at rebalancing

@dataclass(slots=True)
class OnlineExchangeConfig(OnlineGreedyConfig):
    runtime: int = 10
    random_state: int = 42
    time_limit: int = 1000

    size_delta: int = 5
    objective: str = "diversity" # 'diversity' or 'variance'
    rebalance_method: str = "offline" # 'offline' or 'incremental
    size_balance_all_assignments: bool = False # 'True' > maintain size balance at each assignment step, 'False' > only at rebalancing




@dataclass(slots=True)
class OnlineDenStreamConfig(OnlineBaseConfig):
    runtime: int = 10
    random_state: int = 42
    time_limit: int = 1000

    n_microclusters: int = 100
    lambda_decay: float = 0.01

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if self.n_microclusters <= 0:
            raise ValueError("n_microclusters must be a positive integer")
        if self.lambda_decay <= 0:
            raise ValueError("lambda_decay must be a positive float")