from __future__ import annotations

import numpy as np

from typing import List



class ModelAntiClusterOnline:
    def __init__(self):
        pass

    def solve(self):
        raise NotImplementedError
    

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility types
Vector = Sequence[float]          # a data point
ItemID = str | int                # unique loan / record identifier
ClusterID = int                   # running index 0 .. K-1


# ---------------------------------------------------------------------------
# Online cluster representation
# ---------------------------------------------------------------------------
@dataclass
class OnlineCluster:
    """Container for items flowing into an 'anti-cluster' in streaming mode.

    Parameters
    ----------
    cid : int
        Numerical id of the cluster (0-based).
    dim : int
        Data dimensionality.
    max_size : int
        Target / upper-bound cardinality of the cluster.
    """
    cid: ClusterID
    dim: int
    max_size: int
    _ids: Deque[ItemID] = field(default_factory=deque, repr=False, init=False)
    _sum: np.ndarray = field(init=False, repr=False)
    _sq_sum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sum = np.zeros(self.dim)
        self._sq_sum = np.zeros(self.dim)

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> Tuple[ItemID, ...]:
        return tuple(self._ids)

    # ---------------------------------------------------------- online updates
    def add(self, iid: ItemID, x: Vector) -> None:
        """Append a new observation and update running moments."""
        x = np.asarray(x)
        if len(x) != self.dim:
            raise ValueError("Dimension mismatch")
        if len(self) == self.max_size:
            raise RuntimeError("Cluster is already full")
        self._ids.append(iid)
        self._sum += x
        self._sq_sum += x ** 2

    def remove(self, iid: ItemID, x: Vector) -> None:
        """Remove an item (needed for exchange heuristics)."""
        try:
            self._ids.remove(iid)
        except ValueError as exc:
            raise KeyError(f"id {iid!r} not in cluster {self.cid}") from exc
        x = np.asarray(x)
        self._sum -= x
        self._sq_sum -= x ** 2

    # ------------------------------------------------ aggregate descriptors ---
    @property
    def mean(self) -> np.ndarray:
        return self._sum / max(len(self), 1)

    @property
    def var(self) -> np.ndarray:
        n = max(len(self), 1)
        return self._sq_sum / n - self.mean ** 2
