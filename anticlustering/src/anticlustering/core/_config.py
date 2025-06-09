
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence
import numpy as np


@dataclass(slots=True)
class BaseConfig:
    """Generic knobs that *any* anticlustering solver may use.

    Concrete subclasses can extend this dataclass in their own modules
    (e.g.  see :class:`ILPConfig` in *ilp_anticluster_improved.py*).
    """
    n_clusters: int
    random_state: Optional[int] = None

@dataclass(slots=True)
class ILPConfig(BaseConfig):
    """
    Tunable parameters for :class:`ILPAntiCluster`. Inherits from :class:`BaseConfig`.

    Notes
    -----
    * ``time_limit`` is interpreted in *seconds* and forwarded to the underlying
      MIP solver if supported (Gurobi, CBC, CPLEX, HiGHS).
    * ``warm_start`` may come from a fast heuristic such as the exchange
      algorithm; it should be a 1‑D integer array of length *N* containing group
      labels *(0 … K‑1)*.
    """

    n_clusters      : int                   = 2  # number of clusters (K)
    solver_name     : str                   = "gurobi"
    max_n           : Optional[int]         = None  # max number of items (N) to solve
    time_limit      : Optional[int]         = None
    mip_gap         : Optional[float]       = None  # absolute or relative depending on solver
    warm_start      : Optional[np.ndarray]  = None
    preclustering   : bool                  = False  # whether to use preclustering
    verbose         : bool                  = False  # print solver output
    lazy_transitivity   : bool              = True  # whether to use lazy transitivity cuts

    # future extensions ------------------------------------------------------
    categories  : Optional[np.ndarray]      = None  # 1‑D categorical strata

    # guard rails ------------------------------------------------------------

    def validate(self, n_items: int) -> None:
        if self.n_clusters <= 1:
            raise ValueError("n_clusters must be >= 2")
        if n_items % self.n_clusters:
            raise ValueError(
                f"Number of items {n_items} not divisible by K={self.n_clusters}."
            )
        if n_items > self.max_n:
            raise ValueError(
                f"Problem too large for the exact ILP (N={n_items} > {self.max_n})."
            )


@dataclass(slots=True)
class ExchangeConfig(BaseConfig):
    """
    Tunable parameters for :class:`ExchangeAntiCluster`. Inherits from :class:`BaseConfig`.

    Notes
    -----
    * ``max_iter`` is the maximum number of iterations to run the exchange algorithm.
    * ``max_n`` is the maximum number of items (N) to solve.
    * ``time_limit`` is interpreted in *seconds* and limits the runtime of the algorithm.
    """
    objective       : str                   = "diversity"
    metric          : str                   = "euclidean"
    max_sweeps      : int                   = 10
    patience        : int                   = 100
    verbose         : bool                  = False
    time_limit      : Optional[int]         = None  # seconds


@dataclass(slots=True)
class OnlineConfig(BaseConfig):
    """Streaming-specific knobs (inherits common fields from OnlineConfig stub)."""
    hard_balance_cols       : Sequence[str] = ("grade", "sub_grade")
    size_tolerance          : int = 1
    rebalance_frequency     : int = 3           # months; 0 ⇒ never
    stream_start            : str | None = None       # ISO date
    stream_end              : str | None = None

    numeric_feature_columns : Sequence[str] = field(default_factory=tuple)
    feature_weights         : dict[str, float] = field(
        default_factory=lambda: {
            "grade"     : 1.0,
            "sub_grade" : 1.0,
            "loan_amnt" : 1.0,
            "annual_inc": 1.0,
        }
    )



class Status(str, Enum):
    """Solver status codes used across the anticlustering package."""

    optimal   = "optimal"
    timeout   = "timeout"
    error     = "error"
    skipped   = "skipped"
    heuristic = "heuristic"
    stopped   = "stopped"
    solved    = "solved"

    # -------- convenience helpers ------------------------------------
    @classmethod
    def from_string(cls, value: str) -> "Status":
        """Coerce an arbitrary string into a Status enum (raises on unknown)."""
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown status '{value}'. Valid choices: {valid}") from exc

    @classmethod
    def choices(cls) -> list[str]:
        """Return the plain-string choices – useful for CLI or argparse."""
        return [m.value for m in cls]
    
