from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping
from hashlib import blake2b

import numpy as np
import pandas as pd



def _hash_array(arr: np.ndarray) -> str:
    """Hash the *content* (not the id) of an array -> short hex string."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("Expected a numpy array")
    
    arr = np.ascontiguousarray(arr)  # ensure contiguous memory layout
    
    h = blake2b(arr.view(np.uint8), digest_size=8)   # 64-bit hash
    return h.hexdigest()

# ------------------------------------------------------------------
# one run  (labels, score, X-hash)
# ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SolverResult:
    labels   : np.ndarray
    score    : float
    x_hash   : str                       # only the hash is kept, not X itself
    run_time : float = field(default=0.0, repr=False)  # optional runtime

    @classmethod
    def from_run(cls, X: np.ndarray, labels: np.ndarray, score: float, run_time: float) -> "SolverResult":
        return cls(labels=labels, score=score, x_hash=_hash_array(X), run_time=run_time)


# ------------------------------------------------------------------
# many runs that *share the same X*
# ------------------------------------------------------------------
@dataclass(slots=True)
class SolverResults:
    tag        : str                                     # identifier of the dataset
    _results   : Dict[str, SolverResult] = field(default_factory=dict, repr=False)
    _x_hash    : str = field(init=False, repr=False)

    # -------------- constructor helpers -----------------
    def __post_init__(self) -> None:                     # called by dataclass
        if not self._results:
            raise ValueError("Empty results collection")

        # ensure all hashes identical
        hashes = {r.x_hash for r in self._results.values()}
        if len(hashes) != 1:
            raise ValueError("All SolverResult objects must stem from the same X")
        object.__setattr__(self, "_x_hash", hashes.pop())

    # --------------  mapping-like behaviour -------------
    def __getitem__(self, key: str) -> SolverResult:
        """sr['kmeans'] -> SolverResult for that solver."""
        return self._results[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._results)

    def keys(self) -> Iterable[str]:
        return self._results.keys()

    # --------------  nice helpers -----------------------
    def compare_scores(self) -> Dict[str, float]:
        return {name: res.score for name, res in self._results.items()}

    def compare_labels(self) -> Dict[str, np.ndarray]:
        return {name: res.labels for name, res in self._results.items()}

    # table that is easy to persist (Parquet/CSV/…)
    def to_table(self) -> pd.DataFrame:
        """Return long table – one row per solver."""
        return (
            pd.DataFrame([
                {"solver": name,
                 "score" : res.score,
                 "x_hash": res.x_hash,
                 "labels": res.labels}                 # keep labels as object column
                for name, res in self._results.items()
            ])
            .set_index("solver")
        )

    # --------------  factory used from your node --------
    @classmethod
    def from_runs(cls, 
            tag: str,
            runs: Mapping[str, tuple[np.ndarray, float]],
            X:   np.ndarray
        ) -> "SolverResults":
        """
        `runs`   : dict[solver_name -> (labels, score)]
        `X`      : the data that was fed into *all* solvers
        """
        x_hash = _hash_array(X)
        results = {
            name: SolverResult(labels=lab, score=sco, x_hash=x_hash)
            for name, (lab, sco) in runs.items()
        }
        return cls(tag=tag, _results=results)
    
    