"""
Turn a batch of LoanRecord objects into a numeric matrix.
Optional: scaling, log1p, weighting – *solver-side* only.
"""

from __future__ import annotations
from typing import Sequence, Callable, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..lending_club.core.loan import LoanRecord

# --------------------------------------------------------------------- #
#                     stateless vectorisation helper                    #
# --------------------------------------------------------------------- #
def vectorise_records(
    records         : Sequence[LoanRecord],
    columns         : Sequence[str],
    *,
    scale           : bool = False,
    log_columns     : Optional[Sequence[str]] = None,
    weights         : Optional[dict[str,float]] = None,
    scaler          : Optional[StandardScaler] = None,
) -> tuple[np.ndarray, StandardScaler | None]:
    """
    Parameters
    ----------
    records      : iterable of LoanRecord
    columns      : fields to extract in order
    scale        : if True, fit/transform with StandardScaler
    log_columns  : subset of *columns* to apply np.log1p
    weights      : optional per-dimension weights AFTER scaling
    scaler       : if provided, **reuse** instead of fitting new

    Returns
    -------
    X        : (n, d) numeric ndarray
    fitted_scaler | None
    """
    arr = np.array([[getattr(r, c) for c in columns] for r in records], dtype=float)

    # log-space where requested
    if log_columns:
        for j, c in enumerate(columns):
            if c in log_columns:
                arr[:, j] = np.log1p(arr[:, j])

    if scale:
        if scaler is None:
            scaler = StandardScaler().fit(arr)
        arr = scaler.transform(arr)
    else:
        scaler = None

    if weights:
        w = np.array([weights.get(c, 1.0) for c in columns])
        arr *= w

    return arr, scaler
