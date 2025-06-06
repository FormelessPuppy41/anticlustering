"""
metrics/quality.py
==================

Diagnostic helpers for evaluating **online anticlustering quality** at any
time-point.  They are deliberately light—using only NumPy and Pandas—so they
can run inside unit tests or scheduled monitoring jobs.

Public API
----------

balance_score_categorical(...)
within_group_variance(...)
group_summary(...)

Author:  Your Name <your.email@example.com>
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..core.anticluster import AnticlusterManager
from ..core.features import vectorise as default_feature_vector
from ..core.loan import LoanRecord


# --------------------------------------------------------------------------- #
#                       -----  Categorical balance  -----                     #
# --------------------------------------------------------------------------- #


def balance_score_categorical(
    manager         : AnticlusterManager,
    loans_by_id     : Dict[str, LoanRecord],
    col             : str,
) -> float:
    """
    Gini-style dispersion score ∈ [0, 1] for how *evenly* the distinct values
    of a categorical variable are spread across K anticlusters.

    • 0  ⇒ perfect balance (every group has exactly the overall proportion)  
    • 1  ⇒ worst imbalance (all instances of at least one category sit in a
           single group)

    Parameters
    ----------
    manager
        AnticlusterManager holding current partition.
    loans_by_id
        Mapping “loan_id → LoanRecord” for quick look-ups.
    col
        Attribute name of the LoanRecord you’d like to check (must be hashable).

    Returns
    -------
    float
        Balance score (lower = better).
    """
    snapshot = manager.snapshot()  # {group_idx: [loan_id, …]}
    categories: Dict[str, List[int]] = {}  # cat_value → counts per group

    # For each group, count occurrences of each category value
    # (e.g. { "A": [10, 5, 0], "B": [0, 2, 8] } for 3 groups)
    for g_idx, ids in snapshot.items():
        for lid in ids:
            cat_val = getattr(loans_by_id[lid], col)
            if cat_val not in categories:
                categories[cat_val] = [0] * manager.k
            categories[cat_val][g_idx] += 1

    # For each category, compute normalised variance of its distribution
    disp_scores = []
    for counts in categories.values():
        total = sum(counts)
        if total == 0:
            continue
        probs = np.array(counts) / total
        # Gini-style measure: ½ Σ_i Σ_j |p_i − p_j|
        gini = 0.5 * np.sum(np.abs(probs.reshape(-1, 1) - probs))
        # Normalise by max possible dispersion (1 − 1/K)
        max_disp = 1.0 - 1.0 / manager.k
        disp_scores.append(gini / max_disp)

    return float(np.mean(disp_scores)) if disp_scores else 0.0


# --------------------------------------------------------------------------- #
#             -----  Within-group variance of numeric features  -----         #
# --------------------------------------------------------------------------- #

def within_group_variance(
    manager: "AnticlusterManager",
    loan_lookup: dict[str, "LoanRecord"],
    feat_fn: callable = default_feature_vector,
) -> float:
    """
    Average within-group variance of the feature vectors.

    Parameters
    ----------
    manager
        The current AnticlusterManager (has groups + membership).
    loan_lookup
        Mapping {loan_id -> LoanRecord}.  Needed because
        `manager._index[lid]` stores only the group number.
    feat_fn
        Function that converts a LoanRecord to a numeric 1-D vector.
    """
    all_vecs: list[np.ndarray] = []

    for grp in manager._groups:
        if grp.size == 0:
            continue

        vecs = np.stack(
            [feat_fn(loan_lookup[lid]) for lid in grp.members]   # <-- FIX
        )
        all_vecs.append(vecs)

    if not all_vecs:
        return 0.0

    variances = [np.var(v, axis=0).mean() for v in all_vecs]
    return float(np.mean(variances))


# --------------------------------------------------------------------------- #
#                         -----  Quick group summary  -----                   #
# --------------------------------------------------------------------------- #


def group_summary(
    manager         : AnticlusterManager,
    loans_by_id     : Dict[str, LoanRecord],
    cat_cols        : Sequence[str] | None                  = None,
    feat_fn         : Callable[[LoanRecord], np.ndarray]    = default_feature_vector,
) -> pd.DataFrame:
    """
    Return a DataFrame with *one row per group* summarising:

    • size  
    • centroid of numeric features  
    • share of each categorical value (optional)  

    This is handy for quick eyeballing in a notebook or logging snapshots.
    """
    cat_cols = tuple(cat_cols or [])
    feat_dim = len(feat_fn(next(iter(loans_by_id.values()))))

    records = []
    for g_idx, grp in enumerate(manager._groups):
        row = {"group": g_idx, "size": grp.size}
        # numeric centroid
        if grp.centroid is not None:
            for d in range(feat_dim):
                row[f"centroid_{d}"] = grp.centroid[d]
        else:
            for d in range(feat_dim):
                row[f"centroid_{d}"] = np.nan

        # categorical proportions
        for col in cat_cols:
            values = [getattr(loans_by_id[lid], col) for lid in grp.members]
            if values:
                counts = pd.Series(values).value_counts(normalize=True)
                for cat_val, prop in counts.items():
                    row[f"{col}={cat_val}"] = prop
        records.append(row)

    return pd.DataFrame(records).fillna(0.0)
