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
import logging

from collections import Counter

from .stream_manager import AnticlusterManager
from ..loan.vectorizer import LoanVectorizer
from ..loan.loan import LoanRecord


_LOG = logging.getLogger(__name__)


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
    loans_by_id: dict[str, "LoanRecord"]
) -> float:
    """
    Average within-group variance of the feature vectors.

    Parameters
    ----------
    manager
        The current AnticlusterManager (has groups + membership).
    loans_by_id
        Mapping {loan_id -> LoanRecord}.  Needed because
        `manager._index[lid]` stores only the group number.
    feat_fn
        Function that converts a LoanRecord to a numeric 1-D vector.
    """
    vectorizer = manager.vectorizer
    all_var: list[np.ndarray] = []
    _LOG.info(
        "within_group_variance: computing variance for %d groups",
    )
    for grp in manager._groups:
        if grp.size == 0:
            continue

        member_vecs = vectorizer.transform([loans_by_id[lid] for lid in grp.members])
        all_var.append(np.var(member_vecs, axis=0).mean())

    return float(np.mean(all_var)) if all_var else 0.0


# --------------------------------------------------------------------------- #
#                         -----  Quick group summary  -----                   #
# --------------------------------------------------------------------------- #


def group_summary(
    manager         : AnticlusterManager,
    loans_by_id     : Dict[str, LoanRecord],
    cat_cols        : Sequence[str] | None                  = None
) -> pd.DataFrame:
    """
    Return a DataFrame with *one row per group* summarising:

    • size  
    • centroid of numeric features  
    • share of each categorical value (optional)  

    This is handy for quick eyeballing in a notebook or logging snapshots.
    """
    cat_cols = tuple(cat_cols or [])
    vec_dim = manager.vectorizer.transform(
        [next(iter(loans_by_id.values()))]
    ).shape[1]
    
    records: List[dict] = []
    for g_idx, grp in enumerate(manager._groups):
        row: dict = {"group": g_idx, "size": grp.size}

        # numeric centroid
        if grp.centroid is not None:
            row.update({f"centroid_{d}": grp.centroid[d] for d in range(vec_dim)})
        else:
            row.update({f"centroid_{d}": np.nan for d in range(vec_dim)})

        # categorical proportions
        for col in cat_cols:
            values = [getattr(loans_by_id[lid], col) for lid in grp.members]
            if values:
                counts = Counter(values)
                total = sum(counts.values())
                for val, cnt in counts.items():
                    row[f"{col}={val}"] = cnt / total
        records.append(row)

    return pd.DataFrame(records).fillna(0.0)
