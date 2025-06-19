# src/anticlustering/metrics/quality_metrics.py

"""
Diagnostic helpers for evaluating **online anticlustering quality** at any
time-point. They integrate with the StreamingDataStore + AnticlusterManager
architecture: grouping state is tracked in manager._groups, and raw data in
manager.store.
"""

from __future__ import annotations
import math
from typing import Dict, List, Sequence, Any

import numpy as np
import pandas as pd
import logging
from collections import Counter

from .data_store import StreamingDataStore
from ..core.streaming.stream_manager import AnticlusterManager
from ..core.loans.loan import LoanRecord

_LOG = logging.getLogger(__name__)


def balance_score_categorical(
    manager     : AnticlusterManager,
    loans_by_id : Dict[str, LoanRecord],
    col         : str,
) -> float:
    """
    Gini‐style dispersion score ∈ [0,1] for how evenly the values of a
    categorical attribute are spread across the K clusters.

    0 ⇒ perfect balance; 1 ⇒ maximal imbalance.
    """
    # number of groups
    K = len(manager._groups)

    # tally counts per category per group
    categories: Dict[Any, List[int]] = {}
    for gi, group in enumerate(manager._groups):
        for lid in group.members:
            val = getattr(loans_by_id[lid], col)
            if val not in categories:
                categories[val] = [0] * K
            categories[val][gi] += 1

    # compute per‐category Gini normalized by (1 - 1/K)
    scores: List[float] = []
    for counts in categories.values():
        total = sum(counts)
        if total == 0:
            continue
        p = np.array(counts, dtype=float) / total
        # Gini: ½ Σ_i Σ_j |p_i - p_j|
        gini = 0.5 * np.sum(np.abs(p[:, None] - p[None, :]))
        max_gini = 1.0 - 1.0 / K
        scores.append(gini / max_gini if max_gini > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def within_group_variance(
    manager     : AnticlusterManager,
    loans_by_id : Dict[str, LoanRecord]
) -> float:
    """
    Average within‐group variance of the numeric feature vectors.
    """
    vec = manager.store  # StreamingDataStore
    variances: List[float] = []

    for group in manager._groups:
        if group.size == 0:
            continue
        records = [loans_by_id[lid] for lid in group.members]
        X = vec.vectorizer.transform(records)
        var = np.var(X, axis=0).mean()
        variances.append(var)

    if not variances:
        return 0.0

    avg_var = float(np.mean(variances))
    if math.isnan(avg_var) or avg_var < 0:
        _LOG.warning("within_group_variance: invalid value %r, returning 0", avg_var)
        return 0.0
    return avg_var


def group_summary(
    manager      : AnticlusterManager,
    loans_by_id  : Dict[str, LoanRecord],
    cat_cols     : Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Return a DataFrame, one row per group, with:
      - 'size'
      - numeric centroid components 'centroid_0', 'centroid_1', …
      - categorical proportions for each col=value
    """
    cat_cols = cat_cols or []
    vec = manager.store
    # determine vector dimension
    # take an arbitrary loan to compute total dims
    sample = next(iter(loans_by_id.values()), None)
    if sample:
        total_dim = vec.vectorizer.transform([sample]).shape[1]
    else:
        total_dim = 0

    rows: List[Dict[str, Any]] = []
    for gi, group in enumerate(manager._groups):
        row: Dict[str, Any] = {"group": gi, "size": group.size}

        # numeric centroid
        if group.centroid is not None:
            for d in range(total_dim):
                row[f"centroid_{d}"] = float(group.centroid[d])
        else:
            for d in range(total_dim):
                row[f"centroid_{d}"] = np.nan

        # categorical proportions
        for col in cat_cols:
            vals = [getattr(loans_by_id[lid], col) for lid in group.members]
            if not vals:
                continue
            cnt = Counter(vals)
            total = sum(cnt.values())
            for val, c in cnt.items():
                row[f"{col}={val}"] = c / total

        rows.append(row)

    return pd.DataFrame(rows).fillna(0.0)
