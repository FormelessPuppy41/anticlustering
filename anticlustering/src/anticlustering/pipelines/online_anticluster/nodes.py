
from __future__ import annotations

"""
This is a boilerplate pipeline 'online_anticluster'
generated using Kedro 0.19.13
"""
"""
pipelines/simulate_stream.py
============================

Pipeline #3 ― *Simulate the arrival / departure stream*

Inputs
------
* ``loans_raw``      : List[LoanRecord]     (from ingest pipeline)

Parameters (conf/*/parameters.yml)
----------------------------------
stream_start_date        : "YYYY-MM-DD"  # null ⇒ min(issue_d)
stream_end_date          : "YYYY-MM-DD"  # null ⇒ until last loan departs

Outputs
-------
* ``stream_monthly_events`` : pandas.DataFrame  (one row per calendar month)

Schema of *stream_monthly_events*
---------------------------------
date            datetime64[ns]  (month-end, always the 1st of month)
arrivals_ids    object          (Python list[str])
departures_ids  object          (Python list[str])

Author
------
Your Name  <your.email@example.com>
"""

import ast
import datetime as _dt
import logging
from typing import Dict, List, Sequence, Optional, Any

import pandas as pd
import numpy as np

from ...core.loans.loan import LoanRecord, LoanRecordFeatures
from ...core.loans.vectorizer import LoanVectorizer
from ...core.online._registry import get_online_solver
from ...streaming.stream import StreamEngine
from ...core.streaming.stream_manager import AnticlusterManager
from ...streaming.quality_metrics import (
    balance_score_categorical,
    within_group_variance,
)


_LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                               Node functions                                #
# --------------------------------------------------------------------------- #


def simulate_stream(
    loans: List[LoanRecord],
    stream_start_date: str | None = None,
    stream_end_date: str | None = None,
) -> pd.DataFrame:
    """
    Run `StreamEngine` month-by-month and emit an **event log** DataFrame.

    Parameters
    ----------
    loans
        Parsed LoanRecord list.
    stream_start_date, stream_end_date
        ISO-date strings (``YYYY-MM-DD``) or *null*.

    Returns
    -------
    pd.DataFrame
        One row per calendar month with two list-columns: arrivals_ids,
        departures_ids.  These are consumed by the *update_anticluster*
        pipeline.
    """
    start: _dt.date | None = (
        _dt.date.fromisoformat(stream_start_date) if stream_start_date else None
    )
    end: _dt.date | None = (
        _dt.date.fromisoformat(stream_end_date) if stream_end_date else None
    )

    _LOG.info(
        "simulate_stream: Simulating stream from %s to %s over %d loans",
        start or "min(issue_d)",
        end or "final departure",
        len(loans),
    )
    _LOG.info("simulate_stream: Example loans: %s", loans[0] if loans else "No loans provided")

    engine = StreamEngine(loans, start_date=start, end_date=end)

    records: List[dict] = []
    for date, arrivals, departures in engine.run():
        records.append(
            {
                "date": pd.Timestamp(date),
                "arrivals_ids": [lo.loan_id for lo in arrivals],
                "departures_ids": [lo.loan_id for lo in departures],
            }
        )

    df_events = pd.DataFrame(records)
    _LOG.info("simulate_stream: Generated %d monthly event rows", len(df_events))
    _LOG.info("simulate_stream: Sample events:\n%s", df_events.head(48))
    return df_events






from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ...core.online._config import OnlineExchangeConfig
from ...core.loans.vectorizer import LoanVectorizerConfig

_LOG = logging.getLogger(__name__)


def update_anticlusters(
    loans: List[LoanRecord],
    events_df: pd.DataFrame,
    k: int,
    kaggle_cols: Dict[str, List[str]],
    metrics_cat_cols: List[str],
    hard_balance_cols: List[str] | None = None,
) -> List[pd.DataFrame]:
    """
    Processes a stream of monthly events, returning:
      - df_assign: long table of (date, loan_id, group)
      - df_metrics: wide table of metrics per date
    """

    # Map loan_id → record
    loan_map = {ln.loan_id: ln for ln in loans}
    vector_config = LoanVectorizerConfig(
        kaggle_columns=kaggle_cols,
        num_scaler=StandardScaler(),
        cat_encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False),
    )
    online_exchange_config = OnlineExchangeConfig(n_clusters=k)
    # Initialize manager with chosen solver
    solver = get_online_solver("online_exchange", config=online_exchange_config)  # or pass in solver_name via params
    mgr = AnticlusterManager(
        solver=solver,
        vectorizer_config=vector_config,
        hard_balance_cols=hard_balance_cols
    )

    assignments_rows: List[Dict[str,Any]] = []
    metrics_rows:     List[Dict[str,Any]] = []
    prev_assignment: Dict[str,int] = {}

    # Process each event date in order
    for _, row in events_df.sort_values("date").iterrows():
        date = LoanRecord._parse_date(row["date"])

        # parse arrivals/departures lists
        raw_arr = row["arrivals_ids"]
        arr_ids = ast.literal_eval(raw_arr) if isinstance(raw_arr, str) else row["arrivals_ids"]
        raw_dep = row["departures_ids"]
        dep_ids = ast.literal_eval(raw_dep) if isinstance(raw_dep, str) else row["departures_ids"]

        arrivals   = [loan_map[lid] for lid in arr_ids]
        departures = [loan_map[lid] for lid in dep_ids]

        # feed arrivals
        if arrivals:
            mgr.on_arrival(arrivals)

        # feed departures
        if departures:
            mgr.on_departure(departures)

        # snapshot & record any moves
        current = mgr.get_assignments()  # Dict[str,int]
        for lid, grp in current.items():
            old = prev_assignment.get(lid)
            if old is None or old != grp:
                assignments_rows.append({"date": date, "loan_id": lid, "group": grp})
        prev_assignment = current.copy()

        # compute metrics row
        row_metrics: Dict[str,Any] = {
            "date": date,
            "group_sizes": mgr.group_sizes,
            "within_var": within_group_variance(mgr, loan_map),
            "group_centroids": mgr.centroids,
        }
        for cat in metrics_cat_cols:
            row_metrics[f"balance_{cat}"] = balance_score_categorical(mgr, loan_map, cat)

        metrics_rows.append(row_metrics)

    # assemble outputs exactly as before
    df_assign = pd.DataFrame(assignments_rows)
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics["group_sizes"] = df_metrics["group_sizes"].apply(np.array)

    return [df_assign, df_metrics]
