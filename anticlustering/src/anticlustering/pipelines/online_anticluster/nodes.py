
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
from typing import Dict, List, Sequence

import pandas as pd
import numpy as np

from ...loan.loan import LoanRecord, LoanRecordFeatures
from ...loan.vectorizer import LoanVectorizer
from ...streaming.stream import StreamEngine
from ...streaming.stream_manager import AnticlusterManager
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


def update_anticlusters(
    loans: List[LoanRecord],
    events_df: pd.DataFrame,
    k_groups: int,
    kaggle_cols: Dict[str, List[str]],
    hard_balance_cols: Sequence[str],
    size_tolerance: int,
    rebalance_frequency: int,
    metrics_cat_cols: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Drive **AnticlusterManager** month-by-month, record group assignments and
    quality metrics.

    Returns a dict of DataFrames so Kedro can wire them to two distinct
    catalog entries.
    """
    vectorizer = LoanVectorizer.fit(
        loans,
        kaggle_cols
    )
    # ---- prep ----------------------------------------------------------- #
    mgr = AnticlusterManager(
        k=k_groups,
        vectorizer=vectorizer,
        hard_balance_cols=hard_balance_cols,
        size_tolerance=size_tolerance,
    )
    loan_map: Dict[str, LoanRecord] = {lo.loan_id: lo for lo in loans}

    assignments_rows: List[dict] = []
    metrics_rows: List[dict] = []

    # Shouldn't we extract all loans that depart in the events_df but have never arrived?
    # We then use the exchange heuristic to assign them to group and use this as our initial allocation.
    # The reason for this, is that we now have a set of loans that are not assigned to any group, 
    # but they do somehow depart bcs they expire.


    # ---- main loop ------------------------------------------------------ #
    for row_idx, row in events_df.sort_values("date").iterrows():
        date = LoanRecord._parse_date(row["date"])

        # ---------- normalise the two list columns -----------------
        arrival_ids = row["arrivals_ids"]
        if isinstance(arrival_ids, str):
            arrival_ids = ast.literal_eval(arrival_ids) or []   # "" → []
        departure_ids = row["departures_ids"]
        if isinstance(departure_ids, str):
            departure_ids = ast.literal_eval(departure_ids) or []
        # -----------------------------------------------------------

        # ---------- use the normalised variables; DO NOT touch row[...] again ---
        arrivals   = [loan_map[lid] for lid in arrival_ids]
        departures = [loan_map[lid] for lid in departure_ids]
        #departures = departure_ids

        # ----- arrivals ----- #
        # Process arrivals all at once
        if arrivals:
            mgr.add_loans(arrivals)


        # ----- departures ----- #
        if departures:
            mgr.remove_loans(departures)
        
        # ----- rebalance yearly ----- #TODO: Can probably be removed bcs we rebalance if threshold is exceeded.
        if rebalance_frequency > 0 and (row_idx % rebalance_frequency == 0):
            swaps = mgr.rebalance()
            _LOG.info(
                "Rebalancing at %s (row %d). Number of swaps perfomed: %d", date, row_idx, swaps
            )

        # rebalance if the groupsizes differ more than the tolerance
        min_group_size = min(mgr.group_sizes())
        max_group_size = max(mgr.group_sizes())
        _LOG.info(
            "update_anticlusters: Group sizes at %s (row %d): min: %d, max: %d",
            date,
            row_idx,
            min_group_size,
            max_group_size,
        )
        if max_group_size - min_group_size > size_tolerance:
            initial_sizes = mgr.group_sizes()
            swaps = mgr.rebalance()
            _LOG.info(
                "Rebalancing at %s (row %d) due to imbalance of groups (min: %d, max: %d). Number of swaps perfomed: %d",
                date,
                row_idx,
                min_group_size,
                max_group_size,
                swaps,
            )
            _LOG.info(
                "update_anticlusters: Rebalanced - initial group sizes: %s. Final group sizes: %s",
                initial_sizes,
                mgr.group_sizes()
            )
            

        # ----- record assignments (long) ----- #
        for g_idx, member_ids in mgr.snapshot().items():
            for lid in member_ids:
                assignments_rows.append(
                    {"date": date, "loan_id": lid, "group": g_idx}
                )

        # ----- compute metrics (wide) ----- #
        metrics_row = {
            "date": date,
            "group_sizes": mgr.group_sizes(),
            "within_var": within_group_variance(mgr, loan_map),
            "group_centroids": mgr.group_centroids(),
        }
        for cat in metrics_cat_cols:
            metrics_row[f"balance_{cat}"] = balance_score_categorical(
                mgr, loan_map, cat
            )
        metrics_rows.append(metrics_row)

    # ---- assemble outputs ---------------------------------------------- #
    df_assign = pd.DataFrame(assignments_rows)
    df_metrics = pd.DataFrame(metrics_rows)

    # tidy the list columns for easier downstream use
    df_metrics["group_sizes"] = df_metrics["group_sizes"].apply(np.array)

    return [df_assign, df_metrics]


