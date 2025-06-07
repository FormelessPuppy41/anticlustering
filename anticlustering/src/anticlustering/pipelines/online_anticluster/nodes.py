
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

import datetime as _dt
import logging
from typing import List

import pandas as pd
from kedro.pipeline import Pipeline, node

from ...lending_club.core.loan import LoanRecord, LoanRecordFeatures
from ...lending_club.core.stream import StreamEngine

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                               Node function                                 #
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

    log.info(
        "Simulating stream from %s to %s over %d loans",
        start or "min(issue_d)",
        end or "final departure",
        len(loans),
    )
    log.info("Example loans: %s", loans[0] if loans else "No loans provided")

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
    log.info("Generated %d monthly event rows", len(df_events))
    log.info("Sample events:\n%s", df_events.head(48))
    return df_events


"""
pipelines/update_anticluster.py
===============================

Pipeline #4 ― *Maintain online anticlusters & compute quality metrics*

Inputs
------
* ``loans_raw``             : List[LoanRecord]         (from ingest)
* ``stream_monthly_events`` : pandas.DataFrame         (from simulate_stream)

Parameters (conf/*/parameters.yml)
----------------------------------
k_groups                  : 4
hard_balance_cols         : ["grade"]          # perfectly balanced
size_tolerance            : 1
rebalance_frequency       : 3                  # months; 0 ⇒ never
metrics_cat_cols          : ["grade", "purpose"]

Outputs
-------
* ``anticluster_assignments`` : pandas.DataFrame  (long format)
* ``anticluster_metrics``     : pandas.DataFrame  (wide format)

"""

import ast
import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from kedro.pipeline import Pipeline, node

from ...lending_club.core.anticluster import AnticlusterManager
from ...lending_club.core.loan import LoanRecord
from ...lending_club.core.quality_metrics import (
    balance_score_categorical,
    within_group_variance,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                               Node function                                 #
# --------------------------------------------------------------------------- #


def update_anticlusters(
    loans: List[LoanRecord],
    events_df: pd.DataFrame,
    k_groups: int,
    kaggle_cols: List[str],
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
    # ---- prep ----------------------------------------------------------- #
    loan_map: Dict[str, LoanRecord] = {lo.loan_id: lo for lo in loans}
    mgr = AnticlusterManager(
        k=k_groups,
        numeric_feature_cols=LoanRecordFeatures.numeric_fields(),
        hard_balance_cols=hard_balance_cols,
        size_tolerance=size_tolerance,
    )

    assignments_rows: List[dict] = []
    metrics_rows: List[dict] = []

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
        departures = departure_ids

        # ----- arrivals ----- #
        # Process arrivals all at once
        if arrivals:
            mgr.add_loans(arrivals)


        # ----- departures ----- #
        for lid in departures:
            try:
                mgr.remove_loan(lid)
            except KeyError:
                log.warning("Departure %s not in active pool (date=%s)", lid, date)

        # ----- optional rebalance ----- #
        if rebalance_frequency > 0 and (row_idx % rebalance_frequency == 0):
            mgr.rebalance()

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


