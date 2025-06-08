"""
online.py
=========

Thin adapter that makes the *online / streaming* anticluster behave like any
other solver in your replication framework.

Key traits
----------
* Inherits from `AntiCluster` (your existing abstract base class).
* Consumes an `OnlineConfig` dataclass that mirrors the style of `ILPConfig`.
* Implements the familiar `fit(X, D=None)` contract, but expects **X to be
  a list of `LoanRecord`s** (or a DataFrame convertible to that list).
* Publishes the usual post-fit artefacts:
    • `labels_`      – final group assignment per loan  
    • `timeline_`    – DataFrame of month-by-month assignments (optional)  
    • `score_`       – heterogeneity score (placeholder)  
    • `runtime_`     – elapsed wall-clock seconds  
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .base import AntiCluster, Status
from ..core import register_solver  # <- decorator to register this solver
from ._config import OnlineConfig  # <- add this dataclass to your config.py
from ..loan.loan import LoanRecord
from ..streaming.stream import StreamEngine
from ..streaming.stream_manager import AnticlusterManager
from ..streaming.quality_metrics import within_group_variance

from sklearn.preprocessing import StandardScaler



_LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                      -----  OnlineAntiCluster class  -----                  #
# --------------------------------------------------------------------------- #
@register_solver("online")
class OnlineAntiCluster(AntiCluster):
    """
    Online anticlustering solver that processes loans in a streaming fashion.

    This class inherits from `AntiCluster` and implements the `fit` method to
    perform anticlustering on a stream of `LoanRecord` objects. It manages the
    addition and removal of loans dynamically, allowing for real-time updates
    to the clustering solution.

    Parameters
    ----------
    config : OnlineConfig
        Configuration object containing parameters for the online anticluster,
        such as the number of clusters, start and end dates for the stream,
        and other tuning knobs.

    Attributes
    ----------
    config : OnlineConfig
        The configuration for the online anticluster, including parameters like
        `n_clusters`, `stream_start`, `stream_end`, and others.
    _timeline : pd.DataFrame | None
        A DataFrame that records the month-by-month assignment of loans to groups.
    _manager : AnticlusterManager | None
        The manager responsible for handling the loans and their clustering.
    _id_index : dict[str, int]
        A mapping from loan IDs to their respective row indices in the input data,
        used for quick lookups during the streaming process.
    """
    def __init__(self, config: OnlineConfig, scaler: StandardScaler | None = None) -> None:
        super().__init__(config)

        # overwrite for class
        self.config         : OnlineConfig = config

        self.scaler         = scaler
        self.weights = np.array(
            [config.feature_weights.get(col, 1.0) for col in config.numeric_feature_columns],
            dtype=float
        )

        self._manager       : AnticlusterManager | None = None
        self._id_index      : dict[str, int] = {}          # loan_id → row index in X
        
        # public after fit():
        self._timeline      : pd.DataFrame | None = None

    # ---- mandatory API  ---- #
    def fit(
            self, 
            X: Iterable[LoanRecord] | pd.DataFrame, 
            D=None
        ) -> "OnlineAntiCluster":  # noqa: N802
        """
        Run the streaming anticlustering end-to-end *inside* this call so that
        timing & scoring remain comparable to ILP / exchange solvers.

        Parameters
        ----------
        X
            Preferably a list of LoanRecord objects; if a DataFrame is passed,
            it is converted via `LoanRecord.from_dict`.
        D
            Ignored; present only to satisfy the base-class signature.
        """
        # --- 1. normalise input to List[LoanRecord] --------------------- #
        loans = self._coerce_to_loans(X)
        
        numeric_cols = self.config.numeric_feature_columns
        N = len(loans)
        F = len(numeric_cols)
        all_raw = np.zeros((N, F), dtype=float)

        for i, loan in enumerate(loans):
            all_raw[i, :] = np.array([
                getattr(loan, col, 0.0) for col in numeric_cols
            ], dtype=float)

        all_scaled = self.scaler.transform(all_raw)

        scaled_map: dict[str, np.ndarray] = {
            loan.loan_id: all_scaled[i, :]
            for i, loan in enumerate(loans)
        }

        def feature_fn(loan: LoanRecord) -> np.ndarray:
            """
            Extracts the features from a LoanRecord instance, applying scaling
            if configured. The features are expected to be numeric and are
            returned as a NumPy array.

            Parameters
            ----------
            loan : LoanRecord
                The loan record from which to extract features.

            Returns
            -------
            np.ndarray
                Scaled feature vector for the loan.
            """
            scaled_vec = scaled_map[loan.loan_id]
            return scaled_vec * self.weights

        # --- 2. build stream + anticluster objects ---------------------- #
        stream = StreamEngine(
            loans,
            start_date  =self._parse_date(self.config.stream_start),
            end_date    =self._parse_date(self.config.stream_end),
        )
        self._manager = AnticlusterManager(
            k                   =self.config.n_clusters,
            hard_balance_cols   =self.config.hard_balance_cols,
            size_tolerance      =self.config.size_tolerance,
            feature_fn          =feature_fn
        )
        start_ts = time.time()

        # --- 3. main loop (synchronous) -------------------------------- #
        timeline_rows: list[dict] = []
        for date, arrivals, departures in stream.run():
            # arrivals
            for lo in arrivals:
                self._manager.add_loan(lo)
            # departures
            for lid in departures:
                self._manager.remove_loan(lid)
            # optional periodic repair
            if self.config.rebalance_frequency and date.month % self.config.rebalance_frequency == 0:
                self._manager.rebalance()
            # record snapshot
            for gid, member_ids in self._manager.snapshot().items():
                for lid in member_ids:
                    timeline_rows.append({"date": date, "loan_id": lid, "group": gid})

        run_time = time.time() - start_ts
        self._timeline = pd.DataFrame(timeline_rows)

        # --- 4. final outputs ------------------------------------------ #
        final_groups    = self._manager.snapshot()
        self._set_labels(self._labels_from_snapshot(final_groups, loans))
        self._set_score(within_group_variance(self._manager))
        self._set_runtime(run_time)
        self._set_status(Status.solved)

        return self  # consistent with scikit-learn style

    @property
    def timeline_(self) -> pd.DataFrame:
        """
        DataFrame with columns *date*, *loan_id*, *group* that shows the
        month-by-month assignment of loans to groups.
        
        Returns
        -------
        pd.DataFrame
            Timeline of group assignments.
        """
        if self._timeline is None:
            raise RuntimeError("Call `.fit()` first!")
        return self._timeline
    
    # ---- helper utilities --------------------------------------------- #

    def _coerce_to_loans(
            self, 
            X: Iterable[LoanRecord] | pd.DataFrame
        ) -> List[LoanRecord]:
        """
        Convert the input *X* to a list of LoanRecord objects if it is not
        already one. This is necessary because the OnlineAntiCluster expects
        a list of LoanRecord instances to work with, preserving the order of
        the input data. If a DataFrame is provided, it is converted to LoanRecord
        instances using the `LoanRecord.from_dict` method.

        Parameters
        ----------
        X
            Either an iterable of LoanRecord objects or a pandas DataFrame
            convertible to LoanRecord objects.

        Returns
        -------
        List[LoanRecord]
            List of LoanRecord instances, preserving the order of input.
        """
        if isinstance(X, pd.DataFrame):
            return [LoanRecord.from_dict(rec) for rec in X.to_dict(orient="records")]
        if isinstance(X, list) and isinstance(X[0], LoanRecord):
            # already a list of LoanRecord instances
            return X
        raise TypeError(
            f"Expected an iterable of LoanRecord instances or a DataFrame, "
            f"got {type(X)} instead."
        )

    @staticmethod
    def _parse_date(date_str: str | None):
        """
        Parse a date string in ISO format (e.g., "2023-10-01") into a
        Python date object. If the input is None, return None.

        Parameters
        ----------
        date_str : str | None
            The date string to parse. If None, returns None.

        Returns
        -------
        date | None
            Parsed date object or None if input is None.
        """
        import datetime as _dt, dateutil.parser as _p
        return _p.isoparse(date_str).date() if date_str else None

    def _labels_from_snapshot(
        self, 
        snapshot    : dict[int, list[str]], 
        loans       : list[LoanRecord]
    ) -> np.ndarray:
        """
        Convert the snapshot of group assignments into a NumPy array of labels.
        Each loan's group assignment is determined by looking up its `loan_id`
        in the snapshot dictionary, which maps group IDs to lists of loan IDs.
        If a loan's ID is not found in the snapshot, it is assigned to group -1.

        Parameters
        ----------
        snapshot : dict[int, list[str]]
            A dictionary where keys are group IDs and values are lists of loan IDs
            assigned to those groups.
        loans : list[LoanRecord]
            List of LoanRecord objects for which to generate group labels.

        Returns
        -------
        np.ndarray
            An array of integers where each element corresponds to the group ID
            of the respective loan in the input list. If a loan's `loan_id` is not
            found in the snapshot, it is assigned a group ID of -1.
        """
        id2group = {
            lid: gid 
            for gid, lids in snapshot.items() 
            for lid in lids
        }
        return np.array([id2group.get(lo.loan_id, -1) for lo in loans], dtype=int)
