"""
This is a boilerplate pipeline 'kaggle_data'
generated using Kedro 0.19.13
"""
import kagglehub
import os
import pandas as pd
import logging
from typing import List, Optional, Any
import datetime as _dt
import joblib

from ...preprocessing.online_data import preprocess_node
from ...lending_club.core.loan import LoanRecord, LoanStatus
from ...lending_club.core.simulator import LoanSimulator
from ...lending_club.core.features import parse_raw_row

_LOG = logging.getLogger(__name__)

def load_kaggle_data(name: str):
    kagglehub.login()
    # Download latest version
    path = kagglehub.dataset_download("beatafaron/loan-credit-risk-and-population-stability")

    # Load CSV
    csv_file = os.path.join(path, name)
    df = pd.read_csv(csv_file)
    return df


#FIXME: Remove this node and use the reduce_n parameter in the process_kaggle_data node instead
def create_test_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a test version of the Kaggle data for quick checks."""
    # Example: take the first 100 rows for testing
    return df.head(100).copy()


def _reduce_sample(
        df          : pd.DataFrame, 
        n           : int = 100, 
        rng_number  : int = 42
    ) -> pd.DataFrame:
    """Reduce the DataFrame to a sample of n rows."""
    if len(df) > n:
        return df.sample(n=n, random_state=rng_number).reset_index(drop=True)
    return df.copy()

def process_kaggle_data(
        df              : pd.DataFrame, 
        kaggle_columns  : dict[str, Any],
        reduce_n        : int | None = None,
        scale           : bool = False,
        rng_number      : int = 42
    ) -> pd.DataFrame:
    """Clean raw Kaggle LendingClub data and (optionally) sample+scale it.

    Parameters
    ----------
    kaggle_columns
        Dict loaded from YAML with keys:
        ``percentage_columns``, ``log_numeric_columns``,
        ``special_numeric_columns``, ``keep_columns``.
    """
    #  ---- Unpack the kaggle column yml dict --------------------------------------------
    percentage_cols         : list  = kaggle_columns["percentage_columns"]
    log_numeric_cols        : list  = kaggle_columns["log_numeric_columns"]
    special_numeric_cols    : list  = kaggle_columns["special_numeric_columns"]
    date_cols               : list  = kaggle_columns["date_columns"]
    ordinal_cols            : dict  = kaggle_columns["ordinal_columns"]
    categorical_cols        : list  = kaggle_columns["categorical_columns"]
    passthrough_cols        : list  = kaggle_columns["passthrough_columns"]
    keep_cols               : list  = kaggle_columns["keep_columns"]
    # --------------------------------------------

    if reduce_n == "None" or reduce_n is None or reduce_n == "0" or reduce_n == 0 or reduce_n == "":
        reduce_n = None
    
    if reduce_n is not None:
        prev_len    = len(df)
        df          = _reduce_sample(df, n=int(reduce_n), rng_number=rng_number)
        _LOG.info("Reduced Kaggle data to %d rows from %s rows (random sampling)", len(df), prev_len)
    
    numeric_feature_cols = set(keep_cols.copy()) \
        - set(passthrough_cols) - set(date_cols) - set(percentage_cols) \
        - set(ordinal_cols.keys()) - set(categorical_cols) - set(special_numeric_cols)
    numeric_feature_cols = list(numeric_feature_cols)
    
    return preprocess_node(
        df,
        keep_columns            =keep_cols,
        numeric_feature_columns =numeric_feature_cols,
        scale                   =scale,
        return_df               =True,
        percentage_columns      =percentage_cols,
        date_columns            =date_cols,
        ordinal_columns         =ordinal_cols,
        categorical_columns     =categorical_cols,
        special_numeric_columns =special_numeric_cols,
        log_numeric_columns     =log_numeric_cols,
        passthrough_columns     =passthrough_cols
    )
    


def kaggle_df_to_loan_records(df: pd.DataFrame) -> List[LoanRecord]:
    """
    Convert a Kaggle Lending-Club DataFrame → List[LoanRecord] **fast**.

    Improvements over the earlier loop:
    • Centralises all messy field handling (%, ' 36 months', dates) in
      ``parse_raw_row`` – no duplicated parsing logic here.
    • Uses ``DataFrame.itertuples`` (zero Python dict allocations).
    • Parallel mapping via joblib; falls back gracefully on bad rows.
    """
    if df.empty:
        _LOG.warning("Received empty DataFrame – returning empty list.")
        return []

    # ── helper that swallows bad rows but keeps exception for logging ──
    def _safe_parse(r) -> LoanRecord | Exception:
        try:
            return parse_raw_row(r._asdict())
        except Exception as exc:
            return exc

    # ── parallel map over lightweight named-tuples ─────────────────────
    rows_iter = df.itertuples(index=False, name="LC")
    parsed = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(_safe_parse)(row) for row in rows_iter
    )

    # ── separate successes from failures ───────────────────────────────
    records: list[LoanRecord] = []
    skipped_count = 0
    for item in parsed:
        if isinstance(item, LoanRecord):
            records.append(item)
        else:
            skipped_count += 1
            _LOG.debug("Row skipped: %s", item)   # detailed trace at DEBUG

    if skipped_count:
        _LOG.warning("Skipped %d rows during conversion.", skipped_count)

    _LOG.info("Converted %d rows → LoanRecord objects.", len(records))
    return records


def loan_records_to_long_df(
    loans                       : List[LoanRecord],
    as_of_str                   : Optional[str] = None,
    assume_regular_prepayment   : bool = True,
) -> pd.DataFrame:
    """
    Wrapper around ``LoanSimulator.batch_generate`` that also handles the
    optional *cut-off* date (`as_of`) supplied via YAML parameters.

    Parameters
    ----------
    loans
        Parsed LoanRecord objects from the ingest stage.
    as_of_str
        Optional ISO date string (``YYYY-MM-DD``).  If given, histories are
        truncated at *that* month (inclusive), which is handy when replaying
        year-by-year rather than the full horizon.
    assume_regular_prepayment
        Passed straight through to :py:meth:`LoanSimulator.batch_generate`.

    Returns
    -------
    pd.DataFrame
        Concatenated histories (long format).
    """
    as_of: _dt.date | None = (
        _dt.date.fromisoformat(as_of_str) if as_of_str else None
    )

    _LOG.info(
        "Generating synthetic histories for %d loans … (as_of=%s)",
        len(loans),
        as_of,
    )
    df = LoanSimulator.batch_generate(
        loans, 
        as_of                       =as_of, 
        assume_regular_prepayment   =assume_regular_prepayment
    )
    _LOG.info("→ Produced %d loan-month rows", len(df))
    _LOG.info("Sample of the generated DataFrame:\n%s", df.head(30))
    return df


