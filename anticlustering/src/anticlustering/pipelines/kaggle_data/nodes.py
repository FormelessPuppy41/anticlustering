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


from ...loan.loan import LoanRecord, LoanStatus
from ...streaming.simulator import LoanSimulator
from ...loan.features import parse_raw_row

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



from ...loan.preprocessor import parse_kaggle_dataframe
def parse_kaggle_data(
    df_raw,
    kaggle_columns : dict[str, Any],
    reduce_n       : int | None = None,
    rng_number     : int = 42,
):
    """
    Thin, irreversible parsing only – no scaling or log transforms.
    """
    if reduce_n:
        prev_len = len(df_raw)
        df_raw = _reduce_sample(df_raw, n=int(reduce_n), rng_number=rng_number)
        _LOG.info("Reduced Kaggle data to %d rows from %s rows (random sampling)", len(df_raw), prev_len)
    
    keep_cols        = kaggle_columns.get("keep_columns")            or []
    percent_cols     = kaggle_columns.get("percentage_columns")      or []
    term_cols        = kaggle_columns.get("special_numeric_columns") or []
    date_cols        = kaggle_columns.get("date_columns")            or []
    log_cols         = kaggle_columns.get("log_numeric_columns")     or []
    log_cols         = []
    ordinal_cols     = kaggle_columns.get("ordinal_columns")         or {}
    categorical_cols = kaggle_columns.get("categorical_columns")     or []
    passthrough_cols = kaggle_columns.get("passthrough_columns")     or []

    return parse_kaggle_dataframe(
        df_raw,
        keep_cols       = keep_cols,
        percentage_cols = percent_cols,
        term_cols       = term_cols,
        date_cols       = date_cols,
        log_numeric_cols= log_cols,
        ordinal_cols    = ordinal_cols,
        categorical_cols= categorical_cols,
        passthrough_cols= passthrough_cols,
        fill_numeric_nan= 0.0,
    )
    print(df)
    return df


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
    records : list[LoanRecord] = []
    errors  : list[tuple[str, Exception]] = []
    for item in parsed:
        if isinstance(item, LoanRecord):
            records.append(item)
        else:
            errors.append(item)

    if errors:
        _LOG.warning("Encountered %d errors while parsing rows: %s", len(errors), errors[:5])

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


