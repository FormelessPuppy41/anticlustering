"""
core.loan_feature_extractor
---------------------------

Single source of truth for
1. turning a raw Kaggle Lending-Club row → ``LoanRecord``
2. turning a ``LoanRecord``         → numeric feature vector.

Keep this file *pure* (no I/O, no global state) so it can be imported
by both the offline and online solvers without side-effects.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Optional

import numpy as np
import pandas as pd
from dateutil import parser as _p

from sklearn.preprocessing import StandardScaler

from .loan import LoanRecord, LoanStatus
from .utils import _parse_date

# ── helpers ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from typing import Union

# --------------------------------------------------------------------------- #
#                               % parser                                      #
# --------------------------------------------------------------------------- #
def _parse_percent(x: Union[str, float, int, pd.Series]) -> Union[float, pd.Series]:
    """
    Convert percentages to a *plain* numeric value in **percent units**.

    • '13.56%'  → 13.56  
    • '13.56'   → 13.56  
    •  0.1356   → 13.56 (scalar or Series)

    Accepts scalars **or** pd.Series.  Series processing is vectorised.
    """
    # ---------------------  vectorised branch  ---------------------------- #
    if isinstance(x, pd.Series):
        s = x.astype(str).str.strip()

        # mark entries that end with '%'
        has_pct = s.str.endswith('%')

        # strip '%' and coerce to numeric (errors→NaN)
        num = pd.to_numeric(s.str.rstrip('%'), errors='coerce')

        # if original had '%': already in percent units
        # else: values ≤1 are interpreted as fractional and multiplied by 100
        num = np.where(has_pct, num, np.where(num <= 1, num * 100, num))

        return pd.Series(num, index=x.index, name=x.name)

    # ---------------------  scalar branch  -------------------------------- #
    if isinstance(x, (int, float)):
        return float(x) * (100 if x <= 1 else 1)
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            return float(x.rstrip('%'))
        try:
            return _parse_percent(float(x))  # recursion on numeric path
        except ValueError:
            raise ValueError(f"Cannot parse percentage from string: {x}")
    raise TypeError(f"Unsupported type: {type(x)}. Expected str, int, float, or pd.Series.")


def _parse_term(x: Union[str, int, pd.Series]) -> Union[int, pd.Series]:
    """
    Convert loan term strings like ' 36 months' to an integer **number of months**.

    Accepts scalars **or** pd.Series.
    """
    if isinstance(x, pd.Series):
        cleaned = (
            x.astype(str)
             .str.extract(r'(\d+)', expand=False)   # keep digits
             .astype(float)                         # NaN→float
             .astype('Int64')                       # optional pandas nullable int
        )
        return cleaned  # Series of ints/NA

    if isinstance(x, int):
        return x
    if isinstance(x, str):
        digits = ''.join(ch for ch in x if ch.isdigit())
        if not digits:
            raise ValueError(f"No digits found in term string: {x}")
        return int(digits)
    raise TypeError(f"Unsupported type: {type(x)}. Expected str, int, or pd.Series.")


def _parse_log_numeric(col: pd.Series) -> pd.Series:
    col = pd.to_numeric(col, errors="coerce")
    return np.log1p(col)  # log1p handles log(0) correctly as 0.0

def _parse_ordinal(col: pd.Series, ordinal_values: list) -> pd.Series:
    """
    Map an ordered category list → numeric codes (float).

    Unknown / unseen labels ⇒ NaN, so they don’t distort scaling stats.
    The returned Series keeps the **original index**.
    """
    cat = pd.Categorical(col, categories=ordinal_values, ordered=True)

    # cat.codes is a NumPy array (int8/16) where unknowns are -1
    codes = pd.Series(cat.codes, index=col.index, dtype="float")
    codes.replace(-1, np.nan, inplace=True)    # unknown → NaN

    return codes


def _parse_categorical(col: pd.Series) -> pd.Categorical:
    """
    Convert a column to categorical type.
    """
    return pd.Categorical(col)


# ── public API ────────────────────────────────────────────────────────────
def parse_raw_row(row: Dict[str, Any]) -> LoanRecord:
    """
    Convert a raw Kaggle row *dict* into a fully-typed ``LoanRecord``.

    Handles the messy fields **term** and **int_rate** here so other
    modules never repeat that parsing work.
    """
    return LoanRecord(
        loan_id=           str(row["id"]),
        loan_amnt=         float(row["loan_amnt"]),
        term=               _parse_term(row["term"]),
        issue_d=            _parse_date(row["issue_d"]),
        int_rate=          _parse_percent(row["int_rate"]),
        grade=             row["grade"],
        sub_grade=         row["sub_grade"],
        last_pymnt_d=       _parse_date(row.get("last_pymnt_d")),
        loan_status=       LoanStatus.from_raw(row["loan_status"]),
        total_rec_prncp=   float(row["total_rec_prncp"]),
        recoveries=        float(row["recoveries"]),
        total_rec_int=     float(row["total_rec_int"]),
        annual_inc=        float(row["annual_inc"]),
    )


def parse_kaggle_dataframe(
    df                : pd.DataFrame,
    *,
    keep_cols         : list[str],
    percentage_cols   : list[str],
    term_cols         : list[str],
    date_cols         : list[str],
    log_numeric_cols  : list[str],
    ordinal_cols      : dict[str, int],
    categorical_cols  : list[str],
    passthrough_cols  : list[str],
    fill_numeric_nan  : float = 0.0,
) -> pd.DataFrame:
    """
    Returns
    -------
    Cleaned copy of *df* – **same columns**, same dtypes as LoanRecord expects.
    """
    # ---------- whitelist & validate columns --------------------------------
    df = df.copy()
    missing = set(keep_cols) - set(df.columns)
    if missing:
        raise ValueError(f"keep_cols missing in DataFrame: {missing}")
    df = df[keep_cols].copy()                    # drop everything else

    declared_cols = (
        percentage_cols
        + term_cols
        + date_cols
        + log_numeric_cols
        + list(ordinal_cols.keys())
        + categorical_cols
        + passthrough_cols
    )
    absent = set(declared_cols) - set(df.columns)
    if absent:
        raise ValueError(f"Declared columns not present in DataFrame: {absent}")

    # ---------- fast vectorised parsing -------------------------------------
    
    for c in percentage_cols:
        df[c] = _parse_percent(df[c])

    for c in term_cols:
        df[c] = _parse_term(df[c])

    for c in date_cols:
        df[c] = _parse_date(df[c])

    for c in log_numeric_cols:
        df[c] = _parse_log_numeric(df[c])

    for c, ordinal_value in ordinal_cols.items():
        df[c] = _parse_ordinal(df[c], ordinal_value)

    for c in categorical_cols:
        df[c] = _parse_categorical(df[c])

    # minimal NaN handling so downstream objects receive proper floats
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(fill_numeric_nan) 
    #TODO: consider using a more sophisticated NaN handling strategy. e.g. imputation, etc. 

    return df
