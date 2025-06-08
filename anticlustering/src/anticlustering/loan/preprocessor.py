"""
Vectorised *parsing-only* clean-up for Lending-Club CSV rows.
No scaling, no log transforms – irreversible cleaning only.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# --------------------------------------------------------------------- #
#                     vectorised helper functions                       #
# --------------------------------------------------------------------- #
def parse_percent(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.rstrip("%"), errors="coerce") / 100.0

def parse_term(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.str.extract(r"(\d+)")[0], errors="coerce")

def parse_date(col: pd.Series) -> pd.Series:
    """
    Parse a mixed string column → *first-of-month* `datetime.date`.

    Examples
    --------
    'Jan-2017'  → 2017-01-01
    '2017-09-01'→ 2017-09-01
    'Jun 2020'  → 2020-06-01
    """
    ts = pd.to_datetime(
        col,
        errors="coerce",
        infer_datetime_format=True,   # auto-detect ISO / '%b-%Y' / '%b %Y'
    )
    # normalise to first of month so comparisons are simple equality
    return ts.dt.to_period("M").dt.to_timestamp().dt.date


def parse_log_numeric(col: pd.Series) -> pd.Series:
    col = pd.to_numeric(col, errors="coerce")
    return np.log1p(col)  # log1p handles log(0) correctly as 0.0

def parse_ordinal(col: pd.Series, ordinal_values: list) -> pd.Series:
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


def parse_categorical(col: pd.Series) -> pd.Categorical:
    """
    Convert a column to categorical type.
    """
    return pd.Categorical(col)

# --------------------------------------------------------------------- #
#                          public entry point                           #
# --------------------------------------------------------------------- #
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
        df[c] = parse_percent(df[c])

    for c in term_cols:
        df[c] = parse_term(df[c])

    for c in date_cols:
        df[c] = parse_date(df[c])

    for c in log_numeric_cols:
        df[c] = parse_log_numeric(df[c])

    for c, ordinal_value in ordinal_cols.items():
        df[c] = parse_ordinal(df[c], ordinal_value)

    for c in categorical_cols:
        df[c] = parse_categorical(df[c])

    # minimal NaN handling so downstream objects receive proper floats
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(fill_numeric_nan) 
    #TODO: consider using a more sophisticated NaN handling strategy. e.g. imputation, etc. 

    return df
