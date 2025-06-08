
import datetime as _dt
import math
import numpy as np
import pandas as pd
from dateutil import parser as _p
from dateutil.relativedelta import relativedelta
from typing import Any, Optional, Union, Dict, List

import logging

_LOG = logging.getLogger(__name__)

def _parse_date(val: Union[Any, pd.Series]) -> Union[Optional[_dt.date], pd.Series]:
    """
    Coerce *scalar* **or** *pd.Series* of mixed objects to `datetime.date`.

    • Accepts str (e.g. 'May-2020', '2017-09-01'), datetime, numeric epochs,
      or already-a-date.
    • Vectorised branch for Series → fast, returns Series[date | NaT]
    """

    # -----------------------  VECTORISED (Series)  ------------------------ #
    if isinstance(val, pd.Series):
        s = val

        # If numeric: decide unit column-wise
        if s.dtype.kind in "iuf":                       # int/uint/float
            ns_mask = s > 1e13
            s_mask  = (s > 1e9) & ~ns_mask      # seconds since epoch
            d_mask  = ~(ns_mask | s_mask)       # days   since epoch

            out = pd.Series(index=s.index, dtype="datetime64[ns]")

            if ns_mask.any():
                out.loc[ns_mask] = pd.to_datetime(
                    s.loc[ns_mask].astype("int64"), unit="ns", errors="coerce"
                )
            if s_mask.any():
                out.loc[s_mask] = pd.to_datetime(
                    s.loc[s_mask].astype("int64"), unit="s", errors="coerce"
                )
            if d_mask.any():
                out.loc[d_mask] = pd.Timestamp("1970-01-01") + pd.to_timedelta(
                    s.loc[d_mask].astype(float), unit="D"
                )

            return out.dt.date

        # Non-numeric → let pandas figure it out in vectorised form
        out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        return out.dt.date

    # -----------------------  SCALAR branch  ------------------------------ #
    # already datetime-like
    if isinstance(val, (_dt.date, _dt.datetime, pd.Timestamp)):
        return val.date() if hasattr(val, "date") else val

    # None / NaN
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None

    # numeric epochs
    if isinstance(val, (int, np.integer, float, np.floating)):
        if val > 1e13:
            ts = pd.to_datetime(int(val), unit="ns", errors="coerce")
        elif val > 1e9:
            ts = pd.to_datetime(int(val), unit="s", errors="coerce")
        else:
            ts = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(val))
        return None if pd.isna(ts) else ts.date()

    # strings
    try:
        return _p.parse(str(val)).date()
    except Exception:                                    # pragma: no cover
        _LOG.warning("Unparsable date value: %s", val)
        return None


def _add_months(d: _dt.date, months: int) -> _dt.date:
    """Excel-style month arithmetic that keeps month-ends intuitive."""
    return (d + relativedelta(months=months))

def _raise(msg: str) -> None:                                   # tiny helper
    raise ValueError(msg)
