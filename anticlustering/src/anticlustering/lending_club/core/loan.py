from __future__ import annotations

"""loan.py
===========
Domain model for Lending Club loan records.  
Pure business logic – no I/O – so the module is easy to unit‑test and reuse.

Key abstractions
----------------
* **LoanStatus** – Canonical enumeration of final loan outcomes as observed in the LoanStats snapshot.
* **LoanRecord** – Immutable dataclass representing a single loan row plus convenient financial helpers.

Author: Thesis project – Erasmus University Rotterdam
"""
import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from dateutil import parser as _p
from dateutil.relativedelta import relativedelta


_LOG = logging.getLogger(__name__)

__all__ = [
    "LoanStatus",
    "LoanRecord",
]

# ------------------------------------------------------------------------------- #
# Enums
# ------------------------------------------------------------------------------- #
class LoanStatus(str, Enum):
    """Canonical terminal states for a Lending Club loan.
    
    We map the raw LoanStats values to a smaller, analysis‑friendly set.
    """
    CURRENT                 = "Current"  # still active in snapshot (right‑censored)
    FULLY_PAID              = "Fully Paid"
    CHARGED_OFF             = "Charged Off"  # default / loss event

    # TODO:
    # Default has been deprecated in favour of Charged Off.
    # Often only see: "Current", "Fully Paid", "Charged Off".
    # Does not meet the credit policy are legacy loans from pre-2010.
    # Issued, Cancelled, Withdrawn are extremely rare and usually dropped. 
    #
    # Probably best to just drop all but the main three:
    # "Current", "Fully Paid", "Charged Off".
    # !!!! simply remove the rest from the mapping.
    DEFAULT                 = "Default"  # very rare – legacy LC term
    LATE16_30               = "Late (16-30 days)"  # not used in synthetic schedule
    LATE31_120              = "Late (31-120 days)"  # not used in synthetic schedule
    GRACE_PERIOD            = "In Grace Period"  # not used in synthetic schedule
    ISSUED                  = "Issued"  # not used in synthetic schedule
    NO_CREDIT_FULLY_PAID    = "Does not meet the credit policy. Status: Fully Paid"  # not used in synthetic schedule
    NO_CREDIT_CHARGED_OFF   = "Does not meet the credit policy. Status: Charged Off"  # not used in synthetic schedule
    CANCELLED               = "Cancelled"  # not used in synthetic schedule
    WITHDRAWN               = "Withdrawn"  # not used in synthetic schedule
    

    @classmethod
    def from_raw(cls, raw: str) -> "LoanStatus":
        """Map the raw string from LoanStats to :class:`LoanStatus`."""
        mapping = {
            "Fully Paid"                : cls.FULLY_PAID,
            "Charged Off"               : cls.CHARGED_OFF,
            "Current"                   : cls.CURRENT,
            # "Default"                   : cls.CHARGED_OFF,  # treat similarly for amortisation
            # "Late (16-30 days)"         : cls.LATE16_30,
            # "Late (31-120 days)"        : cls.LATE31_120,
            # "In Grace Period"           : cls.GRACE_PERIOD,
            # "Issued"                    : cls.ISSUED,
            # "Does not meet the credit policy. Status: Fully Paid"       : cls.NO_CREDIT_FULLY_PAID,
            # "Does not meet the credit policy. Status: Charged Off"      : cls.NO_CREDIT_CHARGED_OFF,
            # "Cancelled"                 : cls.CANCELLED,
            # "Withdrawn"                 : cls.WITHDRAWN,
        }
        if raw is None or raw == "" or raw not in mapping:
            raise ValueError(f"Unknown loan status: {raw!r}. Valid values: {list(mapping.keys())}")
        return mapping.get(raw, cls.CURRENT)


# ------------------------------------------------------------------------------- #
# Helpers
# ------------------------------------------------------------------------------- #

def _parse_date(val: Any) -> Optional[_dt.date]:
    """Coerce most imaginable inputs → `datetime.date | None`."""
    # already datetime-like
    if isinstance(val, (_dt.date, _dt.datetime, pd.Timestamp)):
        return val.date() if hasattr(val, "date") else val
    # None / NaN
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    # epoch numbers
    if isinstance(val, (int, np.integer, float, np.floating)):
        if val > 1e13: ts = pd.to_datetime(int(val), unit="ns", errors="coerce")
        elif val > 1e9: ts = pd.to_datetime(int(val), unit="s", errors="coerce")
        else:           ts = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(val))
        return None if pd.isna(ts) else ts.date()
    # strings
    try:
        return _p.parse(str(val)).date()
    except Exception:                                           # pragma: no cover
        _LOG.warning("Unparsable date value: %s", val)
        return None


def _add_months(d: _dt.date, months: int) -> _dt.date:
    """Excel-style month arithmetic that keeps month-ends intuitive."""
    return (d + relativedelta(months=months))

def _raise(msg: str) -> None:                                   # tiny helper
    raise ValueError(msg)

# ----------------------------------------------------------------------------
# Core dataclass
# ----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LoanRecord:
    """
    LoanRecord
    ===========
    Immutable representation of a Lending Club loan record with financial helpers.

    If used for anticlustering, the columns of the data should be scaled to avoid
    bias based on the scale of the data. This class is designed to be easily
    instantiated from a *LoanStats* CSV row or a dictionary with the same structure. 
    
    Attributes
    ----------
    loan_id: str
        Unique identifier for the loan.
    loan_amnt: Decimal
        Original loan amount in USD.
    term_months: int
        Scheduled number of monthly payments (36 or 60 for Lending Club).
    issue_date: date
        Date when the loan was issued (in format "Jan-2015").
    int_rate: Decimal   
        Annual nominal interest rate in *percent* (e.g. 13.56).
    last_pymnt_date: Optional[date]
        Date of the last payment made (in format "Jan-2015").
    loan_status: LoanStatus
        Final status of the loan, represented as a :class:`LoanStatus` enum.
    total_rec_prncp: Decimal
        Total principal repaid by the borrower (in USD).
    recoveries: Decimal
        Total amount recovered from the loan after default (in USD).
    total_rec_int: Decimal
        Total interest received from the borrower (in USD).
    """

    loan_id             : str
    loan_amnt           : float
    term_months         : int
    issue_date          : _dt.date
    int_rate            : float  # annual nominal rate in *percent* – e.g. 13.56
    grade               : str  # e.g. "A", "B", "C" etc.
    sub_grade           : str  # e.g. "A1", "B2", "C3" etc.
    last_pymnt_date     : Optional[_dt.date]
    loan_status         : LoanStatus
    total_rec_prncp     : float
    recoveries          : float
    total_rec_int       : float
    annual_inc          : float # annual income of the borrower (in USD)
    # lazily computed cache – excluded from equality / repr
    _monthly_payment    : Decimal | None = field(init=False, default=None, repr=False, compare=False)

    # ── automatic type-coercion ───────────────────────────────────────────
    def __post_init__(self) -> None:
        object.__setattr__(
            self, 
            "issue_date",
             _parse_date(self.issue_date) or _raise("issue_date missing")
            )
        if self.last_pymnt_date:
            object.__setattr__(
                self, 
                "last_pymnt_date", 
                _parse_date(self.last_pymnt_date)
            )
        if self.int_rate > 1.0:
            # convert from percent to decimal
            object.__setattr__(self, "int_rate", self.int_rate / 100.0)

    # ------- Core financial helpers --------------------------------
    
    @property
    def monthly_rate(self) -> float:
        """Monthly nominal rate (decimal, not percent)."""
        return (self.int_rate / 12)

    @property
    def monthly_payment(self) -> Decimal:
        if self._monthly_payment == 0:
            r, n = self.monthly_rate, self.term_months
            pay = (self.loan_amnt / n) if r == 0 else self.loan_amnt * r * (1 + r) ** n / ((1 + r) ** n - 1)
            object.__setattr__(self, "_monthly_payment", Decimal(str(round(pay, 2))))
        return self._monthly_payment


    
    #  ----- Lifecycle helpers --------------------------------
    @property
    def maturity_date(self) -> _dt.date:
        return _add_months(self.issue_date, self.term_months)


    @property
    def departure_date(self) -> _dt.date:
        return (
            self.maturity_date 
            if self.loan_status is LoanStatus.CURRENT
            else self.last_pymnt_date or self.maturity_date
        )

    @property
    def outstanding_principal(self) -> float:
        return max(self.loan_amnt - self.total_rec_prncp, 0.0)
    

    # ------ Factory constructors from raw CSV row (or dict‑like) ------
    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "LoanRecord":
        """Parse a LoanStats CSV row into :class:`LoanRecord`.

        Date columns are in format ``"Jan‑2015"`` → parsed via `datetime.strptime`.
        """
        return cls(
            loan_id         =   str(                row["id"]),
            loan_amnt       =   float(              row["loan_amnt"]),
            term_months     =   int(                row["term"]),
            issue_date      =   _parse_date(        row["issue_d"]),
            int_rate        =   float(              row["int_rate"]),
            grade           =   str(                row["grade"]),
            sub_grade       =   str(                row["sub_grade"]),
            last_pymnt_date =   _parse_date(        row["last_pymnt_d"]),
            loan_status     =   LoanStatus.from_raw(row["loan_status"]),
            total_rec_prncp =   float(              row["total_rec_prncp"]),
            recoveries      =   float(              row["recoveries"]),
            total_rec_int   =   float(              row["total_rec_int"]),
            annual_inc      =   float(              row["annual_inc"])
        )

    @classmethod
    def from_dict(cls, row: dict) -> "LoanRecord":
        return cls(
            loan_id         =   str(                row["id"]),
            loan_amnt       =   float(              row["loan_amnt"]),
            term_months     =   int(                row["term"]),
            issue_date      =   _parse_date(        row["issue_d"]),
            int_rate        =   float(              row["int_rate"]),
            grade           =   str(                row["grade"]),
            sub_grade       =   str(                row["sub_grade"]),
            last_pymnt_date =   _parse_date(        row["last_pymnt_d"]),
            loan_status     =   LoanStatus.from_raw(row["loan_status"]),
            total_rec_prncp =   float(              row["total_rec_prncp"]),
            recoveries      =   float(              row["recoveries"]),
            total_rec_int   =   float(              row["total_rec_int"]),
            annual_inc      =   float(              row["annual_inc"])
        )

    # ------ Helpers for data conversion ----------------------------
    ### robust date parser (handles datetime, str, NaN, float)
    @staticmethod
    def _parse_date(val: str | _dt.date | _dt.datetime | float | None) -> Optional[_dt.date]:
        """
        Robust date parser that accepts:
        • pd.Timestamp / datetime / date
        • string  (e.g. 'Jan-2017', '2020-05-01')
        • numeric:
            – nanoseconds since epoch  (e.g. 1483920000000000000)
            – seconds since epoch      (e.g. 1483920000)
            – days    since epoch      (e.g. 17175.0)
        Returns a `datetime.date` or None on failure.
        """
        import math
        import numpy as np
        import pandas as pd
        from dateutil import parser as _p

        # 1 – already a datetime‐like
        if isinstance(val, pd.Timestamp):
            return val.date()                   # <-- ensure datetime.date
        if isinstance(val, _dt.datetime):
            return val.date()                   # <-- ensure datetime.date
        if isinstance(val, _dt.date):
            return val                     # <-- ensure datetime.date

        # 2 – NaNs
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None

        # 3 – numeric pathways
        if isinstance(val, (int, np.integer, float, np.floating)):
            # nanoseconds?
            if val > 1e13:                      # > ~2001 in ns
                ts = pd.to_datetime(int(val), unit="ns", errors="coerce")
            # seconds?
            elif val > 1e9:                     # > 2001 in s
                ts = pd.to_datetime(int(val), unit="s", errors="coerce")
            # days?
            else:
                ts = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(val))
            return None if pd.isna(ts) else ts.date()

        # 4 – assume string
        try:
            return _p.parse(str(val)).date()
        except Exception:                       # pragma: no cover
            _LOG.warning("Unparsable date value: %s", val)
            return None


    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # pragma: no cover – cosmetic only
        return (
            f"LoanRecord(id={self.loan_id}, amt={self.loan_amnt}, term={self.term_months}, "
            f"rate={self.int_rate}, status={self.loan_status})"
        )
