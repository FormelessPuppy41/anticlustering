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
from __future__ import annotations

import datetime as _dt
import pandas as pd
import enum
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

__all__ = [
    "LoanStatus",
    "LoanRecord",
]

# ------------------------------------------------------------------------------- #
# Enums
# ------------------------------------------------------------------------------- #
class LoanStatus(str, enum.Enum):
    """Canonical terminal states for a Lending Club loan.
    
    We map the raw LoanStats values to a smaller, analysis‑friendly set.
    """

    # TODO:
    # Default has been deprecated in favour of Charged Off.
    # Often only see: "Current", "Fully Paid", "Charged Off".
    # Does not meet the credit policy are legacy loans from pre-2010.
    # Issued, Cancelled, Withdrawn are extremely rare and usually dropped. 
    #
    # Probably best to just drop all but the main three:
    # "Current", "Fully Paid", "Charged Off".
    # !!!! simply remove the rest from the mapping.

    CURRENT                 = "Current"  # still active in snapshot (right‑censored)
    FULLY_PAID              = "Fully Paid"
    CHARGED_OFF             = "Charged Off"  # default / loss event
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

#TODO: Is this correct? some are use "Jan-2015" format others use "12-01-2015" format etc.
_DATE_FMT_IN = "%b-%Y"
def _parse_date(s: str | None) -> pd.Timestamp | None:
    return pd.to_datetime(s, format=_DATE_FMT_IN, errors="coerce") if s else None

def _add_months(date_: pd.Timestamp, months: int) -> pd.Timestamp:
    """Return `date_` shifted forward by *months* calendar months.

    Works for end‑of‑month dates – if original day > target month length, clamp
    to last day of target month (similar to Excel behaviour).
    """
    month       = date_.month - 1 + months
    year        = date_.year + month // 12
    month       = month % 12 + 1
    day         = min(date_.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return pd.Timestamp(year, month, day)



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
    issue_date          : pd.Timestamp
    int_rate            : float  # annual nominal rate in *percent* – e.g. 13.56
    grade               : str  # e.g. "A", "B", "C" etc.
    sub_grade           : str  # e.g. "A1", "B2", "C3" etc.
    last_pymnt_date     : Optional[pd.Timestamp]
    loan_status         : LoanStatus
    total_rec_prncp     : float = 0.0
    recoveries          : float = 0.0
    total_rec_int       : float = 0.0
    annual_inc          : float = 0.0  # annual income of the borrower (in USD)
    # lazily computed cache – excluded from equality / repr
    _monthly_payment    : Decimal | None = field(init=False, default=None, repr=False, compare=False)

    # ------- Core financial helpers --------------------------------
    
    @property
    def monthly_rate(self) -> Decimal:
        """Monthly nominal rate (decimal, not percent)."""
        return (self.int_rate / Decimal("100")) / Decimal("12")

    @property
    def monthly_payment(self) -> Decimal:
        """Fixed amortising instalment according to standard annuity formula."""
        if self._monthly_payment is None:
            r = float(self.monthly_rate)
            n = self.term_months
            if r == 0:
                payment = float(self.loan_amnt) / n
            else:
                payment = float(self.loan_amnt) * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
            object.__setattr__(self, "_monthly_payment", Decimal(str(round(payment, 2))))
        return self._monthly_payment  # type: ignore [return‑value]

    
    #  ----- Lifecycle helpers --------------------------------

    @property
    def maturity_date(self) -> pd.Timestamp:
        """Calendar end date if loan runs full term."""
        return _add_months(self.issue_date, self.term_months)

    @property
    def departure_date(self) -> pd.Timestamp:
        """Date when loan exited, based on snapshot information."""
        if self.loan_status is LoanStatus.CURRENT:
            # Approximate: assume still active through scheduled maturity.
            return self.maturity_date
        # Fallback to last payment date when available, else scheduled.
        return self.last_pymnt_date or self.maturity_date


    # ------- Convenience predicates ------------------------------

    @property
    def is_fully_paid(self) -> bool:
        return self.loan_status is LoanStatus.FULLY_PAID

    @property
    def is_defaulted(self) -> bool:
        return self.loan_status in {LoanStatus.CHARGED_OFF}

    @property
    def outstanding_principal(self) -> Decimal:
        """Current outstanding principal (total minus recovered)."""
        return max(self.loan_amnt - self.total_rec_prncp, 0.0)
    

    # ------ Factory constructors from raw CSV row (or dict‑like) ------
    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "LoanRecord":
        """Parse a LoanStats CSV row into :class:`LoanRecord`.

        Date columns are in format ``"Jan‑2015"`` → parsed via `datetime.strptime`.
        """
        return cls(
            loan_id         =   str(row["id"]),
            loan_amnt       =   float(row["loan_amnt"]),
            term_months     =   int(row["term"]),
            issue_date      =   pd.to_datetime(row["issue_d"]),
            int_rate        =   float(row["int_rate"]),
            grade           =   str(row["grade"]),
            sub_grade       =   str(row["sub_grade"]),
            last_pymnt_date =   pd.to_datetime(row["last_pymnt_d"]),
            loan_status     =   LoanStatus.from_raw(row["loan_status"]),
            total_rec_prncp =   float(row["total_rec_prncp"]),
            recoveries      =   float(row["recoveries"]),
            total_rec_int   =   float(row["total_rec_int"]),
            annual_inc      =   float(row["annual_inc"])
        )
        return cls(
            loan_id         =   row["id"] or row.get("loan_id", ""),
            loan_amnt       =   Decimal(row["loan_amnt"]),
            term_months     =   int(row["term"].strip().split()[0]),
            issue_date      =   _parse_date(row["issue_d"]),
            int_rate        =   Decimal(row["int_rate"].strip("%")),
            grade           =   str(row["grade"]),
            sub_grade       =   str(row["sub_grade"]),
            last_pymnt_date =   _parse_date(row.get("last_pymnt_d", "")),
            loan_status     =   LoanStatus.from_raw(row["loan_status"]),
            total_rec_prncp =   Decimal(row.get("total_rec_prncp", "0")),
            recoveries      =   Decimal(row.get("recoveries", "0")),
            total_rec_int   =   Decimal(row.get("total_rec_int", "0")),
            annual_inc      =   float(row["annual_inc"])
        )

    @classmethod
    def from_dict(cls, row: dict) -> "LoanRecord":
        return cls(
            loan_id         =   str(row["id"]),
            loan_amnt       =   float(row["loan_amnt"]),
            term_months     =   int(row["term"]),
            issue_date      =   pd.to_datetime(row["issue_d"]),
            int_rate        =   float(row["int_rate"]),
            grade           =   str(row["grade"]),
            sub_grade       =   str(row["sub_grade"]),
            last_pymnt_date =   pd.to_datetime(row["last_pymnt_d"]),
            loan_status     =   LoanStatus.from_raw(row["loan_status"]),
            total_rec_prncp =   float(row["total_rec_prncp"]),
            recoveries      =   float(row["recoveries"]),
            total_rec_int   =   float(row["total_rec_int"]),
            annual_inc      =   float(row["annual_inc"])
        )
        return cls(
            loan_id          =  row["id"],               # adapt column names
            loan_amnt        =  float(row["loan_amnt"]),
            term_months      =  int(str(row["term"]).strip()[:2]),
            issue_date       =  _parse_date(row["issue_d"]),
            int_rate         =  float(row["int_rate"].strip("%"))/100,
            grade           =   str(row["grade"]),
            sub_grade       =   str(row["sub_grade"]),
            last_pymnt_date  =  _parse_date(row.get("last_pymnt_d")),
            loan_status      =  LoanStatus(row["loan_status"]),
            total_rec_prncp  =  float(row.get("total_rec_prncp", 0.0)),
            recoveries       =  float(row.get("recoveries", 0.0)),
            total_rec_int    =  float(row.get("total_rec_int", 0.0)),
            annual_inc      =   float(row["annual_inc"])
        )

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # pragma: no cover – cosmetic only
        return (
            f"LoanRecord(id={self.loan_id}, amt={self.loan_amnt}, term={self.term_months}, "
            f"rate={self.int_rate}, status={self.loan_status})"
        )
