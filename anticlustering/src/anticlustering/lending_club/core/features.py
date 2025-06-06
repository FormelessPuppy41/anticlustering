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
from typing import Any, Dict, Iterable, List

import numpy as np
from dateutil import parser as _p

from .loan import LoanRecord, LoanStatus


# ── helpers ───────────────────────────────────────────────────────────────
def _parse_percent(p: str | float) -> float:
    """'13.56%' → 13.56; '13.56' → 13.56; 0.1356 → 13.56"""
    if isinstance(p, (int, float)):
        return float(p) * (100 if p <= 1 else 1)
    return float(str(p).strip().rstrip("%"))


def _parse_term(term: str | int) -> int:
    """' 36 months' → 36"""
    if isinstance(term, int):
        return term
    return int(str(term).strip().split()[0])


def _parse_date(val: Any):
    if val in ("", None) or (isinstance(val, float) and math.isnan(val)):
        return None
    return _p.parse(str(val)).date()


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
        term_months=       _parse_term(row["term"]),
        issue_date=        _parse_date(row["issue_d"]),
        int_rate=          _parse_percent(row["int_rate"]),
        grade=             row["grade"],
        sub_grade=         row["sub_grade"],
        last_pymnt_date=   _parse_date(row.get("last_pymnt_d")),
        loan_status=       LoanStatus.from_raw(row["loan_status"]),
        total_rec_prncp=   float(row["total_rec_prncp"]),
        recoveries=        float(row["recoveries"]),
        total_rec_int=     float(row["total_rec_int"]),
        annual_inc=        float(row["annual_inc"]),
    )


_NUMERIC_ATTRS: List[str] = [
    "loan_amnt",
    "int_rate",
    "term_months",
    "total_rec_prncp",
    "total_rec_int",
    "recoveries",
    "annual_inc",
]

#TODO: Add a scaler and a feature weight vector to this module.
def vectorise(loan: LoanRecord) -> np.ndarray:
    """
    Turn a ``LoanRecord`` into a **1-D numpy array** of numeric features.
    Online/offline solvers can plug this into distance or variance
    calculations.

    By default we expose the seven attributes in ``_NUMERIC_ATTRS`` in
    the declared order.  Adjust `_NUMERIC_ATTRS` if you need more.
    """
    return np.fromiter((getattr(loan, a) for a in _NUMERIC_ATTRS), dtype=float)
