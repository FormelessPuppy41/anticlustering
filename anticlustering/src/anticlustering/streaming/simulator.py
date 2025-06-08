"""
core/simulator.py
=================

Generate synthetic month-by-month loan histories from static Lending Club
“LoanStats” snapshots.

The key public entry points are

    ▸ LoanSimulator.generate_history(...)
    ▸ LoanSimulator.batch_generate(...)

Both return `pandas.DataFrame` objects in **long format**:

| loan_id | period_number | period_date | payment_due | principal_paid |
| interest_paid | remaining_principal | status |

"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..loan.loan import LoanRecord, LoanStatus
from .amortization import AmortizationSchedule, PaymentPeriod

# --------------------------------------------------------------------------- #
#                            ――  Public facade  ――                            #
# --------------------------------------------------------------------------- #


class LoanSimulator:
    """Create synthetic monthly time-series for Lending Club loans."""

    #: columns order for histories
    _COLS = [
        "loan_id",
        "period_number",
        "period_date",
        "payment_due",
        "principal_paid",
        "interest_paid",
        "remaining_principal",
        "status",
    ]

    # ----------------------------  single loan  ---------------------------- #

    @staticmethod
    def generate_history(
        loan: LoanRecord,
        as_of: _dt.date | None = None,
        assume_regular_prepayment: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate the life-cycle of *one* loan.

        Parameters
        ----------
        loan
            A fully-populated :class:`LoanRecord`.
        as_of
            Optional cut-off date (inclusive).  Months after this
            date are not simulated – useful for “current” loans when
            replaying history month by month.
        assume_regular_prepayment
            • *True* (default): if the static snapshot shows an early
              payoff (`loan_status == Fully Paid` but `last_pymnt_d`
              < maturity), allocate **equal extra principal** to each
              remaining scheduled payment so that the balance hits 0
              precisely on `last_pymnt_d`.
            • *False*: treat *last* payment as a **lump-sum** that
              clears the residual principal.

        Returns
        -------
        pd.DataFrame
            Long-format table – one row per simulated month.
        """

        # ------------------------------------------------------------------ #
        # 1. Establish a schedule “skeleton” from contractual terms           #
        # ------------------------------------------------------------------ #
        sched = AmortizationSchedule.from_loan_record(loan)

        # ------------------------------------------------------------------ #
        # 2. Post-process for default / early payoff                         #
        # ------------------------------------------------------------------ #
        df = _schedule_to_frame(sched, loan.loan_id)

        # if an `as_of` cut-off is requested, drop future months
        if as_of is not None:
            df = df.loc[df["period_date"] <= pd.Timestamp(as_of)]

        status_series = _derive_status_column(loan, df["period_date"])
        df["status"] = status_series.values

        # adjust remaining principal for “current” loans beyond as_of
        if loan.loan_status == LoanStatus.CURRENT and as_of is not None:
            last_row_idx = df.index[-1]
            df.loc[last_row_idx, "remaining_principal"] = loan.outstanding_principal 

        df = df[LoanSimulator._COLS].copy()
        return df.reset_index(drop=True)

    # ---------------------------  batch helper  --------------------------- #

    @staticmethod
    def batch_generate(
        loans: Iterable[LoanRecord],
        as_of: _dt.date | None = None,
        assume_regular_prepayment: bool = True,
    ) -> pd.DataFrame:
        """
        Generate synthetic histories for a *collection* of loans.

        Parameters
        ----------
        loans
            Iterable of :class:`LoanRecord` objects.
        as_of   
            Optional cut-off date (inclusive).  Months after this
            date are not simulated – useful for “current” loans when
            replaying history month by month.
        assume_regular_prepayment
            • *True* (default): if the static snapshot shows an early
              payoff (`loan_status == Fully Paid` but `last_pymnt_d`
              < maturity), allocate **equal extra principal** to each
              remaining scheduled payment so that the balance hits 0
              precisely on `last_pymnt_d`.
            • *False*: treat *last* payment as a **lump-sum** that
              clears the residual principal.

        Returns
        -------
        pd.DataFrame
            Concatenated histories with an index per loan.
        """
        frames: List[pd.DataFrame] = []
        for loan in loans:
            frames.append(
                LoanSimulator.generate_history(
                    loan,
                    as_of                       =as_of,
                    assume_regular_prepayment   =assume_regular_prepayment,
                )
            )
        return pd.concat(frames, axis=0, ignore_index=True)

    # ------------------------------------------------------------------ #
    #                    ――  internal helper logic  ――                   #
    # ------------------------------------------------------------------ #


def _schedule_to_frame(
    sched: AmortizationSchedule,
    loan_id: str | int,
) -> pd.DataFrame:
    """Convert :class:`AmortizationSchedule` → tidy pandas.DataFrame."""
    if not sched.payments:
        raise ValueError("Amortization schedule is empty")
    
    periods_as_dicts: List[dict] = []
    for p in sched.payments:
        periods_as_dicts.append(
            {
                "loan_id"               : loan_id,
                "period_number"         : p.period_number,
                "period_date"           : pd.Timestamp(p.period_date),
                "payment_due"           : p.payment_amount,
                "principal_paid"        : p.principal_component,
                "interest_paid"         : p.interest_component,
                "remaining_principal"   : p.remaining_principal,
            }
        )
    return pd.DataFrame(periods_as_dicts)


def _derive_status_column(
    loan: LoanRecord,
    date_series: pd.Series,
) -> pd.Series:
    """
    Compute a 'status' vector aligned with the simulated period_date series.

    All comparisons are performed on plain `datetime.date` objects to avoid
    Timestamp/date mismatches.
    """
    default_cutoff = (
        loan.last_pymnt_date
        if loan.loan_status in {LoanStatus.DEFAULT, LoanStatus.CHARGED_OFF}
        else None
    )

    statuses: List[str] = []
    for ts in date_series:
        pd_date = ts.date() if isinstance(ts, pd.Timestamp) else ts

        if pd_date < loan.issue_date:
            raise ValueError("Simulated date precedes issue_d")

        if default_cutoff and pd_date >= default_cutoff:
            statuses.append(loan.loan_status.value)
        elif pd_date >= loan.departure_date:
            statuses.append(loan.loan_status.value)
        else:
            statuses.append(LoanStatus.CURRENT.value)

    return pd.Series(statuses, index=date_series.index)


# --------------------------------------------------------------------------- #
#                               ――  dataclass ――                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimulationConfig:
    """
    Container for knobs that *might* vary between experiments.

    Currently unused in `LoanSimulator` (all knobs are arguments).
    Left here as a hint for future configurability (e.g., delinquency model).
    """

    prepayment_behavior: str = "pro-rata"  # or "lump-sum"
    accrue_interest_daily: bool = False


# --------------------------------------------------------------------------- #
#                             ――  module self-test ――                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import json
    from pathlib import Path

    # quick smoke test with a JSON-encoded LoanRecord fixture --------------
    fixture_path = Path(__file__).with_name("loan_fixture.json")
    if fixture_path.exists():
        raw = json.loads(fixture_path.read_text())
        loan = LoanRecord.from_dict(raw)
        hist = LoanSimulator.generate_history(loan)
        print(hist.head())
        print(hist.tail())
    else:
        print("No fixture found – run unit tests instead.")
