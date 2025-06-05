"""
amortization.py
================

Pure utility module for constructing amortization schedules from
:class:`~anticluster_stream.core.loan.LoanRecord` instances.

The implementation intentionally stays free of I/O so it can be imported
from any environment (ETL worker, Jupyter, or unit‑test) without side‑effects.
"""

from __future__ import annotations

import pandas as pd

from math import isclose, pow
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator, List, Sequence

from .loan import LoanRecord, LoanStatus

__all__ = [
    "PaymentPeriod",
    "AmortizationSchedule",
]


def _add_months(d: pd.Timestamp, months: int) -> pd.Timestamp:
    """Return a date *months* periods after *d* (day clamped to ≤28)."""
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year
    m += 1
    # clamp day so Feb always works
    day = min(d.day, 28)
    return pd.Timestamp(y, m, day)

def _annual_rate_to_monthly(annual_rate_pct: float) -> float:
    """Convert annual (percentage dd.dd) rate to monthly decimal (0.dddd) rate."""
    if annual_rate_pct < 0:
        raise ValueError("Annual rate must be non-negative")
    if 0 <= annual_rate_pct <= 1.0:
        return annual_rate_pct / 12.0  # already in decimal form
    return (annual_rate_pct / 100.0) / 12.0

@dataclass(frozen=True, slots=True)
class PaymentPeriod:
    """A single row of an amortization table."""
    period_number           : int
    period_date             : pd.Timestamp
    payment_amount          : float
    principal_component     : float
    interest_component      : float
    remaining_principal     : float


class AmortizationSchedule(Sequence[PaymentPeriod]):
    """Container and generator for a loan’s amortization cash‑flows."""

    def __init__(self, payments: List[PaymentPeriod], loan: LoanRecord):
        """
        Initialize the amortization schedule with a list of payment periods
        and the associated loan record.

        Parameters
        ----------
        payments : List[PaymentPeriod]
            List of payment periods in the amortization schedule.
        loan : LoanRecord
            The loan record associated with this schedule.
        """
        self.payments = payments
        self.loan = loan

    # ------------------------------------------------------------------ #
    #  Python data‑model helpers
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[PaymentPeriod]:
        return iter(self.payments)

    def __getitem__(self, idx):
        return self.payments[idx]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.payments)

    # ------------------------------------------------------------------ #
    #  Factory helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
        """
        Fixed instalment amount for a fully‑amortising loan.

        Parameters
        ----------
        principal : float
            Original loan amount.
        annual_rate : float
            Nominal annual interest rate in percent (e.g. 13.99) or decimal (e.g. 0.1399).
        term_months : int
            Scheduled number of monthly payments (36 or 60 for Lending Club).

        Returns
        -------
        float
            Constant monthly cash amount (principal + interest).
        """
        if term_months <= 0:
            raise ValueError("term_months must be positive")

        r = _annual_rate_to_monthly(annual_rate)
        factor = pow(1 + r, term_months)
        return principal * r * factor / (factor - 1)

    @classmethod
    def from_loan_record(cls, loan: LoanRecord) -> "AmortizationSchedule":
        """
        Build a schedule honouring the observed *terminal* state of the loan.

        If the loan finished early (Fully Paid) or defaulted (Charged Off),
        the schedule is truncated at :pyattr:`LoanRecord.last_pymnt_d`.
        Loans still **Current** in the snapshot are projected to full term.

        Notes
        -----
        The method is deterministic and makes *no* assumption about late
        payments beyond the final state reported in the snapshot, because the
        original month‑by‑month tape is unavailable.
        """
        cls.loan = loan
        principal = loan.loan_amnt
        term = loan.term_months
        annual_rate = loan.int_rate
        start_date = loan.issue_date

        instalment = cls.monthly_payment(principal, annual_rate, term)
        r = _annual_rate_to_monthly(annual_rate)

        payments: List[PaymentPeriod] = []
        balance = principal
        current_date = start_date # payday

        for i in range(1, term + 1):
            # Stop if we reach last_pymnt_d for loans that ended prematurely
            if (
                loan.last_pymnt_date is not None
                and current_date > loan.last_pymnt_date
                and loan.loan_status
                in {
                    LoanStatus.FULLY_PAID,
                    LoanStatus.CHARGED_OFF,
                    LoanStatus.DEFAULT,
                }
            ):
                break

            interest_component = balance * r
            principal_component = instalment - interest_component

            # Protect against rounding blowing up the last repayment
            if principal_component > balance:
                principal_component = balance
                instalment = principal_component + interest_component

            balance -= principal_component

            payments.append(
                PaymentPeriod(
                    period_number=i,
                    period_date=current_date,
                    payment_amount=instalment,
                    principal_component=principal_component,
                    interest_component=interest_component,
                    remaining_principal=balance,
                )
            )

            if isclose(balance, 0.0, abs_tol=1e-2):
                break

            current_date = _add_months(start_date, i)

        return cls(payments, loan)

    # ------------------------------------------------------------------ #
    #  Convenience analytics
    # ------------------------------------------------------------------ #
    @property
    def paid_principal(self) -> float:
        """Total principal repaid across all periods."""
        return sum(p.principal_component for p in self.payments)

    @property
    def paid_interest(self) -> float:
        """Total interest paid across all periods."""
        return sum(p.interest_component for p in self.payments)

    @property
    def remaining_principal(self) -> float:
        """Outstanding balance after the final generated period."""
        return self.payments[-1].remaining_principal if self.payments else self.loan.loan_amnt
