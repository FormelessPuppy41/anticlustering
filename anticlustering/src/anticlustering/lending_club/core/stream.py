"""
core/stream.py
==============

A light-weight engine that replays Lending Club loans month-by-month so that
other components (e.g. AnticlusterManager) can operate on a *live* universe
of active loans.

Key abstractions
----------------
• **ActivePool**     – in-memory set / dict of currently active loans.  
• **StreamEngine**   – orchestrates arrivals & departures while advancing a
                       calendar pointer.

The engine is *deterministic*: given the same list of LoanRecord objects it
always produces the same sequence of states.

Author:  Your Name <your.email@example.com>
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

import bisect
from dateutil.relativedelta import relativedelta

from .loan import LoanRecord, LoanStatus, _add_months


# --------------------------------------------------------------------------- #
#                                 ActivePool                                  #
# --------------------------------------------------------------------------- #

class ActivePool:
    """
    Container for loans that are *currently* alive in the simulation.

    Exposes:
    --------
    • ``add(loan)``         – O(1) insert  
    • ``remove(loan_id)``   – KeyError if loan absent  
    • ``__iter__`` / ``__len__`` – iterates *LoanRecord* objects
    """

    def __init__(self) -> None:
        self._loans: Dict[str, LoanRecord] = {}

    # ---------------  mutating ops  --------------- #

    def add(self, loan: LoanRecord) -> None:
        self._loans[loan.loan_id] = loan

    def remove(self, loan_id: str) -> LoanRecord:
        return self._loans.pop(loan_id)

    # ---------------  containers API  ------------- #

    def __iter__(self) -> Iterator[LoanRecord]:
        return iter(self._loans.values())

    def __len__(self) -> int:                  # pragma: no cover
        return len(self._loans)

    def __contains__(self, loan_id: str) -> bool:  # pragma: no cover
        return loan_id in self._loans

    # ---------------  helpers  -------------------- #

    def snapshot_ids(self) -> List[str]:
        """Return **sorted** list of active `loan_id`s (for deterministic logs)."""
        return sorted(self._loans.keys())


# --------------------------------------------------------------------------- #
#                                 StreamEngine                                #
# --------------------------------------------------------------------------- #

class StreamEngine:
    """
    Replay a *static* Lending-Club dataset as if it were a **time-ordered
    stream**.  One calendar step == **one month**.

    Parameters
    ----------
    loans
        Iterable of *LoanRecord*s (order irrelevant; will be sorted internally)
    start_date
        Calendar date to start the simulation (defaults to min(issue_d))
    end_date
        Inclusive cut-off; if *None* the engine runs until every loan departs.
    """

    # ----------  construction  ---------- #

    def __init__(
        self,
        loans: Iterable[LoanRecord],
        start_date: _dt.date | None = None,
        end_date: _dt.date | None = None,
    ) -> None:
        # sort loans by issue_date so we can bisect arrivals efficiently
        self._all_loans: List[LoanRecord] = sorted(loans, key=lambda lo: lo.issue_date)
        self._arrival_dates: List[_dt.date] = [lo.issue_date for lo in self._all_loans]

        self.start_date   = start_date or min(lo.issue_date for lo in loans)
        self.end_date     = end_date if end_date else None

        self.current_date: _dt.date = self.start_date
        
        self.pool = ActivePool()

        # internal cursor: index into _all_loans for next arrival candidate
        self._next_arrival_idx: int = 0

    # ------------------------------------------------------------------ #
    #                           main public API                           #
    # ------------------------------------------------------------------ #

    def run(self) -> Iterator[Tuple[_dt.date, List[LoanRecord], List[LoanRecord]]]:
        """
        Generator that yields **(date, arrivals, departures)** for each step.

        Stops when ``self.end_date`` is reached *or* no active loans / arrivals
        remain.

        Yields
        ------
        date
            The “as-of” month-end (always the *first* of month for clarity).
        arrivals
            List of LoanRecord objects that *entered* at this date.
        departures
            List of LoanRecord objects that *left* at this date.
        """
        while True:
            arrived = self._process_arrivals()
            departed = self._process_departures()

            yield (self.current_date, arrived, departed)

            # advance one calendar month
            self.current_date = _add_months(self.current_date, 1)

            # stopping criteria
            if self.end_date and self.current_date > self.end_date:
                break
            if self._next_arrival_idx >= len(self._all_loans) and len(self.pool) == 0:
                break

    # ------------------------------------------------------------------ #
    #                         internal mechanics                          #
    # ------------------------------------------------------------------ #

    # ----------  arrivals: issue_date == current_date  ---------- #

    def _process_arrivals(self) -> List[LoanRecord]:
        arrived: List[LoanRecord] = []

        # find slice of loans whose issue_date == current_date
        lo, hi = self._arrival_window()
        for idx in range(lo, hi):
            loan = self._all_loans[idx]
            self.pool.add(loan)
            arrived.append(loan)

        # move cursor forward
        self._next_arrival_idx = hi
        return arrived

    def _arrival_window(self) -> Tuple[int, int]:
        """
        Return (lo, hi) slice indices into ``_all_loans`` whose
        `issue_date` == `self.current_date`.
        """
        lo = bisect.bisect_left(self._arrival_dates, self.current_date, self._next_arrival_idx)
        hi = bisect.bisect_right(self._arrival_dates, self.current_date, lo)
        return lo, hi

    # ----------  departures: departure_date == current_date  ---------- #

    def _process_departures(self) -> List[LoanRecord]:
        departed: List[LoanRecord] = []

        # collect IDs to avoid mutating dict while iterating
        to_remove = [
            loan_id
            for loan_id, loan in self.pool._loans.items()
            if loan.departure_date <= self.current_date
        ]
        for loan_id in to_remove:
            departed.append(self.pool.remove(loan_id))

        return departed


# --------------------------------------------------------------------------- #
#                                quick smoke-test                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import json
    from pathlib import Path

    fixture_fp = Path(__file__).with_name("loan_fixture.json")
    if not fixture_fp.exists():
        print("No fixture json available – run unit tests.")
        raise SystemExit(0)

    rows = json.loads(fixture_fp.read_text())
    loans = [LoanRecord.from_dict(row) for row in rows]

    engine = StreamEngine(loans)
    for date, arr, dep in engine.run():
        print(
            f"{date:%Y-%m}: "
            f"+{len(arr):3d} arrivals, "
            f"-{len(dep):3d} departures, "
            f"{len(engine.pool):5d} active"
        )
