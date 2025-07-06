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
from typing import Dict, Iterable, Iterator, List, Tuple, Callable, Optional
from dataclasses import dataclass

import bisect
import logging
from dateutil.relativedelta import relativedelta

from ..core.loans.loan import LoanRecord, LoanStatus, _add_months

_LOG = logging.getLogger(__name__)

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
#                                 StreamEvent                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StreamEvent:
    """
    Represents a single calendar step in the stream.
    Contains the date, a list of loans that arrived at this date, and a list
    of loans that departed at this date.
    """
    date: _dt.date
    arrivals: List[LoanRecord]
    departures: List[LoanRecord]

class CalendarTicker(Iterator[_dt.date]):
    def __init__(self, start_date: _dt.date, end_date: _dt.date):
        if end_date < start_date:
            raise ValueError("end_date must be greater than or equal to start_date")
        self.current_date = start_date
        self.end_date = end_date

    def __iter__(self) -> "CalendarTicker":
        return self
    
    def __next__(self) -> _dt.date:
        if self.current_date > self.end_date:
            raise StopIteration
        next_date = self.current_date
        # Advance to the first of the next month
        self.current_date = _add_months(self.current_date, 1)
        return next_date.replace(day=1)

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
    initial_active_pool : bool (default: **False**)
        If *True*, the engine starts with all loans started before the start_date in the active pool.
        If *False*, the pool is empty at the start and only populated by arrivals.
        Current data starts from start_date, so this is not useful in this dataset.
    """

    # ----------  construction  ---------- #

    def __init__(
        self,
        loans: List[LoanRecord],
        start_date: Optional[_dt.date] = None,
        end_date: Optional[_dt.date] = None,
        hooks: Optional[List[Callable[[StreamEvent], None]]] = None,
    ) -> None:
        if not loans:
            raise ValueError("StreamEngine requires a non‐empty list of LoanRecord")

        # Default time window
        self.start_date = (
            start_date
            if start_date is not None
            else min(lo.issue_d for lo in loans)
        )
        self.end_date = (
            end_date
            if end_date is not None
            else max(lo.departure_date for lo in loans)
        )
        if self.end_date < self.start_date:
            raise ValueError(
                f"After defaults, end_date {self.end_date!r} < start_date {self.start_date!r}"
            )

        self.loans = loans
        self.hooks = hooks or []

        # Pre‐index arrivals/departures by date
        self._arrivals: Dict[_dt.date, List[LoanRecord]] = self._index_by(lambda lo: lo.issue_d)
        self._departures: Dict[_dt.date, List[LoanRecord]] = self._index_by(lambda lo: lo.departure_date)

    def _index_by(
            self, 
            key_fn: Callable[[LoanRecord], _dt.date]
        ) -> Dict[_dt.date, List[LoanRecord]]:
        idx: Dict[_dt.date, List[LoanRecord]] = {}
        for lo in self.loans:
            idx.setdefault(key_fn(lo), []).append(lo)
        return idx

    def _validate_day(self, current: _dt.date, arrivals: List[LoanRecord], departures: List[LoanRecord]) -> None:
        # No loan can both arrive and depart (or be duplicated) on the same day
        ids = [lo.loan_id for lo in arrivals + departures]
        if len(ids) != len(set(ids)):
            #extract the duplicates
            seen: Dict[str, int] = {}
            for loan_id in ids:
                if loan_id in seen:
                    seen[loan_id] += 1
                else:
                    seen[loan_id] = 1
            duplicates = [loan_id for loan_id, count in seen.items() if count > 1]

            # extract the loan records for the duplicates
            loans = [lo for lo in arrivals + departures if lo.loan_id in duplicates]
            raise ValueError(f"Duplicate loan IDs on {current!r}: {duplicates}. Loans: {loans!r}")


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
        for current in CalendarTicker(self.start_date, self.end_date):
            arrivals = self._arrivals.get(current, [])
            departures = self._departures.get(current, [])
            try:
                self._validate_day(current, arrivals, departures)
            except:
                continue

            evt = StreamEvent(current, arrivals, departures)
            for hook in self.hooks:
                hook(evt)

            # *** Crucial: yield a 3‐tuple so unpacking works! ***
            yield current, arrivals, departures

    # ------------------------------------------------------------------ #
    #                         internal mechanics                          #
    # ------------------------------------------------------------------ #

    # ----------  arrivals: issue_date == current_date  ---------- #
    """
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
        ""
        Return (lo, hi) slice indices into ``_all_loans`` whose
        `issue_date` == `self.current_date`.
        ""
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
    """

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
