"""
core/anticluster.py
===================

Incremental anticlustering manager for a streaming universe of Lending-Club
loans.

The design follows “exchange heuristic” intuition:

• Each of the K *anticlusters* keeps a running centroid of feature vectors.
• A new loan is routed to the anticluster where it *maximises* distance
  from the existing centroid (subject to soft size parity and any explicit
  hard constraints on categorical counts).
• When a loan departs, the centroid and counts are updated in O(1).
• Optional `rebalance()` method performs a bounded-time local repair pass
  (pairwise swaps) to smooth out drift that accumulated after many arrivals/
  departures.

This module **does not** know anything about the StreamEngine; you call
`manager.handle_events(arrivals, departures)` from the outside.

"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Callable

import numpy as np

from .loan import LoanRecord, LoanStatus
from .features import vectorise_loan, vectorise_records

# --------------------------------------------------------------------------- #
#                               Anticluster class                             #
# --------------------------------------------------------------------------- #


@dataclass
class _GroupState:
    """Internal running stats for one anticluster."""

    size: int = 0
    centroid: np.ndarray | None = None
    members: set[str] = field(default_factory=set)
    # Example categorical balance tracker: grade counts etc.
    cat_counts: Counter = field(default_factory=Counter)

    # -------------  incremental math  ------------ #

    def add(self, loan_id: str, vec: np.ndarray, cat_keys: Sequence[str]) -> None:
        """
        O(1) update of centroid and counters after adding one member.
        """
        if self.size == 0:
            self.centroid = vec.copy()
        else:
            # new_centroid = old + (vec - old) / (n + 1)
            self.centroid += (vec - self.centroid) / (self.size + 1)
        self.size += 1
        self.members.add(loan_id)
        self.cat_counts.update(cat_keys)

    def remove(self, loan_id: str, vec: np.ndarray, cat_keys: Sequence[str]) -> None:
        """
        O(1) removal using historical centroid formula (requires current centroid).
        """
        if self.size == 0 or loan_id not in self.members:
            raise KeyError("Loan not in group or group empty")

        self.members.remove(loan_id)
        self.cat_counts.subtract(cat_keys)
        self.size -= 1
        if self.size == 0:
            self.centroid = None
        else:
            # new_centroid = old - (vec - old)/n
            self.centroid -= (vec - self.centroid) / self.size


# --------------------------------------------------------------------------- #
#                          AnticlusterManager (public)                        #
# --------------------------------------------------------------------------- #


class AnticlusterManager:
    """
    Maintains **K** balanced anticlusters in an online setting.

    Parameters
    ----------
    k
        Number of groups to keep.
    hard_balance_cols
        Optional list of `LoanRecord` attributes that must remain *perfectly*
        balanced across groups (e.g., 'grade').  The manager enforces this at
        assignment time by skipping groups that would violate the constraint.
    feature_fn
        Callable mapping LoanRecord → 1-D numeric `np.ndarray`.  Defaults to
        `default_feature_vector`.
    size_tolerance
        Allowed deviation from perfect size parity before a group is considered
        “too large”.  E.g. if K=4 and tolerance=1, sizes may differ by ≤1.
    """

    def __init__(
        self,
        k                   : int,
        numeric_feature_cols: Sequence[str],
        *,
        log_columns         : Sequence[str] | None      = None,
        scale               : bool = False,
        weights             : Dict[str, float] | None   = None,
        hard_balance_cols   : Sequence[str] | None      = None,
        size_tolerance      : int                       = 1,
    ) -> None:
        if k < 2:
            raise ValueError("Need at least two anticlusters.")
        
        self.k                                  = k
        self.cols                               = numeric_feature_cols
        self.log_cols                           = log_columns
        self.scale                              = scale
        self.weights                            = weights
        self._scaler                            = None  # StandardScaler, if scale=True
        
        self.size_tolerance                     = size_tolerance
        self.hard_cols      : Tuple[str, ...]   = tuple(hard_balance_cols or [])

        self._groups        : List[_GroupState] = [_GroupState() for _ in range(k)]
        self._index         : Dict[str, Tuple[int, np.ndarray, Tuple[str, ...]]] = {}

    # ------------------------------------------------------------------ #
    #                           public facade                            #
    # ------------------------------------------------------------------ #

    # ----------  event handlers  ---------- #
    def add_loans(self, new_loans: list[LoanRecord]) -> None:
        """
        Ingest a *batch* of LoanRecord objects, extend the feature matrix,
        and re-optimise the anticluster assignment.

        Parameters
        ----------
        new_loans : list[LoanRecord]
            Fresh loans arriving at the current iteration.

        Notes
        -----
        • Feature extraction is **vectorised** (single call) via
          `vectorise_records`, so there is no Python-loop per loan.
        • Optional log-transform / scaling / weighting are applied inside
          `vectorise_records`.
        • If ``scale=True`` the running ``StandardScaler`` is updated
          incrementally by passing the previous fitted instance back in.
        """
        if not new_loans:
            return  # nothing to do

        # 0 ── vectorise the whole batch in one call
        vec, _ = vectorise_records(
            new_loans,
            self.cols,
            scale=self.scale,
            log_columns=self.log_cols,
            weights=self.weights,
            scaler=self._scaler
        )
        vec = vec

        # initialise caches if first call
        if not hasattr(self, "_records"):
            self._records  = []
            self._features = np.empty((0, vec.shape[1]), dtype=float)

        # 1 ── iterate over loans & their feature rows (still Python-level,
        #       but all expensive math is pre-vectorised)
        for loan, vec in zip(new_loans, vec, strict=True):

            cat_keys = tuple(getattr(loan, col) for col in self.hard_cols)
            candidate_idxs = self._legal_groups(cat_keys)
            if not candidate_idxs:           # fallback if no legal group
                candidate_idxs = range(self.k)

            best_idx, best_score = None, -math.inf
            for idx in candidate_idxs:
                grp = self._groups[idx]
                # distance to centroid (empty group → huge dist = good)
                if grp.centroid is None:
                    dist = 1e6
                else:
                    dist = np.linalg.norm(vec - grp.centroid)
                # small penalty for oversized groups
                size_penalty = grp.size / max(1, self.avg_group_size())
                score = dist / (1 + size_penalty)
                if score > best_score:
                    best_idx, best_score = idx, score

            # 2 ── commit the assignment
            self._groups[best_idx].add(loan.loan_id, vec, cat_keys)
            self._index[loan.loan_id] = (best_idx, vec, cat_keys)

            # 3 ── append raw + feature for optional analysis later
            self._records.append(loan)
            self._features = np.vstack([self._features, vec])


    def add_loan(self, loan: LoanRecord) -> int:
        """
        Assign a loan to an anticluster *incrementally*.

        Parameters
        ----------
        loan : LoanRecord
            The loan to be added.

        Returns
        -------
        int : Index of the chosen group.
        """
        # Vectorise a single loan
        vec, _ = vectorise_records(
            loan,
            self.cols,
            scale=self.scale,
            log_columns=self.log_cols,
            weights=self.weights,
            scaler=self._scaler
        )
        vec = vec[0]  # unpack single (1, d) → (d,)

        cat_keys = tuple(getattr(loan, col) for col in self.hard_cols)
        candidate_idxs = self._legal_groups(cat_keys)

        if not candidate_idxs:
            candidate_idxs = range(self.k)

        best_idx, best_score = None, -math.inf
        for idx in candidate_idxs:
            group = self._groups[idx]
            if group.centroid is None:
                dist = 1e6
            else:
                dist = np.linalg.norm(vec - group.centroid)
            size_penalty = group.size / max(1, self.avg_group_size())
            score = dist / (1 + size_penalty)
            if score > best_score:
                best_idx, best_score = idx, score

        self._groups[best_idx].add(loan.loan_id, vec, cat_keys)
        self._index[loan.loan_id] = (best_idx, vec, cat_keys)

        return best_idx


    def remove_loan(self, loan_id: str) -> int:
        """
        Remove a loan when it departs.  Returns the group index it left.

        Parameters
        ----------
        loan_id : str
            Unique identifier of the loan to be removed.

        Returns
        -------
        idx  : int
            Index of the group from which the loan was removed (0 … K-1).
        """
        try:
            idx, vec, cat_keys = self._index.pop(loan_id)
        except KeyError as exc:  # pragma: no cover
            raise KeyError(f"Loan {loan_id} not found") from exc
        self._groups[idx].remove(loan_id, vec, cat_keys)
        return idx

    # ----------  monitoring / helpers  ---------- #

    def group_sizes(self) -> List[int]:
        """
        Return a list of current group sizes (number of loans in each group).

        Returns:
        -------
            List[int]: List of sizes of each group (0 … K-1).
        """
        return [g.size for g in self._groups]

    def avg_group_size(self) -> float:
        """
        Calculate the average size of the groups.

        Returns:
        -------
            float: Average size of the groups, computed as the total number of loans
                   divided by the number of groups (`self.k`).
        """
        return sum(self.group_sizes()) / self.k

    def snapshot(self) -> Dict[int, List[str]]:
        """
        Take a snapshot of the current state of the anticlusters.
        This returns a dictionary where keys are group indices (0 … K-1)
        and values are sorted lists of loan IDs in each group.

        Returns:
        -------
            Dict[int, List[str]]: Dictionary mapping group indices to sorted lists of loan IDs.
        """
        return {i: sorted(g.members) for i, g in enumerate(self._groups)}

    # ----------  optional repair pass  ---------- #

    def rebalance(self, max_swaps: int = 10) -> int:
        """
        Perform a local repair pass to balance group sizes by swapping loans
        between the largest and smallest groups.

        This method attempts to reduce the size difference between the largest
        and smallest groups by moving loans from the larger group to the smaller
        group, while respecting hard constraints on categorical balance.

        Parameters
        ----------
        max_swaps : int
            Maximum number of swaps to perform. Default is 10.

        Returns 
        -------
        int
            Number of swaps performed to rebalance the groups.
            If no swaps were needed or possible, returns 0.
        """
        #TODO: Isn't it better to, instead of taking an arbitrary member from `big`,
        #      take the one that is furthest from the centroid? This would help
        #      to maintain the heterogeneity of the groups.
        swaps = 0
        for _ in range(max_swaps):
            big, small = self._largest_and_smallest_groups()
            if big is None or small is None:
                break
            if self._groups[big].size - self._groups[small].size <= self.size_tolerance:
                break  # already within tolerance

            # take an arbitrary member from `big`
            loan_id = next(iter(self._groups[big].members))
            # re-evaluate `loan_id` against `small`
            _, vec, cat_keys = self._index[loan_id]
            if not self._group_accepts(small, cat_keys):
                break  # cannot move – would violate hard constraint

            # perform move
            self._groups[big].remove(loan_id, vec, cat_keys)
            self._groups[small].add(loan_id, vec, cat_keys)
            self._index[loan_id] = (small, vec, cat_keys)
            swaps += 1
        return swaps

    # ------------------------------------------------------------------ #
    #                       internal helper methods                      #
    # ------------------------------------------------------------------ #

    def _legal_groups(self, cat_keys: Tuple[str, ...]) -> List[int]:
        """
        Return indices of groups that won't violate hard constraints.

        Parameters
        ----------
        cat_keys : Tuple[str, ...]
            Categorical keys of the loan being added (aligned with `self.hard_cols`).   

        Returns
        -------
        List[int]
            Indices of groups that can accept the new loan without violating hard constraints.
            If `self.hard_cols` is empty, all groups are considered valid.
        """
        return [
            idx for idx in range(self.k) if self._group_accepts(idx, cat_keys)
        ]

    def _group_accepts(self, idx: int, cat_keys: Tuple[str, ...]) -> bool:
        """
        Check if a group can accept a new loan with given categorical keys.
        This is used to enforce hard constraints on categorical balance.
        
        If `self.hard_cols` is empty, all groups are considered valid.


        Parameters
        ----------
        idx : int
            Index of the group to check.
        cat_keys : Tuple[str, ...]
            Categorical keys of the loan being added (aligned with `self.hard_cols`).

        Returns
        -------
            bool: True if the group can accept the loan, False otherwise.
        """
        if not self.hard_cols:
            return True
        counts = self._groups[idx].cat_counts
        # A cat_key is a tuple aligned with self.hard_cols
        return all(counts[k] + 1 <= self.avg_group_size() + self.size_tolerance for k in cat_keys)

    def _largest_and_smallest_groups(self) -> Tuple[int | None, int | None]:
        sizes = self.group_sizes()
        if not any(sizes):
            return None, None
        largest = int(np.argmax(sizes))
        smallest = int(np.argmin(sizes))
        return largest, smallest


# --------------------------------------------------------------------------- #
#                              minimal smoke-test                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    # fabricate three tiny loans
    loans = [
        LoanRecord(
            loan_id=str(i),
            loan_amnt=10000 + i * 500,
            term_months=36,
            int_rate=0.10 + i * 0.01,
            issue_date=None,            # not used here
            last_payment_date=None,     # not used here
            status=LoanStatus.CURRENT,
            total_rec_prncp=0.0,
        )
        for i in range(10)
    ]

    mgr = AnticlusterManager(k=3)
    for lo in loans:
        idx = mgr.add_loan(lo)
        print(f"Added {lo.loan_id} → group {idx}")

    print("Group sizes after adds:", mgr.group_sizes())
    # remove a couple
    mgr.remove_loan("1")
    mgr.remove_loan("4")
    print("Group sizes after removals:", mgr.group_sizes())
    print("Rebalance swaps:", mgr.rebalance())
    print("Final snapshot:", mgr.snapshot())
