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
import logging

_LOG = logging.getLogger(__name__)

import numpy as np

from ..loan.loan import LoanRecord, LoanStatus
from ..loan.vectorizer import LoanVectorizer

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
        vectorizer          : LoanVectorizer,
        *,
        hard_balance_cols   : Sequence[str] | None      = None,
        size_tolerance      : int                       = 1,
    ) -> None:
        if k < 2:
            raise ValueError("Need at least two anticlusters.")
        
        self.k                                  = k
        self.vectorizer                         = vectorizer
        self._n_numeric: int                    = vectorizer.n_numeric

        self.size_tolerance                     = size_tolerance
        self.hard_cols      : Tuple[str, ...]   = tuple(hard_balance_cols or [])

        self._groups        : List[_GroupState] = [_GroupState() for _ in range(k)]
        self._index         : Dict[str, Tuple[int, np.ndarray, Tuple[str, ...]]] = {}
        self._dim           : int               = 0 # current common feature vector length
        

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
        # _LOG.info(
        #     "AntiMan.add_loans: Current size of anticlusters: %s. \nAdding %d new loans to anticlusters. \nCurrent Centroids: %s",
        #     self.group_sizes(),
        #     len(new_loans),
        #     [g.centroid.tolist() if g.centroid is not None else None for g in self._groups],
        # )

        if not new_loans:
            return  # nothing to do

        # 0 ── vectorise the whole batch in one call
        X = self.vectorizer.transform(new_loans)
        self._ensure_dim(X.shape[1])

        # initialise caches if first call
        if not hasattr(self, "_records"):
            self._records  = []
            self._features = np.empty((0, X.shape[1]), dtype=float)

        # 1 ── iterate over loans & their feature rows (still Python-level,
        #       but all expensive math is pre-vectorised)
        for loan, vec in zip(new_loans, X, strict=True):
            self._assign_single(loan, vec)

        a, b = self.vectorizer.partial_update(new_loans)
        self._rescale_all_vectors(a, b)
        # _LOG.info(
        #     "AntiMan.add_loans: Updated anticlusters after adding %d loans: %s. With centroids: %s",
        #     len(new_loans),
        #     self.group_sizes(),
        #     [g.centroid.tolist() if g.centroid is not None else None for g in self._groups],
        # )
        
    def remove_loans(self, old_loans: list[LoanRecord]) -> List[int]:
        """
        Remove a batch of loans by their IDs.  Returns the group indices they left.

        Parameters
        ----------
        old_loans : list[LoanRecord]
            Loans to be removed from the anticlusters.  The loans must have
            been previously added to the manager.

        Returns
        -------
        List[int]
            Indices of the groups from which the loans were removed (0 … K-1).
        """
        if not old_loans:
            _LOG.warning("AntiMan.remove_loans: No loans to remove; skipping.")
            return []
        
        if not isinstance(old_loans, list):
            raise TypeError("Expected a list of LoanRecord objects to remove.")
        
        loan_ids = [loan.loan_id for loan in old_loans]

        idxs = []
        for loan_id in loan_ids:
            try:
                idx, vec, cat_keys = self._index.pop(loan_id)
            except KeyError as exc:  # pragma: no cover
                _LOG.warning(
                    "AntiMan.remove_loans: Loan %s not found in index; skipping removal.", loan_id
                )
                raise KeyError(f"Loan {loan_id} not found") from exc
            self._groups[idx].remove(loan_id, vec, cat_keys)
            idxs.append(idx)
        _LOG.info(
            "AntiMan.remove_loans: Removed %d loans from groups: %s. Current group sizes: %s",
            len(loan_ids),
            idxs,
            self.group_sizes(),
        )
        return idxs
    

    # ------------------------------------------------------------------ #
    #                     internal assignment logic                      #
    # ------------------------------------------------------------------ #
    def _assign_single(self, loan: LoanRecord, vec: np.ndarray) -> int:
        """
        Assign a single loan to the best anticluster based on its feature vector.
        This method finds the group that maximises the distance to the centroid
        while respecting hard constraints on categorical balance.

        Parameters
        ----------
        loan : LoanRecord
            The loan to be assigned.
        vec : np.ndarray
            Feature vector of the loan, already transformed by the vectorizer.

        Returns
        -------
        int
            Index of the group to which the loan was assigned (0 … K-1).

        Raises
        ------
        ValueError
            If the loan's feature vector does not match the expected dimension.
        KeyError
            If the loan's categorical keys do not match the expected hard columns.
        """
        if vec.shape[0] != self._dim:
            # reshape if vectorizer returns 1d
            vec = vec.flatten()
        cat_keys = tuple(int(getattr(loan, c)) for c in self.hard_cols)

        best_score = float('inf')
        best_idx = 0

        for i, g in enumerate(self._groups):
            dist = np.linalg.norm(vec - g.centroid) if g.centroid is not None else 1e6
            penalty = 0.0
            if self.hard_cols:
                limit = self.avg_group_size() + self.size_tolerance
                for val in cat_keys:
                    if g.cat_counts.get(val, 0) + 1 > limit:
                        penalty += (g.cat_counts[val] + 1 - limit)
            score = dist + self.size_tolerance * penalty
            if score < best_score:
                best_score, best_idx = score, i

        self._groups[best_idx].add(loan.loan_id, vec, cat_keys)
        self._index[loan.loan_id] = (best_idx, vec, cat_keys)
        return best_idx
        
    # ------------------------------------------------------------------ #
    #                 helpers to keep scales consistent                  #
    # ------------------------------------------------------------------ #
    def _ensure_dim(self, new_dim: int) -> None:
        """Pad all stored vectors / centroids with zeros if dimension grew."""
        if new_dim <= self._dim:
            return

        pad = (0, new_dim - self._dim)
        for g in self._groups:
            if g.centroid is not None:
                g.centroid = np.pad(g.centroid, pad)

        for lid, (idx, vec, cat_keys) in list(self._index.items()):
            vec = np.pad(vec, pad)
            self._index[lid] = (idx, vec, cat_keys)

        self._dim = new_dim

    def _rescale_all_vectors(self, a: np.ndarray, b: np.ndarray) -> None:
        """
        Apply an affine rescale x_new = a * x_old + b to every stored vector
        and centroid, but only over the first `n_numeric` dimensions.

        Parameters
        ----------
        a : np.ndarray of shape (n_numeric,)
            Multiplicative factors per numeric dimension.
        b : np.ndarray of shape (n_numeric,)
            Additive offsets per numeric dimension.

        Raises
        ------
        ValueError
            If len(a) or len(b) does not equal n_numeric.
        """
        n = self._n_numeric
        # Nothing to do if there are no numeric features
        #TODO: This is an issue, bcs we do need to rescale but n is not representative.
        if n == 0:
            _LOG.warning(
                "AntiMan._rescale_all_vectores: No numeric features to rescale; skipping rescale operation."
                " (a=%s, b=%s)", a, b
            )
            return

        # Ensure the scaling factors match expected dimension
        if a.size != n or b.size != n:
            _LOG.error(
                "AntiMan._rescale_all_vectores: Rescale factors have incorrect length: "
                "expected %d, got a=%s, b=%s", n, a.size, b.size
            )
            raise ValueError(
                f"Cannot rescale: expected factors of length {n}, "
                f"got lengths a={a.size}, b={b.size}"
            )

        # --- 1) Update centroids ---
        for g in self._groups:
            if g.centroid is None or g.centroid.size == 0:
                _LOG.warning(
                    "AntiMan._rescale_all_vectores: Skipping rescale for empty group centroid: %s", g.centroid
                )
                continue
            # centroid is 1d array of length n_total
            g.centroid = self.vectorizer.rescale_features(
                g.centroid[np.newaxis, :],  # shape (1, n_total)
                a, b
            )[0]  # back to 1d

        for loan_id, (idx, vec, cat_keys) in self._index.items():
            if vec.size == 0:
                _LOG.warning(
                    "AntiMan._rescale_all_vectores: Skipping rescale for empty vector of loan %s: %s",
                    loan_id, vec
                )
                continue
            new_vec = self.vectorizer.rescale_features(
                vec[np.newaxis, :],
                a, b
            )[0]
            self._index[loan_id] = (idx, new_vec, cat_keys)
        
        _LOG.info(
            "AntiMan._rescale_all_vectores: Rescaled all vectors and centroids with factors a=%s, b=%s",
        )

    # ----------  monitoring / helpers  ---------- #

    def group_sizes(self) -> List[int]:
        """
        Return a list of current group sizes (number of loans in each group).

        Returns:
        -------
            List[int]: List of sizes of each group (0 … K-1).
        """
        return [g.size for g in self._groups]
    
    def group_centroids(self) -> List[np.ndarray | None]:
        """
        Return a list of current group centroids (feature vectors).

        Returns:
        -------
            List[np.ndarray | None]: List of centroids for each group.
                                     None if the group is empty.
        """
        return [g.centroid.tolist() if g.centroid is not None else None for g in self._groups]

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
        old_centroid = self.group_centroids()

        moved = 0
        while True:
            sizes = self.group_sizes()
            largest, smallest = int(np.argmax(sizes)), int(np.argmin(sizes))
            diff = sizes[largest] - sizes[smallest]
            if diff <= self.size_tolerance:
                break

            best_loan, best_penalty = None, float('inf')
            for loan_id in list(self._groups[largest].members):
                _, vec, cat_keys = self._index[loan_id]
                penalty = 0.0
                limit = self.avg_group_size() + self.size_tolerance
                for val in cat_keys:
                    over = self._groups[smallest].cat_counts.get(val, 0) + 1 - limit
                    if over > 0:
                        penalty += over
                if penalty < best_penalty:
                    best_penalty, best_loan = penalty, loan_id
                if penalty == 0:
                    break

            if best_loan is None:
                _LOG.info("AntiMan.Rebalance: no move possible")
                break

            _, vec, cat_keys = self._index.pop(best_loan)
            self._groups[largest].remove(best_loan, vec, cat_keys)
            self._groups[smallest].add(best_loan, vec, cat_keys)
            self._index[best_loan] = (smallest, vec, cat_keys)
            moved += 1

        _LOG.info("AntiMan.Rebalance completed: %d moved", moved)
        new_centroid = self.group_centroids()
        if old_centroid != new_centroid:
            _LOG.info(
                "AntiMan.Rebalance: centroids changed from %s to %s",
                old_centroid, new_centroid
            )
        else:
            _LOG.info("AntiMan.rebalance: centroids unchanged after rebalancing.")

        return moved

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
        limit = self.avg_group_size() + self.size_tolerance
        for val in cat_keys:
            c = counts.get(int(val), 0)  # convert str to int if needed
            if c + 1 > limit:
                _LOG.debug(
                    "AntiMan._group_accepts: Group %d cannot accept loan with keys %s; would exceed limit %d.",
                    idx, cat_keys, limit
                )
                return False
        # A cat_key is a tuple aligned with self.hard_cols
        return True

    def _largest_and_smallest_groups(self) -> Tuple[int | None, int | None]:
        sizes = self.group_sizes()
        if not any(sizes):
            return None, None
        largest = int(np.argmax(sizes))
        smallest = int(np.argmin(sizes))
        return largest, smallest

