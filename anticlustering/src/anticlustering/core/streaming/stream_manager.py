
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np

from ..loans.loan import LoanRecord

_LOG = logging.getLogger(__name__)

# anticlustering/stream_manager.py
from ...streaming.data_store import LoanStreamingDataStore
from ...streaming.group_state import GroupState
from ..loans.loan import LoanRecord
from ...core.loans.vectorizer import LoanVectorizerConfig
from ..online.online_base import BaseOnlineSolver


logger = logging.getLogger(__name__)

class AnticlusterManager:
    """
    Orchestrates LoanStreamingDataStore + BaseOnlineSolver.

    Must first remove old loans, then assign new loans, and finally rebalance.
    """

    def __init__(
        self,
        solver: BaseOnlineSolver,
        vectorizer_config: LoanVectorizerConfig,
        hard_balance_cols: List[str] | None = None
    ):
        if not isinstance(solver, BaseOnlineSolver):
            raise TypeError("solver must implement BaseOnlineSolver")
        self.store = LoanStreamingDataStore(vectorizer_config)
        self.solver = solver
        self.assignments: Dict[str,int] = {}
        self._groups: list[GroupState] = [GroupState() for _ in range(self.solver.config.n_clusters)]

        self.hard_balance_cols = hard_balance_cols or []

        # Use the status to track the 'calls' in the pipeline:
        # 0 = ready for departures - initialized,
        # 1 = ready for arrivals - departures have been processed,
        # 2 = ready for rebalanced - arrivals have been processed,
        # 3 = ready for new departures - rebalancing has been processed,
        self.status = 0

    def on_departure(self, loans: List[LoanRecord]) -> Dict[str,int]:
        if self.status != 0 and self.status != 3:
            raise RuntimeError(f"Cannot process departures at this stage (status:{self.status}). The stage must be 0 or 3.")
        
        old_ids = [ln.loan_id for ln in loans]
        self.store.remove_loans(old_ids)
        self.assignments = self.solver.remove_old(self.store, self.assignments, old_ids)
        self._rebuild_group_states()

        self.status = 1  # After departures, we can process arrivals
        return self.assignments

    def on_arrival(self, loans: List[LoanRecord]) -> Dict[str,int]:
        if self.status != 1:
            raise RuntimeError(f"Cannot process arrivals at this stage (status:{self.status}). The stage must be 1.")
        
        new_ids = self.store.add_loans(loans)
        self.assignments = self.solver.assign_new(self.store, self.assignments, new_ids)
        self._rebuild_group_states()

        self.status = 2  # After arrivals, we can rebalance
        return self.assignments

    def on_rebalance(self) -> Dict[str,int]:
        if self.status != 2:
            raise RuntimeError(f"Cannot rebalance at this stage (status:{self.status}). The stage must be 2.")
        
        self.assignments = self.solver.rebalance(self.store, self.assignments)

        self.status = 3  # After rebalancing, we can process new departures
        return self.assignments

    def get_assignments(self) -> Dict[str,int]:
        return dict(self.assignments)

    def finalize(self) -> None:
        self.solver.finalize()

    @property
    def centroids(self) -> List[np.ndarray|None]:
        return [g.centroid for g in self._groups]

    @property
    def group_sizes(self) -> List[int]:
        return [g.size for g in self._groups]
    
    def _rebuild_group_states(self):
        # after every add/remove/rebalance, rebuild from scratch:
        for g in self._groups:
            g.size = 0; g.centroid = None; g.members.clear(); g.cat_counts.clear()
        for loan_id, feat_vec in zip(self.store.ids, self.store.features):
            grp = self.assignments[loan_id]
            cat_keys = tuple(int(getattr(self.store._id_to_loan[loan_id], c)) for c in self.hard_balance_cols)
            self._groups[grp].add(loan_id, feat_vec, cat_keys)

