# anticlustering/core/anticluster.py

from __future__ import annotations
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..loans.loan import LoanRecord
from ..loans.vectorizer import LoanVectorizer

_LOG = logging.getLogger(__name__)

@dataclass
class _GroupState:
    """Internal running stats for one anticluster."""
    size: int = 0
    centroid: np.ndarray | None = None
    members: set[str] = field(default_factory=set)
    cat_counts: Counter[int] = field(default_factory=Counter)

    def add(self, loan_id: str, vec: np.ndarray, cat_keys: Tuple[int, ...]) -> None:
        """O(1) update on adding one member."""
        if self.size == 0:
            self.centroid = vec.copy()
        else:
            self.centroid += (vec - self.centroid) / (self.size + 1)
        self.size += 1
        self.members.add(loan_id)
        self.cat_counts.update(cat_keys)

    def remove(self, loan_id: str, vec: np.ndarray, cat_keys: Tuple[int, ...]) -> None:
        """O(1) update on removing one member."""
        if loan_id not in self.members:
            raise KeyError(f"Loan {loan_id} not in this group")
        self.members.remove(loan_id)
        self.cat_counts.subtract(cat_keys)
        self.size -= 1
        if self.size == 0:
            self.centroid = None
        else:
            self.centroid -= (vec - self.centroid) / self.size




# anticlustering/stream_manager.py

import logging
from typing import Any, Dict, List

from ...streaming.data_store import StreamingDataStore
from ...streaming.group_state import GroupState
from ..loans.loan import LoanRecord
from ...core.loans.vectorizer import LoanVectorizerConfig
from ..online.online_base import BaseOnlineSolver


logger = logging.getLogger(__name__)

class AnticlusterManager:
    """
    Orchestrates StreamingDataStore + BaseOnlineSolver.
    """

    def __init__(
        self,
        solver: BaseOnlineSolver,
        vectorizer_config: LoanVectorizerConfig,
        hard_balance_cols: List[str] | None = None
    ):
        if not isinstance(solver, BaseOnlineSolver):
            raise TypeError("solver must implement BaseOnlineSolver")
        self.store = StreamingDataStore(vectorizer_config)
        self.solver = solver
        self.assignments: Dict[str,int] = {}
        self._groups: list[GroupState] = [GroupState() for _ in range(self.solver.config.n_clusters)]

        self.hard_balance_cols = hard_balance_cols or []


    def on_arrival(self, loans: List[LoanRecord]) -> Dict[str,int]:
        new_ids = self.store.add_loans(loans)
        self.assignments = self.solver.assign_new(self.store, self.assignments, new_ids)
        self._rebuild_group_states()
        return self.assignments

    def on_departure(self, loans: List[LoanRecord]) -> Dict[str,int]:
        old_ids = [ln.loan_id for ln in loans]
        self.store.remove_loans(old_ids)
        self.assignments = self.solver.remove_old(self.store, self.assignments, old_ids)
        self._rebuild_group_states()
        return self.assignments

    def on_rebalance(self) -> Dict[str,int]:
        self.assignments = self.solver.rebalance(self.store, self.assignments)
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

