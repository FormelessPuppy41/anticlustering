
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Tuple
import numpy as np

@dataclass
class GroupState:
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


