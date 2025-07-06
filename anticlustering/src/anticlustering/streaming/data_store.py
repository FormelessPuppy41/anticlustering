# anticlustering/data_state.py

import logging
from typing import Dict, List, Optional

import numpy as np

from ..core.loans.loan import LoanRecord
from ..core.loans.vectorizer import LoanVectorizerConfig, LoanVectorizer
from anticlustering.metrics.dissimilarity_matrix import get_dissimilarity_matrix

logger = logging.getLogger(__name__)


class StreamingDataStore:
    """
    Base store: tracks IDs, feature matrix, and distance matrix.  
    Subclasses implement how data is ingested and removed.
    """

    def __init__(self, feature_dim: int):
        # feature_dim may be ignored by some subclasses
        self._ids: List[str] = []
        self._X: np.ndarray = np.zeros((0, feature_dim))
        self._D: np.ndarray = np.zeros((0, 0))

    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    @property
    def features(self) -> np.ndarray:
        return self._X.copy()

    @property
    def distances(self) -> np.ndarray:
        return self._D.copy()

    def index_of(self, item_id: str) -> int:
        """
        O(n) lookup of an item ID in the store.
        """
        try:
            return self._ids.index(item_id)
        except ValueError as e:
            raise ValueError(f"ID {item_id!r} not in store") from e










class LoanStreamingDataStore(StreamingDataStore):
    """
    Tracks the single source of truth for:
      - loan_ids (in insertion order)
      - scaled feature matrix (N × D)
      - full distance matrix D (N × N)
    """

    def __init__(
        self,
        vectorizer_config: LoanVectorizerConfig,
    ) -> None:
        super().__init__(feature_dim=0)  # placeholder, overwritten by vectorizer.dimension_
        self.vectorizer = LoanVectorizer(vectorizer_config)
        self._ids = []
        self._X = np.zeros((0, self.vectorizer.dimension_))
        self._D = np.zeros((0, 0))
        self._id_to_loan: Dict[str, LoanRecord] = {}


    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    @property
    def features(self) -> np.ndarray:
        return self._X.copy()

    @property
    def distances(self) -> np.ndarray:
        return self._D.copy()

    def add_loans(self, loans: List[LoanRecord]) -> List[str]:
        """
        1) Update scaler & rescale existing features
        2) Transform new loans → X_new
        3) Append to IDs, X
        4) Recompute full distance matrix D
        Returns list of new loan_ids.
        """
        if not loans:
            return []
        
        if not isinstance(loans, list) or not all(isinstance(loan, LoanRecord) for loan in loans):
            raise TypeError("loans must be a list of LoanRecord instances")
        
        self._id_to_loan.update({loan.loan_id: loan for loan in loans})

        # 1) partial update scaler
        a, b = self.vectorizer.partial_update(loans)
        if self._X.size:
            self._X = self.vectorizer.rescale_features(self._X, a, b)
            logger.debug("Rescaled history by a=%s, b=%s", a, b)

        # 2) new features
        X_new = self.vectorizer.transform(loans)  # shape (n_new, D)
        new_ids = [loan.loan_id for loan in loans]

        # 3) append
        self._ids.extend(new_ids)
        self._X = np.vstack([self._X, X_new])

        # 4) recompute D
        self._D = get_dissimilarity_matrix(self._X)
        logger.debug("Recomputed distance matrix: shape %s", self._D.shape)

        return new_ids

    def remove_loans(self, loan_ids: List[str]) -> List[str]:
        """
        Remove loans by ID:
        1) Filter out from IDs & X
        2) Recompute D
        Returns list of actually removed IDs.
        """
        if not loan_ids:
            return []
        
        if not isinstance(loan_ids, list) or not all(isinstance(lid, str) for lid in loan_ids):
            raise TypeError("loan_ids must be a list of strings")
        
        # Remove loans from internal state
        for lid in loan_ids:
            if lid not in self._id_to_loan:
                logger.debug("remove_loans: loan_id %s not found in store", lid)
            else:
                del self._id_to_loan[lid]

        set_remove = set(loan_ids)
        keep_mask = [lid not in set_remove for lid in self._ids]
        removed = [lid for lid in self._ids if lid in set_remove]

        if not removed:
            logger.debug("remove_loans: none of %s found", loan_ids)
            return []

        self._ids = [lid for lid, keep in zip(self._ids, keep_mask) if keep]
        self._X = self._X[keep_mask, :]

        self._D = get_dissimilarity_matrix(self._X)
        logger.debug("After removal, recomputed D: shape %s", self._D.shape)

        return removed

    def index_of(self, loan_id: str) -> int:
        """
        O(1) index lookup via internal list.
        Raises ValueError if not found.
        """
        try:
            return self._ids.index(loan_id)
        except ValueError as e:
            raise ValueError(f"Loan ID {loan_id!r} not in store") from e
