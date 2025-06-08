# src/loan/vectorize.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import logging

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .loan import LoanRecord   # your dataclass, unchanged

_LOG = logging.getLogger(__name__)

@dataclass
class LoanVectorizer:
    """Vectorises loans according to lists received at run-time."""
    numeric_attrs: Sequence[str]
    categorical_attrs: Sequence[str]
    _num_scaler: StandardScaler
    _cat_encoder: OneHotEncoder

    # ---------- factory ------------------------------------------------------
    @classmethod
    def fit(
            cls,
            loans: List[LoanRecord],
            numeric_attrs: Sequence[str],
            categorical_attrs: Sequence[str]
        ) -> "LoanVectorizer":

        X_num = np.array([[getattr(l, a) for a in numeric_attrs] for l in loans])
        X_cat = np.array([[getattr(l, a) for a in categorical_attrs] for l in loans])
        
        num_scaler = StandardScaler().fit(X_num)
        cat_encoder = OneHotEncoder(handle_unknown="ignore").fit(X_cat)

        return cls(numeric_attrs, categorical_attrs, num_scaler, cat_encoder)


    # ---------- public API ---------------------------------------------------
    def transform(self, loans: list[LoanRecord]) -> np.ndarray:
        """Return dense feature matrix of shape (n_samples, n_features)."""

        # ----- numeric block (always dense) -----
        X_num = self._num_scaler.transform(
            [[getattr(l, a) for a in self.numeric_attrs] for l in loans]
        )

        # ----- categorical block (sparse → dense) -----
        X_cat_sparse = self._cat_encoder.transform(
            [[getattr(l, a) for a in self.categorical_attrs] for l in loans]
        )
        # make it a 2-D dense ndarray, even when n_samples == 1
        X_cat = X_cat_sparse.toarray()

        _LOG.info(
            "Transformed %d loans → %d numeric + %d categorical features",
            len(loans), X_num.shape[1], X_cat.shape[1],
        )
        _LOG.debug("Sample numeric: %s", X_num[0])
        _LOG.debug("Sample categorical: %s", X_cat[0])

        # ----- concatenate blocks -----
        return np.hstack([X_num, X_cat])

    
    def __call__(self, loan: LoanRecord) -> np.ndarray:           # convenience
        return self.transform([loan])[0]


    def partial_update(self, loans: Sequence[LoanRecord]) -> tuple[np.ndarray, np.ndarray]:
        """
        Update numeric scaler with *current* batch and return two arrays `(a, b)`
        such that any **old** vector `x_old` can be mapped to the **new** scale
        via

            x_new = a * x_old + b        (vectorised)

        Parameters
        ----------
        loans : Sequence[LoanRecord]
            Batch that has **already** been transformed with the *previous*
            scaler; we now use it to improve the scaler for the future.

        Returns
        -------
        (a, b) : tuple[np.ndarray, np.ndarray]
            Multiplicative (`a`) and additive (`b`) factors per numeric column.
        """
        if not loans:
            return (
                np.ones(len(self.numeric_attrs), dtype=float),
                np.zeros(len(self.numeric_attrs), dtype=float),
            )

        # --- 1) remember previous parameters --------------------------- #
        old_mean  = self._num_scaler.mean_.copy()
        old_scale = self._num_scaler.scale_.copy()

        # --- 2) fit on current batch ----------------------------------- #
        X_num = np.array([[getattr(l, a) for a in self.numeric_attrs] for l in loans])
        self._num_scaler.partial_fit(X_num)

        new_mean, new_scale = self._num_scaler.mean_, self._num_scaler.scale_

        # --- 3) factors to migrate existing data ----------------------- #
        a = old_scale / new_scale
        b = (old_mean - new_mean) / new_scale
        return a, b
    
    @property
    def n_numeric(self) -> int:
        """Number of numeric features."""
        return len(self.numeric_attrs)