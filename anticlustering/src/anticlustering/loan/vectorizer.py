# src/loan/vectorize.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import logging

from typing import Dict, Optional
import datetime as _dt

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .loan import LoanRecord   # your dataclass, unchanged

_LOG = logging.getLogger(__name__)

@dataclass
class LoanVectorizer:
    """Vectorises loans according to lists received at run-time."""

    def __init__(
            self,
            kaggle_columns: Dict[str, Sequence[str]] = None,
            num_scaler: Optional[StandardScaler] = None,
            cat_encoder: Optional[OneHotEncoder] = None,
        ) -> None:
        """
        Initialize the LoanVectorizer with column specifications and transformers.
        This constructor sets up the vectorizer with the provided column specifications
        and optional transformers for numeric and categorical features.
        It extracts the relevant feature lists from the `kaggle_columns` dictionary
        and initializes the numeric scaler and categorical encoder if provided.
        This allows the vectorizer to transform `LoanRecord` instances into a dense
        feature matrix suitable for machine learning tasks.
        Parameters

        ----------
        kaggle_columns : Dict[str, Sequence[str]]
            A dictionary containing lists of column names for different feature types.
            It can include keys like "numeric_columns", "log_numeric_columns",
            "percentage_columns", "special_numeric_columns", "date_columns",
            "ordinal_columns", and "categorical_columns".
        num_scaler : Optional[StandardScaler]
            An optional `StandardScaler` instance for scaling numeric features.
            If not provided, a new scaler will be created during fitting.
        cat_encoder : Optional[OneHotEncoder]
            An optional `OneHotEncoder` instance for encoding categorical features.
            If not provided, a new encoder will be created during fitting.
        """
        # --- store spec and fallback ---
        self.kaggle_columns = kaggle_columns or {}
        # --- extract feature lists with safe defaults ---
        self.numeric_attrs = list(self.kaggle_columns.get("numeric_columns") or [])
        self.log_numeric_attrs = list(self.kaggle_columns.get("log_numeric_columns") or [])
        self.percentage_numeric_attrs = list(self.kaggle_columns.get("percentage_columns") or [])
        self.special_numeric_attrs = list(self.kaggle_columns.get("special_numeric_columns") or [])
        self.datetime_attrs = list(self.kaggle_columns.get("date_columns") or [])
        self.ordinal_attrs = self.kaggle_columns.get("ordinal_columns") or {}
        self.categorical_attrs = list(self.kaggle_columns.get("categorical_columns") or [])

        self.num_to_scale = (
            self.numeric_attrs +
            self.log_numeric_attrs +
            self.percentage_numeric_attrs +
            self.special_numeric_attrs +
            self.datetime_attrs +
            list(self.ordinal_attrs.keys())
        )

        # --- store transformers ---
        self._num_scaler = num_scaler
        self._cat_encoder = cat_encoder

        
    # ---------- factory ------------------------------------------------------
    @classmethod
    def fit(
        cls,
        loans: List[LoanRecord],
        kaggle_columns: Optional[Dict[str, Sequence[str]]] = None,
    ) -> LoanVectorizer:
        inst = cls(kaggle_columns)

        # Build all numeric blocks
        blocks: List[np.ndarray] = []
        if inst.numeric_attrs:
            blocks.append(_extract_matrix(loans, inst.numeric_attrs))
        if inst.log_numeric_attrs:
            Xlog = _extract_matrix(loans, inst.log_numeric_attrs)
            blocks.append(np.log1p(Xlog))
        if inst.percentage_numeric_attrs:
            blocks.append(_extract_matrix(loans, inst.percentage_numeric_attrs))
        if inst.special_numeric_attrs:
            blocks.append(_extract_matrix(loans, inst.special_numeric_attrs))
        if inst.datetime_attrs:
            Xdate = _extract_matrix(loans, inst.datetime_attrs)
            blocks.append(cls._parse_dates_to_numeric(Xdate))
        if inst.ordinal_attrs:
            blocks.append(_extract_matrix(loans, inst.ordinal_attrs))

        X_all = np.hstack(blocks) if blocks else np.empty((len(loans), 0))
        if X_all.shape[1] > 0:
            num_scaler = StandardScaler()
            num_scaler.partial_fit(X_all)
        else: 
            num_scaler = None
        
        Xcat = _extract_matrix(loans, inst.categorical_attrs)
        if Xcat.size:
            cat_encoder = OneHotEncoder(handle_unknown="ignore").fit(Xcat)
        else:
            _LOG.warning("fit: No categorical attrs; skipping OneHotEncoder fit.")
            cat_encoder = None

        return cls(kaggle_columns, num_scaler, cat_encoder)   

    # ---------- public API ---------------------------------------------------
    def transform(self, loans: List[LoanRecord]) -> np.ndarray:
        raw_blocks: List[np.ndarray] = []
        if self.numeric_attrs:
            raw_blocks.append(_extract_matrix(loans, self.numeric_attrs))
        if self.log_numeric_attrs:
            Xlog = _extract_matrix(loans, self.log_numeric_attrs)
            raw_blocks.append(np.log1p(Xlog))
        if self.percentage_numeric_attrs:
            raw_blocks.append(_extract_matrix(loans, self.percentage_numeric_attrs))
        if self.special_numeric_attrs:
            raw_blocks.append(_extract_matrix(loans, self.special_numeric_attrs))
        if self.datetime_attrs:
            Xdate = _extract_matrix(loans, self.datetime_attrs)
            raw_blocks.append(self._parse_dates_to_numeric(Xdate))
        if self.ordinal_attrs:
            raw_blocks.append(_extract_matrix(loans, self.ordinal_attrs))

        X_raw = np.hstack(raw_blocks) if raw_blocks else np.empty((len(loans), 0))
        if self._num_scaler:
            X_num = self._num_scaler.transform(X_raw)
        else:
            X_num = X_raw

        if self.categorical_attrs and self._cat_encoder:
            X_cat = self._cat_encoder.transform(
                _extract_matrix(loans, self.categorical_attrs)
            ).toarray()
        else:
            X_cat = np.empty((len(loans), 0))

        # _LOG.info(
        #     "transform: Transformed %d loans → %d numeric + %d categorical features. \n The transformed loan ids are: %s",
        #     len(loans), X_num.shape[1], X_cat.shape[1], [lo.loan_id for lo in loans]
        # )
        # Log the difference between the original numeric features and the transformed ones
        # _LOG.info(
        #     "transform: Showing the difference for the first loan in: Original numeric features: %s, Transformed numeric features: %s",
        #     [getattr(loans[0], attr) for attr in self.num_to_scale],
        #     X_num[0, :len(self.num_to_scale)].tolist()
        # )
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
        # No loans or no numeric features → identity
        if not loans or not self.num_to_scale or not self._num_scaler:
            _LOG.warning(
                "partial_update: No loans or no numeric features; returning identity factors. Specifics: "
                "loans=%s, num_to_scale=%s, num_scaler=%s",
                len(loans), self.num_to_scale, self._num_scaler
            )
            N = len(self.num_to_scale)
            return np.ones(N, dtype=float), np.zeros(N, dtype=float)

        # 1) Remember old scaler params
        old_mean, old_scale = (
            self._num_scaler.mean_.copy(),
            self._num_scaler.scale_.copy(),
        )

        # helper to spot NaNs in a 2D block
        def check_nan_in_cols(block: np.ndarray) -> list[int]:
            # returns list of column-indices that contain any NaN
            return [i for i in range(block.shape[1]) if np.isnan(block[:, i]).any()]


        # 2) Build the *full* matrix exactly as in fit/transform
        blocks: List[np.ndarray] = []

        # raw numeric
        if self.numeric_attrs:
            blocks.append(_extract_matrix(loans, self.numeric_attrs))

        # log-numeric
        if self.log_numeric_attrs:
            Xl = _extract_matrix(loans, self.log_numeric_attrs)
            blocks.append(np.log1p(Xl))

        # percentages
        if self.percentage_numeric_attrs:
            blocks.append(_extract_matrix(loans, self.percentage_numeric_attrs))

        # special numeric
        if self.special_numeric_attrs:
            blocks.append(_extract_matrix(loans, self.special_numeric_attrs))

        # dates → timestamp
        if self.datetime_attrs:
            Xd = _extract_matrix(loans, self.datetime_attrs)
            Xd = self._parse_dates_to_numeric(Xd)
            bad = check_nan_in_cols(Xd)
            if bad:
                _LOG.warning(
                    "partial_update: NaNs found in datetime columns %r; parsing may yield NaNs",
                    [self.datetime_attrs[i] for i in bad]
                )
            blocks.append(Xd)

        # ordinals (already numeric)
        if self.ordinal_attrs:
            Xd = _extract_matrix(loans, list(self.ordinal_attrs.keys()))
            bad = check_nan_in_cols(Xd)
            if bad:
                _LOG.warning(
                    "partial_update: NaNs found in ordinal columns %r",
                    [list(self.ordinal_attrs.keys())[i] for i in bad]
                )
            blocks.append(Xd)

        # stack them
        X_full = (
            np.hstack(blocks)
            if blocks and any(b.size for b in blocks)
            else np.empty((len(loans), 0))
        )

        # more informative NaN‐check: locate exactly which loans and which attrs
        mask = np.isnan(X_full)
        if mask.any():
            # find unique bad rows and columns
            bad_rows, bad_cols = np.where(mask)
            bad_rows = sorted(set(bad_rows))
            bad_cols = sorted(set(bad_cols))

            # map row‐indices back to loan_ids and objects
            bad_loan_ids = [loans[i].loan_id for i in bad_rows]
            bad_loan_objs = [loans[i] for i in bad_rows]

            # map col‐indices back to attribute names
            bad_attr_names = [self.num_to_scale[c] for c in bad_cols]

            # pick the very first bad loan
            first_i = bad_rows[0]
            first_loan = loans[first_i]
            first_vector = X_full[first_i, :]

            _LOG.error(
                "partial_update: NaNs found in numeric features for loan_ids %s on attrs %s",
                bad_loan_ids, bad_attr_names
            )
            _LOG.error("partial_update: First bad LoanRecord: %r", first_loan)
            _LOG.error("partial_update: Corresponding full‐vector (with NaNs): %s", first_vector.tolist())

            raise ValueError(
                f"partial_update: NaNs in loans {bad_loan_ids} on numeric attrs {bad_attr_names}."
            )


        # 3) Partial-fit on that full block
        self._num_scaler.partial_fit(X_full)
        new_mean, new_scale = (
            self._num_scaler.mean_,
            self._num_scaler.scale_,
        )
        _LOG.info(
            "Partial update: old mean=%s, old scale=%s; new mean=%s, new scale=%s",
            old_mean, old_scale, new_mean, new_scale
        )
        # 4) Compute migration factors
        a = np.ones_like(old_scale)
        b = np.zeros_like(old_mean)

        # only update where new_scale nonzero
        mask = new_scale != 0
        a[mask] = old_scale[mask] / new_scale[mask]
        b[mask] = (old_mean[mask] - new_mean[mask]) / new_scale[mask]

        _LOG.info("partial_update: safe rescale factors a=%s, b=%s", a.tolist(), b.tolist())
        return a, b
    
    def rescale_features(
        self,
        X: np.ndarray,
        a: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """
        Given a dense feature matrix X of shape (n_samples, n_features_total),
        apply x_new = a * x_old + b *only* on the numeric block (the first
        len(self.num_to_scale) columns), leaving any categorical columns
        untouched. Returns a new array.
        
        Raises if a/b length doesn’t match the number of numeric dimensions.
        """
        n_num = len(self.num_to_scale)
        # nothing to do if no scaler or no numeric features
        if self._num_scaler is None or n_num == 0:
            _LOG.info(
                f"rescale_features: No numeric features or scaler; returning original X (shape {X.shape})"
            )
            return X.copy()

        # sanity check
        if a.shape[0] != n_num or b.shape[0] != n_num:
            raise ValueError(
                f"rescale_features: Rescale mismatch: expected a/b of length {n_num}, "
                f"got lengths {a.shape[0]}, {b.shape[0]}"
            )
        # perform affine update on the first n_num columns
        a = np.nan_to_num(a, nan=1.0, posinf=1.0, neginf=1.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

        X_new = X.copy()
        X_new[:, :n_num] = X[:, :n_num] * a + b

        return X_new
    
    @property
    def n_numeric(self) -> int:
        """Number of numeric features to scale."""
        return len(self.num_to_scale)
    
    # ------ date parsing helper ----------------------------------------------
    @staticmethod
    def _parse_dates_to_numeric(dates: np.ndarray) -> np.ndarray:
        """
        Convert a 2D array of datetime.date or datetime.datetime objects
        to an array of floats (seconds since epoch).
        """
        # expect shape (n_samples, n_date_cols)
        if dates.size == 0:
            return dates.astype(float)
        # ensure 2D
        dates = dates.reshape(-1, dates.shape[-1])
        n, m = dates.shape
        result = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                d = dates[i, j]
                if isinstance(d, _dt.datetime):
                    result[i, j] = d.timestamp()
                elif isinstance(d, _dt.date):
                    dt = _dt.datetime(d.year, d.month, d.day)
                    result[i, j] = dt.timestamp()
                else:
                    result[i, j] = np.nan
        return result
    

def _extract_matrix(loans: List[LoanRecord], attrs: Sequence[str]) -> np.ndarray:
    """
    Helper to build a 2D array of loan attributes.

    Returns shape (n_samples, len(attrs)), or empty array if attrs empty.
    """
    if not attrs:
        return np.empty((len(loans), 0))
    return np.array([[getattr(l, a) for a in attrs] for l in loans])

