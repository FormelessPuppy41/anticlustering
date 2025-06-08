from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .loan import LoanRecord



NUMERIC_ATTRS = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "annual_inc",
    "total_rec_prncp",
    "recoveries",
    "total_rec_int",
]

CATEGORICAL_ATTRS = [
    "grade",
    "sub_grade",
    "loan_status",
    "purpose",
    "home_ownership",
    "verification_status",
    "application_type",
]


@dataclass
class LoanVectorizer:
    num_scaler: StandardScaler
    cat_encoder: OneHotEncoder

    @classmethod
    def fit(cls, loans: List["LoanRecord"]) -> "LoanVectorizer":
        X_num = np.array([[getattr(l, a) for a in NUMERIC_ATTRS] for l in loans])
        X_cat = np.array([[getattr(l, a) for a in CATEGORICAL_ATTRS] for l in loans])
        num_scaler = StandardScaler().fit(X_num)
        cat_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(X_cat)
        return cls(num_scaler, cat_encoder)

    def transform(self, loans: List["LoanRecord"]) -> np.ndarray:
        X_num = self.num_scaler.transform(
            [[getattr(l, a) for a in NUMERIC_ATTRS] for l in loans]
        )
        X_cat = self.cat_encoder.transform(
            [[getattr(l, a) for a in CATEGORICAL_ATTRS] for l in loans]
        )
        return np.hstack([X_num, X_cat])
    
    def fit_transform(self, loans: List["LoanRecord"]) -> np.ndarray:
        """Fit the vectorizer and transform the loans in one step."""
        self.num_scaler, self.cat_encoder = self.fit(loans)
        return self.transform(loans)

    # convenience for single-loan use
    def __call__(self, loan: "LoanRecord") -> np.ndarray:
        return self.transform([loan])[0]