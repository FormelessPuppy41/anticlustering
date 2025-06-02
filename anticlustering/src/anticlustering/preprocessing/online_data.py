# features.py
from __future__ import annotations
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

# ------------- 1. Define schema --------------------------------------------
NUMERIC:  List[str] = [
    "loan_amnt", "annual_inc", "dti", "int_rate", "installment",
    "fico_range_low", "fico_range_high",
]
# ordinal category: A < B < ... < G
GRADE:    List[str] = ["grade"]
# unordered categories
CATEG:    List[str] = [
    "home_ownership", "purpose", "verification_status", "term",
]

# ------------- 2. Column pipelines -----------------------------------------
numeric_pipe = Pipeline(
    steps=[
        ("impute",  SimpleImputer(strategy="median")),
        ("scale",   StandardScaler()),
    ]
)

grade_categories = [["A", "B", "C", "D", "E", "F", "G"]]
grade_pipe = Pipeline(
    steps=[
        ("impute",  SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=grade_categories)),
        ("scale",   StandardScaler()),
    ]
)

categorical_pipe = Pipeline(
    steps=[
        ("impute",  SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

# ------------- 3. Full transformer -----------------------------------------
vectorizer = ColumnTransformer(
    transformers=[
        ("num",    numeric_pipe, NUMERIC),
        ("grade",  grade_pipe,   GRADE),
        ("cat",    categorical_pipe, CATEG),
    ]
)

# ------------- 4. Convenience wrapper --------------------------------------
class FeatureVectorizer:
    """Fits on historical LendingClub data and transforms new records."""

    def __init__(self):
        self._pipe: Pipeline | None = None   # filled in fit()

    # -- offline -------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureVectorizer":
        self._pipe = Pipeline(steps=[("vectorizer", vectorizer)])
        self._pipe.fit(df)
        return self

    # -- online --------------------------------------------------------------
    def transform(self, record: Dict[str, Any]) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("call fit() first or load a fitted model")
        df = pd.DataFrame([record])           # single-row DataFrame
        return self._pipe.transform(df)[0]    # → 1-D numpy array

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self._pipe, path)

    @classmethod
    def load(cls, path: str) -> "FeatureVectorizer":
        import joblib
        obj = cls()
        obj._pipe = joblib.load(path)
        return obj
