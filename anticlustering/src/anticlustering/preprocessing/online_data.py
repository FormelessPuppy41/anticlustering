# lending_club_preprocessor.py
"""
End-to-end preprocessing for Lending Club 2014-2018 data.

Highlights
----------
* User lists **exactly** which columns to keep.
* Each column can be tagged as:
    - numeric
    - percentage
    - date
    - ordinal   (with explicit order)
    - categorical
    - passthrough (copied, never scaled)
* Pipelines for every tag → one ColumnTransformer → single, sparse / dense matrix.

Example
-------
>>> COLS = [
...     "id", "loan_amnt", "int_rate", "grade", "sub_grade",
...     "term", "issue_d", "earliest_cr_line", "target"
... ]
>>> prep = LendingClubPreprocessor(
...     keep_columns       = COLS,
...     date_columns       = ["issue_d", "earliest_cr_line"],
...     percentage_columns = ["int_rate"],
...     ordinal_columns    = {
...         "grade":      list("ABCDEFG"),               # A<B<…<G
...         "sub_grade":  [f"{g}{i}"                    # A1<A2<…<G5
...                        for g in "ABCDEFG" for i in range(1, 6)]
...     },
...     passthrough_columns = ["id"],                    # keep raw
... )
>>> X_train, y_train = prep.fit_transform(df_train, target="target")
>>> X_test,  y_test  = prep.transform(df_test,  target="target")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
)
from ..lending_club.core.loan import LoanRecord
#TODO:
# 0) Sentinal values: think about imputation of 'missing' indicator feature for nan values. 
# 1) Remove heavy skew or outliers using winsorize or log-transforms. (monetary values)
# 2) Categorical drift: Map rare/novel categories to 'other' or 'unknown' (or freeze categories until retraining).
# 3) Text fields: Usually too messy for information gain. Drop them?

# a) 'loan_status' should be passthrough bcs it is unknown at cluster time. But we do use it for replaying their lifetime.

# --------------------------------------------------------------------------- #
#                           Helper transformers                               #
# --------------------------------------------------------------------------- #
def _as_1d(arr):
    """Flatten any 1- or 2-D array‐like to 1-D numpy float array."""
    return np.asarray(arr).reshape(-1)

def _ensure_2d(arr):
    """If arr is 1-D make it (n,1); otherwise leave unchanged."""
    a = np.asarray(arr)
    return a.reshape(-1, 1) if a.ndim == 1 else a

def _parse_percent(arr):
    """'13.4%' → 0.134 (NaNs preserved)."""
    flat = _as_1d(arr)
    series = pd.to_numeric(
        pd.Series(flat).astype(str).str.rstrip("%"),
        errors="coerce"
    )
    return (series / 100.0).to_numpy(dtype=float).reshape(-1, 1)

def _parse_date(arr):
    """Messy strings → days since Unix epoch (float, NaN allowed)."""
    flat = _as_1d(arr)
    dt   = pd.to_datetime(pd.Series(flat), errors="coerce", dayfirst=True)
    print(dt)
    return dt

def _parse_term(arr):
    """Extract integer months from term column: ' 36 months' -> 36."""
    flat = _as_1d(arr)
    return pd.Series(flat).str.extract(r'(\d+)')[0].astype(float).to_numpy().reshape(-1, 1)

def _parse_log(arr):
    """Apply log1p to numeric values, preserving NaNs."""
    flat = _as_1d(arr)
    return np.log1p(np.nan_to_num(flat, nan=0.0)).reshape(-1, 1)


class NamedFunctionTransformer(FunctionTransformer):
    """
    FunctionTransformer with get_feature_names_out.
    """
    def __init__(self, func=None, feature_name=None, **kwargs):
        super().__init__(func=func, **kwargs)
        self.feature_name = feature_name

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray([self.feature_name])

class NamedStandardScaler(StandardScaler):
    """
    StandardScaler that preserves feature name for get_feature_names_out.
    """
    def __init__(self, feature_name):
        super().__init__()
        self.feature_name = feature_name

    def get_feature_names_out(self, input_features=None):
        return np.array([self.feature_name])

# --------------------------------------------------------------------------- #
#                             Main pre-processor                              #
# --------------------------------------------------------------------------- #

#
# TODO:
#
# - What i think should be done is:
# - 1) Make this a general preprocessor. Just formatting, no scaling (nor log)
# - 2) Add somewhere in the online anticluster pipeline an option to scale features (vectorised)
# - 3) This way, the input to the Stream/Simulator/Amortization is always the original data (scales).
# - 3) Which allows us to keep interpretability of the original data. But we can still scale it for the distance metric by
# - 3) scaling the features within the OnlineAnticluster class. (This must be vectorised, not per loan record to avoid runtime issues.)

class LendingClubPreprocessor(BaseEstimator, TransformerMixin):
    """
    Clean & scale Lending Club data with fine-grained column control.

    Parameters
    ----------
    keep_columns : List[str]
        **Whitelist** – only these columns survive the first step.
    scale : bool, default=False
        If True, apply StandardScaler to numeric features (numeric, percentage,
        date, ordinal, special numeric, log numeric).  If False, only impute
        missing values (no scaling).
    numeric_impute_value : float, default 0.0
        Constant for missing numerics / percentages / dates after conversion.
    date_columns : List[str], default None
    percentage_columns : List[str], default None
    ordinal_columns : Dict[str, List[str]], default None
        Mapping *col → ordered list* (explicit rank).
    categorical_columns : List[str], default None
    passthrough_columns : List[str], default None
        Columns copied verbatim and never scaled.
    special_numeric_columns : List[str], default None
        Numeric columns that need special parsing (e.g. 'term' as '36 months').
    log_numeric_columns : List[str], default None
        Numeric columns to apply log1p transformation (e.g. monetary values).
    numeric_feature_columns : List[str]
        The subset of raw numeric columns (from keep_columns) that you want to
        use as *features* in the streaming anticluster distance metric.
        These columns must appear in keep_columns, and must be parseable as floats.
    feature_weights : Optional[Dict[str,float]], default=None
        If provided, a mapping {col_name→weight} (same keys as numeric_feature_columns).
        Used *after* scaling to re-weight each dimension. Missing keys default to 1.0.
    
    """

    # ---- constructor ---------------------------------------------------- #
    def __init__(
        self,
        keep_columns                : List[str],
        numeric_feature_columns     : List[str],
        scale                       : bool                  = False,
        numeric_impute_value        : float                 = 0.0,
        date_columns                : Optional[List[str]]   = None,
        percentage_columns          : Optional[List[str]]   = None,
        ordinal_columns             : Optional[Dict[str, List[str]]] = None,
        categorical_columns         : Optional[List[str]]   = None,
        passthrough_columns         : Optional[List[str]]   = None,
        special_numeric_columns     : Optional[List[str]]   = None,   
        log_numeric_columns         : Optional[List[str]]   = None,   
        feature_weights             : Optional[Dict[str, float]] = None,
    ) -> None:
        self.scale                      = scale
        self.keep_columns               = keep_columns
        self.numeric_impute_value       = numeric_impute_value

        self.date_columns               = date_columns or []
        self.percentage_columns         = percentage_columns or []
        self.ordinal_columns            = ordinal_columns or {}
        self.categorical_columns        = categorical_columns or []
        self.passthrough_columns        = passthrough_columns or []
        self.special_numeric_columns    = special_numeric_columns or []   
        self.log_numeric_columns        = log_numeric_columns or []
        
        if not numeric_feature_columns:
            raise ValueError("numeric_feature_columns must be provided.")
        self.numeric_feature_columns    = numeric_feature_columns
        self.feature_weights            = feature_weights or {}
        self._feature_weights_array     = np.array(
            [self.feature_weights.get(col, 1.0) for col in self.numeric_feature_columns],
            dtype=float
        )

        self.feature_scaler: StandardScaler | None = None

        self._num_feats: List[str] = []
        self._pipeline: Optional[Pipeline] = None

        self._check_unique_tags()
        self._check_presence_of_tags()



    # --------------------------------------------------------------------- #
    #                           Public interface                             #
    # --------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LendingClubPreprocessor":
        X = self._select_and_validate(X)

        raw_numeric_block = X[self.numeric_feature_columns].copy()
        if self.scale:
            self.feature_scaler = StandardScaler().fit(raw_numeric_block)
        else: 
            self.feature_scaler = None

        self._identify_remaining_numeric(X)

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X)

        return self

    def transform(
        self, 
        X           : pd.DataFrame, 
        target      : Optional[str] = None,
        return_df   : bool = False
    ) -> Tuple[np.ndarray, Optional[pd.Series]]:
        if self._pipeline is None:
            raise RuntimeError("Must call `fit` before `transform`.")
        
        X = self._select_and_validate(X)
        features = self._pipeline.transform(X)
        y = None if target is None else X[target].values

        if return_df:
            feature_names   = self._pipeline.named_steps["prep"].get_feature_names_out()
            feature_names   = self._clean_column_names(feature_names)  # Clean names
            features        = pd.DataFrame(features, columns=feature_names, index=X.index)

        return features, y

    def fit_transform(
        self, 
        X           : pd.DataFrame, 
        target      : Optional[str] = None,
        return_df   : bool = False
    ) -> Tuple[np.ndarray, Optional[pd.Series]]:
        return self.fit(X).transform(X, target=target, return_df=return_df)

    def scale_loan_features(self, loan: "LoanRecord") -> np.ndarray:
        """
        Scale a single LoanRecord instance to a feature vector.

        Parameters
        ----------
        loan : LoanRecord
            The loan record to scale.

        Returns
        -------
        np.ndarray
            Scaled feature vector.
        """
        # 1. Build raw vector
        raw_vals = np.array(
            [getattr(loan, col) for col in self.numeric_feature_columns],
            dtype=float,
        ).reshape(1, -1)

        # 2. Scale if needed
        if self.feature_scaler is not None:
            scaled = self.feature_scaler.transform(raw_vals)[0]
        else:
            scaled = raw_vals[0]

        # 3. Apply weights
        return scaled * self._feature_weights_array
    
    # --------------------------------------------------------------------- #
    #                           Internal helpers                             #
    # --------------------------------------------------------------------- #
    # 1. Fast fail if required columns missing
    def _select_and_validate(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.keep_columns) - set(X.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {', '.join(missing)}")
        return X[self.keep_columns].copy()

    # 2. Everything not user-tagged nor passthrough & not object/string → numeric
    def _identify_remaining_numeric(self, X: pd.DataFrame) -> None:
        tagged = (
            set(self.date_columns)
            | set(self.percentage_columns)
            | set(self.ordinal_columns)
            | set(self.categorical_columns)
            | set(self.passthrough_columns)
            | set(self.special_numeric_columns)
            | set(self.log_numeric_columns)
        )
        residual = [c for c in X.columns if c not in tagged]
        self._num_feats = X[residual].select_dtypes(include=["number"]).columns.tolist()

    # 3. Build ColumnTransformer → Pipeline
    def _build_pipeline(self) -> Pipeline:
        transformers = []

        if self._num_feats:
            for col in self._num_feats:
                if self.scale:
                    num_pipe = make_pipeline(
                        NamedFunctionTransformer(
                            func=lambda a: _ensure_2d(np.nan_to_num(a, nan=self.numeric_impute_value)),
                            feature_name=col
                        ),
                        NamedStandardScaler(feature_name=col)
                    )
                else:
                    num_pipe = make_pipeline(
                        NamedFunctionTransformer(
                            func=lambda a: _ensure_2d(np.nan_to_num(a, nan=self.numeric_impute_value)),
                            feature_name=col
                        )
                    )
                transformers.append((f"num_{col}", num_pipe, [col]))

        if self.special_numeric_columns:
            for col in self.special_numeric_columns:
                if self.scale:
                    special_pipe = make_pipeline(
                        NamedFunctionTransformer(
                            func=_parse_term,
                            feature_name=col
                        ),
                        NamedStandardScaler(feature_name=col)
                    )
                else:
                    special_pipe = make_pipeline(
                        NamedFunctionTransformer(
                            func=_parse_term,
                            feature_name=col
                        )
                    )
                transformers.append((f"special_num_{col}", special_pipe, [col]))

        if self.log_numeric_columns:
            for col in self.log_numeric_columns:
                if self.scale:
                    log_pipe = make_pipeline(
                        NamedFunctionTransformer(
                            func=_parse_log,
                            feature_name=col
                        ),
                        NamedStandardScaler(feature_name=col)
                    )
                else:
                    log_pipe = make_pipeline(
                        NamedFunctionTransformer(
                            func=_parse_log,
                            feature_name=col
                        )
                    )
                transformers.append((f"log_{col}", log_pipe, [col]))

        for col in self.percentage_columns:
            if self.scale:
                pct_pipe = make_pipeline(
                    NamedFunctionTransformer(
                        func=_parse_percent,
                        feature_name=col
                    ),
                    NamedFunctionTransformer(
                        func=lambda a: _ensure_2d(np.nan_to_num(a, nan=self.numeric_impute_value)),
                        feature_name=col
                    ),
                    NamedStandardScaler(feature_name=col)
                )
            else:
                pct_pipe = make_pipeline(
                    NamedFunctionTransformer(
                        func=_parse_percent,
                        feature_name=col
                    ),
                    NamedFunctionTransformer(
                        func=lambda a: _ensure_2d(np.nan_to_num(a, nan=self.numeric_impute_value)),
                        feature_name=col
                    )
                )
            transformers.append((f"pct_{col}", pct_pipe, [col]))

        for col in self.date_columns:
            if self.scale:
                date_pipe = make_pipeline(
                    NamedFunctionTransformer(
                        func=_parse_date,
                        feature_name=col
                    ),
                    NamedFunctionTransformer(
                        func=lambda a: _ensure_2d(np.nan_to_num(a, nan=self.numeric_impute_value)),
                        feature_name=col
                    ),
                    NamedStandardScaler(feature_name=col)
                )
                transformers.append((f"date_{col}", date_pipe, [col]))
            else:
                self.passthrough_columns.append(col)


        if self.ordinal_columns:
            for col, order in self.ordinal_columns.items():
                if self.scale:
                    ord_pipe = make_pipeline(
                        OrdinalEncoder(
                            categories=[order],    # Must be inside list! [order]
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan
                        ),
                        NamedFunctionTransformer(
                            func=lambda a: np.nan_to_num(a, nan=self.numeric_impute_value),
                            feature_name=col
                        ),
                        NamedStandardScaler(feature_name=col)
                    )
                else:
                    ord_pipe = make_pipeline(
                        OrdinalEncoder(
                            categories=[order],    # Must be inside list! [order]
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan
                        ),
                        NamedFunctionTransformer(
                            func=lambda a: np.nan_to_num(a, nan=self.numeric_impute_value),
                            feature_name=col
                        )
                    )
                transformers.append((f"ord_{col}", ord_pipe, [col]))


        if self.categorical_columns:
            cat_pipe = OneHotEncoder(handle_unknown="ignore", drop="first")
            transformers.append(("cat", cat_pipe, self.categorical_columns))

        if self.passthrough_columns:
            transformers.append(("pass", "passthrough", self.passthrough_columns))

        col_tf = ColumnTransformer(
            transformers=transformers,
            sparse_threshold=0.3,
            remainder="drop",
        )

        return Pipeline([("prep", col_tf)])


    def _check_unique_tags(self):
        """Every column must belong to at most ONE special tag list."""
        groups = {
            "date"              : self.date_columns,
            "percentage"        : self.percentage_columns,
            "ordinal"           : list(self.ordinal_columns),
            "categorical"       : self.categorical_columns,
            "passthrough"       : self.passthrough_columns,
            "special_numeric"   : self.special_numeric_columns,
            "log_numeric"       : self.log_numeric_columns,
        }
        seen = {}
        duplicates = []
        for tag, cols in groups.items():
            for c in cols:
                if c in seen:
                    duplicates.append(f"{c!r} appears in both {seen[c]} & {tag}")
                seen[c] = tag
        if duplicates:
            raise ValueError("Overlapping column tags:\n  " + "\n  ".join(duplicates))
    
    def _check_presence_of_tags(self):
        """Ensure every tagged column is in keep_columns."""
        def check_list(name, cols):
            missing = set(cols) - set(self.keep_columns)
            if missing:
                raise ValueError(f"{name} columns missing from keep_columns: {missing}")

        check_list("Date",              self.date_columns)
        check_list("Percentage",        self.percentage_columns)
        check_list("Ordinal",           list(self.ordinal_columns))
        check_list("Categorical",       self.categorical_columns)
        check_list("Passthrough",       self.passthrough_columns)
        check_list("Special numeric",   self.special_numeric_columns)
        check_list("Log numeric",       self.log_numeric_columns)
    
    def _clean_column_names(self, names: List[str]) -> List[str]:
        """Remove prefixes like 'num_', 'pct_', 'date_', 'ord_', 'cat_', 'pass_'."""
        clean_names = []
        for name in names:
            name = name.split('__')[-1]       # Remove double prefix
            name = name.replace(" ", "_")      # Replace spaces with underscores
            name = name.lower()                # Lowercase for consistency
            clean_names.append(name)
        return clean_names

# Easy access to preprocessors for kedro pipeline
def preprocess_node(
    df                      : pd.DataFrame,
    keep_columns            : List[str],
    numeric_feature_columns : List[str],
    return_df               : bool                  = True,
    scale                   : bool                  = False,
    *,
    date_columns            : Optional[List[str]]   = None,
    percentage_columns      : Optional[List[str]]   = None,
    ordinal_columns         : Optional[Dict[str, List[str]]] = None,
    categorical_columns     : Optional[List[str]]   = None,
    passthrough_columns     : Optional[List[str]]   = None,
    special_numeric_columns : Optional[List[str]]   = None,
    log_numeric_columns     : Optional[List[str]]   = None,
) -> pd.DataFrame:
    """Convenience function to create and run the preprocessor."""
    prep = LendingClubPreprocessor(
        numeric_feature_columns     =numeric_feature_columns,  # All keep_columns are numeric features
        scale                       =scale,
        keep_columns                =keep_columns,
        date_columns                =date_columns,
        percentage_columns          =percentage_columns,
        ordinal_columns             =ordinal_columns,
        categorical_columns         =categorical_columns,
        passthrough_columns         =passthrough_columns,
        special_numeric_columns     =special_numeric_columns,
        log_numeric_columns         =log_numeric_columns,
    )
    X_out, y =  prep.fit_transform(df, return_df=return_df)  # Return only features
    return X_out


# --------------------------------------------------------------------------- #
#                                 test                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "loan_amnt": [1000, 2200, np.nan],
            "int_rate": ["13.49%", "17.12%", "15.22%"],
            "grade": ["C", "F", "D"],
            "sub_grade": ["C2", "F5", "D1"],
            "term": [" 36 months", " 60 months", " 36 months"],
            "issue_d": ["01-12-2017", "Jun 2020", "Jun-2020"],
            "earliest_cr_line": ["2001-02-01", "Nov 1999", "Mar-1987"],
            "target": [1, 0, 1],
        }
    )

    KEEP = list(demo.columns)  # in real life, exclude 'target' from KEEP
    prep = LendingClubPreprocessor(
        keep_columns=KEEP,
        date_columns=["issue_d", "earliest_cr_line"],
        percentage_columns=["int_rate"],
        ordinal_columns={
            "grade": list("ABCDEFG"),
            "sub_grade": [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)],
        },
        categorical_columns=["term"],
        passthrough_columns=["id"],
    )

    X_, y_ = prep.fit_transform(demo, target="target", return_df=True)
    print("Features shape:", X_.shape)           # verify pipeline runs
    print("Features:\n", X_)
    print("Target:\n", y_)
