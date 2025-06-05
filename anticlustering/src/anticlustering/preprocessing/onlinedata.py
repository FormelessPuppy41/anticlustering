"""
lending_club_preprocessing.py
--------------------------------
Reusable, modular preprocessing pipeline for Lending Club loan-data panels
(2014 – 2020, ≈ 142 columns).  The class follows a scikit‑learn‑compatible
API (``fit`` / ``transform`` / ``fit_transform``) so you can slot it into any
workflow or Kedro node.

Key design goals
================
* **Modular:** each logical cleaning step is an isolated private method
  (``_process_percent_cols``, ``_process_term_and_emp_length`` …) so you can
  override or drop steps with minimal surgery.
* **Configurable:** constructor flags let you keep/drop identifiers, text
  fields, choose encoding schemes, etc.
* **Lightweight:** pure ``pandas`` implementation – no heavyweight deps except
  ``numpy`` & ``pandas``.
* **Testable:** deterministic behaviour; every side‑effect on a column is
  documented & unit‑testable.

Usage example
-------------
```python
from lending_club_preprocessing import LendingClubPreprocessor
import pandas as pd

raw = pd.read_csv("loan_2014_18.csv")
pre = LendingClubPreprocessor()
clean = pre.fit_transform(raw)
```

If you work with **Kedro**, drop the helper ``preprocess_node`` function into
``nodes.py`` and register it in your pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd



# -------------------------------------------------------------------------
# 1.  DEFAULT COLUMN GROUPS  (override in the constructor if your schema
#     differs – useful when Lending Club adds/drops columns over the years)
# -------------------------------------------------------------------------

PERCENT_COLS: Sequence[str] = [
    "int_rate", "revol_util",
]

FICO_RANGE_PAIRS: Sequence[Tuple[str, str]] = [
    ("fico_range_low", "fico_range_high"),
    ("avg_fico_low", "avg_fico_high"),  # appears in 2019‑20 vintages
]

DATE_COLS: Sequence[str] = [
    "issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d",
    "next_pymnt_d", "hardship_start_date", "debt_settlement_flag_date", 
    "sec_app_earliest_cr_line", "hardship_end_date", "payment_plan_start_date"
]

ORDINAL_COLS: Sequence[str] = ["grade", "sub_grade"]

CATEGORICAL_COLS: Sequence[str] = [
    "home_ownership", "verification_status", "purpose", "addr_state",
    "initial_list_status", "application_type", "disbursement_method",
    "pymnt_plan", "loan_status", "hardship_flag", "debt_settlement_flag",
]

TEXT_COLS: Sequence[str] = [
    "title", "emp_title", "desc", "hardship_reason",
    "hardship_loan_status",  # etc.
]

IDENTIFIER_COLS: Sequence[str] = [
    "id", "member_id", "url", "policy_code", "acc_now_delinq",
]

EXCLUDED_SCALING_COLS: Sequence[str] = [
    "id", "issue_d"
]

ZIP_COL: str = "zip_code"

TERM_COL: str = "term"
EMP_LENGTH_COL: str = "emp_length"


# -------------------------------------------------------------------------
# 2.  COLUMNS WE ACTUALLY WANT TO ANALYSE
# -------------------------------------------------------------------------
KEEP_COLS: Sequence[str] = [
    # identifiers
    "id",

    # loan terms / pricing
    "loan_amnt", "funded_amnt", "funded_amnt_inv",
    "term", "int_rate", "installment",

    # internal ratings
    "grade", "sub_grade",

    # borrower capacity & demographic proxies
    "emp_length", "home_ownership", "annual_inc",
    "verification_status", #"purpose", "addr_state",

    # credit history snapshot
    "fico_range_low", "fico_range_high",
    "dti", "revol_bal", "revol_util",
    "open_acc", "total_acc", "delinq_2yrs",
    "inq_last_6mths", "pub_rec",

    # temporal anchors
    "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d",

    # derived & target vars
    "credit_age_months",     # created later
    "loan_status", "total_pymnt", "total_rec_prncp",
    "total_rec_int", "last_pymnt_amnt",

    # balance-sheet context
    "tot_cur_bal", "tot_coll_amt", "total_rev_hi_lim",
]


# The bulk numeric columns (loan_amnt, funded_amnt, dti, …) are *not* enumerated
# here – we detect them dynamically by dtype.

# --------------------------------------------------
# Extend / replace in __init__
# --------------------------------------------------
ordinal_maps: Dict[str, Dict[str, int]] = {
    # VERIFICATION
    #    0 = no proof, 1 = proof supplied (any flavour)
    "verification_status": {
        "not verified": 0,
        "none": 0,
        "unverified": 0,
        "verified": 1,
        "source verified": 1,
        "income verified": 1,
    },

    # HOME OWNERSHIP -- proxy for asset backing
    #    0 = unknown / other, 1 = rent, 2 = mortgage, 3 = own outright
    "home_ownership": {
        "any": 0,
        "none": 0,
        "other": 0,
        "rent": 1,
        "mortgage": 2,
        "own": 3,
    },

    # LOAN STATUS -- ascending “health” scale
    #    0  worst ⇒ 5  best
    "loan_status": {
        "charged off": 0,
        "default": 0,
        "does not meet the credit policy. status: charged off": 0,
        "late (31-120 days)": 1,
        "late (16-30 days)": 2,
        "in grace period": 2,
        "current": 3,
        "does not meet the credit policy. status: fully paid": 3,
        "fully paid": 4,
        "issued": 5,
    },

    # APPLICATION TYPE
    "application_type": {
        "individual": 0,
        "joint app": 1,
        "direct pay": 2,        # rare historical label
    },

    # HARDSHIP *progress* FLAGS – simple binary / ternary scales
    "hardship_loan_status": {     # status of the *loan* while in hardship
        "none": 0,
        "ongoing": 1,
        "paid": 2,
        "paid off": 2,
        "finished": 2,
    },
    "hardship_status": {          # status of the *hardship plan*
        "ongoing": 0,
        "completed": 1,
        "completed – paid": 1,
    },

    # ------------------------------------------------------------------
    # The remaining fields are **nominal** – we collapse spelling variants
    # to a canonical key but do *not* imply an ordered scale.
    # Using integers keeps the API consistent with _encode_ordinals().
    # ------------------------------------------------------------------
    "purpose": {
        # Top-level motives; extend as needed
        "debt_consolidation": 0,
        "credit_card": 1,
        "home_improvement": 2,
        "major_purchase": 3,
        "medical": 4,
        "car": 5,
        "small_business": 6,
        "moving": 7,
        "vacation": 8,
        "house": 9,
        "renewable_energy": 10,
        "wedding": 11,
        "other": 12,
    },

    "title": {
        # Collapsed free-text buckets (add more if you pre-clean titles)
        "debt consolidation": 0,
        "credit card refinancing": 1,
        "home improvement": 2,
        "small business loan": 3,
        "other": 4,
    },

    "hardship_type": {
        "interest only": 0,
        "payment plan": 1,
        "deferment": 2,
        "partial payment": 3,
        "other": 4,
    },

    "hardship_reason": {
        "illness": 0,
        "job loss": 1,
        "income reduction": 2,
        "natural disaster": 3,
        "other": 4,
    },
}



redundant_cols: List[str] = [
    # These are redundant with other columns, so we drop them.
    "emp_title",
    "url",
    "pymnt_plan", # all n
    "purpose",
    "title",
    "zip_code",  # no numeric value
    "addr_state",  # no numeric value
    "hardship_flag",  # no numeric value
    "hardship_type",  # no numeric value
    "hardship_reason",  # no numeric value
    "hardship_status",  # no numeric value
    "hardship_loan_status",  # no numeric value
    "Unnamed: 0",  # index column, if present

    # Columns dropped for ease:
    ""
]

# -------------------------------------------------------------------------
@dataclass
class LendingClubPreprocessor:
    """End‑to‑end cleaner for Lending Club loan data."""

    # ------------------------------------------------------------------
    # Config flags
    # ------------------------------------------------------------------
    keep_identifier_cols: bool = False
    keep_fico_range_cols: bool = False
    drop_text_fields: bool = True
    one_hot_encode: bool = True
    impute_numeric: bool = True
    windsorise_numeric: bool = False  # clip numeric columns at 1st/99th pct
    scale_date_cols: bool = True   # set True if you want ordinal-scaled dates


    # Scaling
    scale_numeric: bool = True  # scale numeric columns to zero mean, unit variance
    scaler_kind: str = "standard"  # "standard" | "minmax" | "robust"
    rare_threshold: float = 0.005  # threshold for rare categories in one-hot encoding
    
    # You can override default column groups if your file differs.
    ordinal_maps: Dict[str, Dict[str, int]] = field(default_factory=lambda: ordinal_maps)
    redundant_cols: Sequence[str] = field(default_factory=lambda: redundant_cols)
    percent_cols: Sequence[str] = field(default_factory=lambda: list(PERCENT_COLS))
    fico_pairs: Sequence[Tuple[str, str]] = field(default_factory=lambda: list(FICO_RANGE_PAIRS))
    date_cols: Sequence[str] = field(default_factory=lambda: list(DATE_COLS))
    ordinal_cols: Sequence[str] = field(default_factory=lambda: list(ORDINAL_COLS))
    categorical_cols: Sequence[str] = field(default_factory=lambda: list(CATEGORICAL_COLS))
    text_cols: Sequence[str] = field(default_factory=lambda: list(TEXT_COLS))
    exclude_scaling_cols: Sequence[str] = field(default_factory=lambda: list(EXCLUDED_SCALING_COLS))

    zip_col: str = ZIP_COL
    term_col: str = TERM_COL
    emp_length_col: str = EMP_LENGTH_COL

    # ------------------------------------------------------------------
    # Runtime artefacts (learned during `fit`)
    # ------------------------------------------------------------------
    _numeric_medians: Dict[str, float] = field(init=False, default_factory=dict)
    _dummy_lookup: Dict[str, List[str]] = field(init=False, default_factory=dict)
    _num_cols_: List[str] = field(init=False, default_factory=list)
    _scaler_: object | None = field(init=False, default=None)


    def __post_init__(self):
        self.categorical_cols = [
            c for c in self.categorical_cols if c not in self.ordinal_maps
        ]
        
        # ------------------------------------------------------------------
        self.keep_cols = set(KEEP_COLS)
        

    # ==================================================================
    # sklearn‑compatible public API
    # ==================================================================

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None):
        """Learn dataset‑specific parameters (e.g., numeric medians)."""

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.impute_numeric:
            self._numeric_medians = df[numeric_cols].median(numeric_only=True).to_dict()
        
        if self.one_hot_encode:
            self._build_dummy_lookup(df)
        
        # ------------------------------------------------------------------
        extra = [c for c in df.columns if c not in self.keep_cols]
        self.redundant_cols = extra.copy()           # start with everything un-kept

        # date columns NOT kept => their *_ord copy is also redundant
        for col in df.columns:
            if col not in self.keep_cols:
                self.redundant_cols.append(f"{col}_ord")

        if self.scale_numeric:
            clean = (
                df.pipe(self._process_percent_cols)
                .pipe(self._process_term_and_emp_length)
                .pipe(self._process_numeric)
                .pipe(self._process_fico_ranges)
                .pipe(self._process_dates)
                .pipe(self._dates_to_ordinal)
                .pipe(self._process_ordinal)
                .pipe(self._encode_semantic_ordinals)
                .pipe(lambda x: x)  # no cat/dummy yet
            )
            print('clean\n', clean.head(10))  # debug output
            self._num_cols_ = [
                c for c in clean.select_dtypes(include=[np.number]).columns
                if not set(clean[c].dropna().unique()).issubset({0, 1})
            ]
            # drop any columns the user wants excluded
            self._num_cols_ = [
                c for c in self._num_cols_ if c not in self.exclude_scaling_cols
            ]

            from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
            scaler_cls = {
                "standard": StandardScaler,
                "robust": RobustScaler,
                "minmax": MinMaxScaler,
            }[self.scaler_kind]
            self._scaler_ = scaler_cls().fit(clean[self._num_cols_])
            print('df \n', df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleaning pipeline; *does not* mutate the input frame."""
        out = df.copy()

        if not self.keep_identifier_cols:
            out = out.drop(columns=[c for c in IDENTIFIER_COLS if c in out.columns], errors="ignore")

        out = self._process_percent_cols(out)
        out = self._process_term_and_emp_length(out)
        out = self._process_numeric(out)
        out = self._process_fico_ranges(out)
        out = self._process_dates(out)
        out = self._dates_to_ordinal(out)  # add ordinal date columns if needed
        out = self._process_ordinal(out)
        out = self._encode_semantic_ordinals(out)
        out = self._process_categoricals(out)
        out = self._process_zip(out)
        
        out = self._derive_features(out)
        out = self._impute(out)
        out = self._scale_numeric(out)  # apply fitted scaler to numeric cols
        out = self._process_text(out)
        out = self._drop_redundant_cols(out)
        out = self._final_cast(out)

        return out

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Convenience one‑liner combining ``fit`` and ``transform``."""
        return self.fit(df, y).transform(df)

    # ==================================================================
    #  Private helpers  (each returns a *new* frame for chainability)
    # ==================================================================

    # ----- 1. percent strings ------------------------------------------------
    def _process_percent_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.percent_cols:
            if col in out.columns:
                out[col] = (
                    out[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .replace({"nan": np.nan, "": np.nan})
                    .astype(float)
                    / 100.0
                )
        return out

    # ----- 2. term + emp_length --------------------------------------------
    def _process_term_and_emp_length(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # term – extract integer months
        if self.term_col in out.columns:
            out[self.term_col] = (
                out[self.term_col]
                .astype(str)
                .str.extract(r"(\d+)")[0]
                .astype(float)
            )
        # emp_length – map text to years
        if self.emp_length_col in out.columns:
            mapping = {
                "< 1 year": 0.5,
                "1 year": 1,
                "2 years": 2,
                "3 years": 3,
                "4 years": 4,
                "5 years": 5,
                "6 years": 6,
                "7 years": 7,
                "8 years": 8,
                "9 years": 9,
                "10+ years": 10,
                "n/a": np.nan,
            }
            out[self.emp_length_col] = (
                out[self.emp_length_col]
                .map(mapping)
                .astype(float)
            )
        return out

    # ----- 3. numeric cleaning ---------------------------------------------
    def _process_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.windsorise_numeric:
            return df.copy()
        out = df.copy()
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        # Winsorise at 1st/99th pct – crude but effective
        for col in numeric_cols:
            q1, q99 = out[col].quantile([0.01, 0.99])
            out[col] = out[col].clip(lower=q1, upper=q99)
        return out

    # ----- 4. fico ranges ---------------------------------------------------
    def _process_fico_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for low, high in self.fico_pairs:
            if low in out.columns and high in out.columns:
                out[f"{low[:-4]}avg"] = out[[low, high]].mean(axis=1)
                if not self.keep_fico_range_cols:
                    out = out.drop(columns=[low, high])
        return out

    # ----- 5. dates ---------------------------------------------------------
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date‑like strings and derive *credit_age_months*.

        * Canonical Lending Club month‑year strings (e.g. "Feb‑2007") are parsed
          with ``format="%b-%Y"`` → first day of month.
        * Any values that fail this pattern fall back to pandas' flexible parser
          (``infer_datetime_format=True``).
        * Adds ``credit_age_months`` when both *issue_d* and *earliest_cr_line*
          are present.
        """
        out = df.copy()
        for col in self.date_cols:
            if col not in out.columns:
                continue
            s = (
                out[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
            )
            primary = pd.to_datetime(s, format="%b-%Y", errors="coerce")
            mask = primary.isna() & s.notna()
            if mask.any():
                primary.loc[mask] = pd.to_datetime(
                    s.loc[mask], errors="coerce", infer_datetime_format=True
                )
            out[col] = primary

        # Derived credit age in *months*
        if {"issue_d", "earliest_cr_line"}.issubset(out.columns):
            delta = out["issue_d"] - out["earliest_cr_line"]
            out["credit_age_months"] = delta.dt.days / 30.44  # approximate month
        return out
    
    def _dates_to_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If scale_date_cols=True, add <col>_ord numeric columns
        (days since 1970-01-01) so they take part in scaling.
        """
        if not self.scale_date_cols:
            return df

        out = df.copy()
        for col in self.date_cols:
            if col in out.columns and col in self.keep_cols:
                # convert nan-safe to integer nanoseconds → days
                out[col] = pd.to_datetime(out[col], errors="coerce").view("int64") // 86_400_000_000_000
                #out[f"{col}_ord"] = out[col].view("int64") // 86_400_000_000_000
        return out

    # ----- 6. ordinal grades ------------------------------------------------
    def _process_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        grade_map = {g: i for i, g in enumerate("ABCDEFG", start=1)}
        if "grade" in out.columns:
            out["grade"] = out["grade"].map(grade_map).astype(float)
        if "sub_grade" in out.columns:
            sub_map = {f"{g}{n}": 5 * (i - 1) + n for i, g in enumerate("ABCDEFG", start=1) for n in range(1, 6)}
            out["sub_grade"] = out["sub_grade"].map(sub_map).astype(float)
        return out
    
    def _encode_semantic_ordinals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, mapping in self.ordinal_maps.items():
            if col not in out.columns or col not in self.keep_cols:
                continue
            out[col] = (
                out[col].astype(str).str.strip().str.lower()
                .map(mapping)
                .astype("Int8")
            )
        return out


    # ----- 7. categoricals --------------------------------------------------
    def _process_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cats = [c for c in self.categorical_cols if c in out.columns and c in self.keep_cols]

        # map strings -> lower/strip
        out[cats] = out[cats].apply(lambda col: col.str.lower().str.strip())

        if self.one_hot_encode and cats:
            # training-aware re-cast
            for col, cat_list in self._dummy_lookup.items():
                if col not in out.columns:
                    continue
                out[col] = np.where(out[col].isin(cat_list), out[col], "other")
                out[col] = pd.Categorical(out[col], categories=cat_list)

            dummies = pd.get_dummies(
                out[self._dummy_lookup.keys()], prefix_sep="=", dtype="uint8"
            )
            dummies = dummies.reindex(
                columns=self._get_dummy_columns(), fill_value=0
            )
            out = pd.concat([out.drop(columns=self._dummy_lookup.keys()), dummies], axis=1)
        return out

    def _get_dummy_columns(self) -> List[str]:
        cols: List[str] = []
        for col, cats in self._dummy_lookup.items():
            cols.extend([f"{col}={c}" for c in cats])
        return cols


    # ----- 8. ZIP processing ------------------------------------------------
    def _process_zip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep first three digits of the masked ZIP, then one‑hot or category."""
        out = df.copy()
        if self.zip_col in out.columns:
            # Extract first 3 digits; Lending Club masks last two with xx.
            out[self.zip_col] = (
                out[self.zip_col]
                .astype(str)
                .str.slice(0, 3)
                .replace({"nan": np.nan, "": np.nan})
            )
            if self.one_hot_encode:
                dummies = pd.get_dummies(out[self.zip_col], prefix="zip3", dummy_na=True, dtype="uint8")
                out = pd.concat([out.drop(columns=[self.zip_col]), dummies], axis=1)
            else:
                out[self.zip_col] = out[self.zip_col].astype("category")
        return out

    # ----- 9. text fields ---------------------------------------------------
    def _process_text(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.drop_text_fields:
            out = out.drop(columns=[c for c in self.text_cols if c in out.columns], errors="ignore")
        # else keep raw – suitable for later NLP feature extraction.
        return out

    # ----- 10. custom derived examples -------------------------------------
    def _derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"loan_amnt", "annual_inc"}.issubset(out.columns):
            out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace({0: np.nan})
        return out

    # ----- 11. numeric imputation ------------------------------------------
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.impute_numeric:
            return df
        out = df.copy()
        for col, median in self._numeric_medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(median)
        return out

    # ----- 12. final dtypes -------------------------------------------------
    def _final_cast(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Cast booleans to uint8 for memory savings
        bool_cols = out.select_dtypes(include=["bool"]).columns
        out[bool_cols] = out[bool_cols].astype("uint8")
        # Downcast small floats/ints where safe
        float_cols = out.select_dtypes(include=["float"]).columns
        out[float_cols] = out[float_cols].apply(pd.to_numeric, downcast="float")
        int_cols = out.select_dtypes(include=["int"]).columns
        out[int_cols] = out[int_cols].apply(pd.to_numeric, downcast="integer")
        return out

    # ----- 13. drop redundant columns --------------------------------------
    def _drop_redundant_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are redundant or not needed for analysis."""
        out = df.copy()
        cols_to_drop = [c for c in self.redundant_cols if c in out.columns]
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop, errors="ignore")
        return out
    
    # --------------------------------------------------
    def _build_dummy_lookup(self, df: pd.DataFrame) -> None:
        for col in self.categorical_cols:
            if col not in df.columns or col not in self.keep_cols:
                continue
            freq = (
                df[col].astype(str).str.strip().str.lower().value_counts(normalize=True)
            )
            major = freq[freq >= self.rare_threshold].index.tolist()
            cats = sorted(major) + ["other"]          # deterministic order
            self._dummy_lookup[col] = cats

    # ----- 11-bis. numeric scaling ------------------------------------------
    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_numeric or self._scaler_ is None:
            return df
        out = df.copy()

        # keep only columns that still exist
        cols = [c for c in self._num_cols_ if c in out.columns]
        if cols:
            out[cols] = self._scaler_.transform(out[cols])
        return out



    
# ==================================================================
# Kedro node wrapper
# ==================================================================
def preprocess_node(
        df: pd.DataFrame,
        *,
        keep_identifier_cols: bool = False,
        keep_fico_range_cols: bool = False,
        drop_text_fields: bool = True,
        one_hot_encode: bool = True,
        impute_numeric: bool = True,
        windsorise_numeric: bool = False,
        scale_numeric: bool = True,
        scaler_kind: str = "standard",
        percent_cols: Sequence[str] = PERCENT_COLS,
        fico_pairs: Sequence[Tuple[str, str]] = FICO_RANGE_PAIRS,
        date_cols: Sequence[str] = DATE_COLS,
        ordinal_cols: Sequence[str] = ORDINAL_COLS,
        categorical_cols: Sequence[str] = CATEGORICAL_COLS,
        text_cols: Sequence[str] = TEXT_COLS,
        zip_col: str = ZIP_COL,
        term_col: str = TERM_COL,
        emp_length_col: str = EMP_LENGTH_COL,
    ) -> pd.DataFrame:  # pragma: no cover
    """Kedro-friendly wrapper – just drops the cleaned frame in the output catalog."""
    return LendingClubPreprocessor(
        keep_identifier_cols=keep_identifier_cols,
        keep_fico_range_cols=keep_fico_range_cols,
        drop_text_fields=drop_text_fields,
        one_hot_encode=one_hot_encode,
        impute_numeric=impute_numeric,
        windsorise_numeric=windsorise_numeric,
        scale_numeric=scale_numeric,
        scaler_kind=scaler_kind,
        percent_cols=percent_cols,
        fico_pairs=fico_pairs,
        date_cols=date_cols,
        ordinal_cols=ordinal_cols,
        categorical_cols=categorical_cols,
        text_cols=text_cols,
        zip_col=zip_col,
        term_col=term_col,
        emp_length_col=emp_length_col
    ).fit_transform(df)
