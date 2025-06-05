"""
This is a boilerplate pipeline 'kaggle_data'
generated using Kedro 0.19.13
"""
import kagglehub
import os
import pandas as pd
import logging
from typing import List, Optional
import datetime as _dt

from ...preprocessing.online_data import preprocess_node
from ...lending_club.core.loan import LoanRecord
from ...lending_club.core.simulator import LoanSimulator

_LOG = logging.getLogger(__name__)

def load_kaggle_data(name: str):
    kagglehub.login()
    # Download latest version
    path = kagglehub.dataset_download("beatafaron/loan-credit-risk-and-population-stability")

    # Load CSV
    csv_file = os.path.join(path, name)
    df = pd.read_csv(csv_file)
    return df


def create_test_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a test version of the Kaggle data for quick checks."""
    # Example: take the first 100 rows for testing
    return df.head(100).copy()


def _reduce_sample(
        df          : pd.DataFrame, 
        n           : int = 100, 
        rng_number  : int = 42
    ) -> pd.DataFrame:
    """Reduce the DataFrame to a sample of n rows."""
    if len(df) > n:
        return df.sample(n=n, random_state=rng_number).reset_index(drop=True)
    return df.copy()

def process_kaggle_data(
        df              : pd.DataFrame, 
        reduce_n        : int | None = None,
        scale           : bool = False,
        rng_number      : int = 42
    ) -> pd.DataFrame:
    """Process the Kaggle data to ensure it is ready for analysis."""
    _LOG.info("Processing with scale=%s and reduce_n=%s", scale, reduce_n)

    if reduce_n == "None":
        reduce_n = None
    
    if reduce_n is not None:
        prev_len = len(df)
        df = _reduce_sample(df, n=int(reduce_n), rng_number=rng_number)
        _LOG.info("Reduced Kaggle data to %d rows from %s rows (random sampling)", len(df), prev_len)
    
    
    # keep_columns = [
    #     'id',
    #     'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
    #     'installment', 'grade', 'sub_grade', 'home_ownership', 'annual_inc',
    #     'verification_status', 'issue_d', 'dti', 'delinq_2yrs', 'earliest_cr_line',
    #     'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec',
    #     'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'total_pymnt',
    #     'total_rec_prncp', 'total_rec_int', 'last_pymnt_d', 'last_credit_pull_d', "loan_status",
    #     'recoveries'
    # ]
    # percentage_columns = ['int_rate', 'revol_util']
    # date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
    # ordinal_columns = {
    #     'grade': list('ABCDEFG'),  # A < B < C < ... < G
    #     'sub_grade': [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]
    # }
    # categorical_columns = ['home_ownership', 'verification_status']
    # special_numeric_columns = ['term']
    # log_numeric_columns = [
    #     'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment',
    #     'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    #     'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
    #     'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_prncp',
    #     'total_rec_int', 'recoveries'
    # ]
    # passthrough_columns = ['id', 'loan_status']

    # Current overwrite of values
    #FIXME: Could be made yaml-configurable

    keep_columns            = [
        'id', 'loan_amnt', 'term', 'int_rate', 'issue_d', 'grade', 'sub_grade', 'last_pymnt_d',
        'loan_status', 'total_rec_prncp', 'total_rec_int', 'recoveries', 'annual_inc'
    ]
    percentage_columns      = ['int_rate']
    date_columns            = ['issue_d', 'last_pymnt_d']
    ordinal_columns         = {
        'grade': list('ABCDEFG'),  # A < B < C < ... < G
        'sub_grade': [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]
    }
    categorical_columns     = []
    special_numeric_columns = ['term']  # 'term' is categorical but represented as numeric
    log_numeric_columns     = [
        'loan_amnt', 'total_rec_prncp', 'total_rec_int', 'recoveries', 'annual_inc'
    ]
    passthrough_columns     = ['id', 'loan_status']  # 'loan_status' is the target variable
    numeric_feature_columns = set(keep_columns.copy()) \
        - set(passthrough_columns) - set(date_columns) - set(percentage_columns) \
        - set(ordinal_columns.keys()) - set(categorical_columns) - set(special_numeric_columns)
    numeric_feature_columns = list(numeric_feature_columns)
    
    return preprocess_node(
        df,
        keep_columns            =keep_columns,
        numeric_feature_columns =numeric_feature_columns,
        scale                   =scale,
        return_df               =True,
        percentage_columns      =percentage_columns,
        date_columns            =date_columns,
        ordinal_columns         =ordinal_columns,
        categorical_columns     =categorical_columns,
        special_numeric_columns =special_numeric_columns,
        log_numeric_columns     =log_numeric_columns,
        passthrough_columns     =passthrough_columns
    )
    

def kaggle_df_to_loan_records(df: pd.DataFrame) -> list[LoanRecord]:
    """
    Convert each row of the (possibly scaled) Lending-Club Kaggle DataFrame
    into a `LoanRecord` instance using `LoanRecord.from_dict`.
    Rows that raise parsing errors are skipped (log a warning).
    """
    _LOG.info("kaggle df:\n %s", df.head(5))
    skipped_records = {}

    loan_records = []
    for row in df.to_dict(orient="records"):
        try:
            loan_records.append(LoanRecord.from_dict(row))
        except Exception as exc:
            skipped_records[row.get("id", "unknown")] = str(exc)
            #_LOG.warning("Row %s skipped: %s", row.get("id"), exc)
    
    if skipped_records:
        _LOG.warning("Skipped %d records due to parsing errors: %s", len(skipped_records), skipped_records)

    _LOG.info("Converted %d rows to LoanRecord instances", len(loan_records))
    
    _LOG.info("LoanRecord sample: %s", loan_records[0] if loan_records else "No records")
    return loan_records


def loan_records_to_long_df(
    loans                       : List[LoanRecord],
    as_of_str                   : Optional[str] = None,
    assume_regular_prepayment   : bool = True,
) -> pd.DataFrame:
    """
    Wrapper around ``LoanSimulator.batch_generate`` that also handles the
    optional *cut-off* date (`as_of`) supplied via YAML parameters.

    Parameters
    ----------
    loans
        Parsed LoanRecord objects from the ingest stage.
    as_of_str
        Optional ISO date string (``YYYY-MM-DD``).  If given, histories are
        truncated at *that* month (inclusive), which is handy when replaying
        year-by-year rather than the full horizon.
    assume_regular_prepayment
        Passed straight through to :py:meth:`LoanSimulator.batch_generate`.

    Returns
    -------
    pd.DataFrame
        Concatenated histories (long format).
    """
    as_of: _dt.date | None = (
        _dt.date.fromisoformat(as_of_str) if as_of_str else None
    )

    _LOG.info(
        "Generating synthetic histories for %d loans … (as_of=%s)",
        len(loans),
        as_of,
    )
    df = LoanSimulator.batch_generate(
        loans, 
        as_of                       =as_of, 
        assume_regular_prepayment   =assume_regular_prepayment
    )
    _LOG.info("→ Produced %d loan-month rows", len(df))
    _LOG.info("Sample of the generated DataFrame:\n%s", df.head(30))
    return df


