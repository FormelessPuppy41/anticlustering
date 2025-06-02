"""
This is a boilerplate pipeline 'kaggle_data'
generated using Kedro 0.19.13
"""
import kagglehub
import os
import pandas as pd

from ...preprocessing.onlinedata import preprocess_node


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
    

def process_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the Kaggle data to ensure it is ready for analysis."""
    # Example processing: drop rows with missing values
    print("Processing Kaggle data...")
    print(f"Initial shape: {df.shape}")
    print(f"Description:\n{df.describe(include='all')}")
    print(f"Data types:\n{df.dtypes}")
    
    return preprocess_node(
        df,
    )


