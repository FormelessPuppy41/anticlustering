"""
This is a boilerplate pipeline 'data_simulation'
generated using Kedro 0.19.12
"""
# src/<your_project>/pipelines/data_simulation/nodes.py

import numpy as np
import pandas as pd

def simulate_dataset(
        n: int, 
        m: int, 
        dist: str, 
        seed: int | None = None
    ) -> pd.DataFrame:
    """
    Simulate a dataset with n samples and m features.
    The dataset can be generated from a normal or uniform distribution.
    The features are named feature_1, feature_2, ..., feature_m.
    The random seed can be set for reproducibility.
    
    Args:
        n (int): number of samples_
        m (int): number of features_
        dist (str): distribution type, either "normal" or "uniform"
        seed (int | None, optional): random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: if the distribution is not supported

    Returns:
        pd.DataFrame: simulated dataset with n samples and m features
    """
    rng = np.random.default_rng(seed)
    if dist == "normal":
        data = rng.normal(0, 1, size=(n, m))
    elif dist == "uniform":
        data = rng.uniform(0, 1, size=(n, m))
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    df = pd.DataFrame(data, columns=[f"feature_{i+1}" for i in range(m)])
    return df

def generate_all_datasets(
        n_values: list[int], 
        num_features: int, 
        distribution: str, 
        seed: int = 42
    ) -> dict:
    """
    Generate multiple datasets with different sample sizes.
    Each dataset is generated using the same number of features and distribution type.
    The datasets are stored in a dictionary with keys "data_1", "data_2", ..., "data_n".
    The random seed is incremented for each dataset to ensure different data.

    Args:
        n_values (list[int]): list of sample sizes for each dataset
        num_features (int): number of features for each dataset
        distribution (str): distribution type, either "normal" or "uniform"
        seed (int, optional): random seed for reproducibility. Defaults to 42.

    Returns:
        dict: dictionary of simulated datasets with keys "data_1", "data_2", ..., "data_n" where n is the sample size
    """
    datasets = {}
    for i, n in enumerate(n_values):
        df = simulate_dataset(n, num_features, distribution, seed + i)
        datasets[f"data_{n}"] = df
    
    #print(f"Generated {len(datasets)} datasets with sample sizes: {n_values}")
    #print(f"Generated datasets: {list(datasets.keys())}")
    #print(f'Generated datasets: {datasets}')
    
    return datasets
    
def combine_datasets(datasets: dict) -> pd.DataFrame:
    """
    Combine multiple datasets into a single DataFrame.
    The datasets are concatenated along the rows.

    Args:
        datasets (dict): dictionary of datasets to combine
    
    Raises:
        ValueError: if no datasets are provided or if the dictionary is empty

    Returns:
        pd.DataFrame: combined dataset
    """
    if datasets is None or len(datasets) == 0:
        raise ValueError("No datasets to combine.")
    combined = []
    for name, df in datasets.items():
        df = df.copy()
        df["dataset"] = name
        combined.append(df)
    
    return pd.concat(combined, ignore_index=True)