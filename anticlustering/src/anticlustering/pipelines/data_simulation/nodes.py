# pipeli​nes/data_simulation/nodes.py
import numpy as np
import pandas as pd
from typing import List, Dict


def simulate_matrices(
    n_values: List[int],
    n_features: int,
    rng_seed: int,
) -> Dict[str, pd.DataFrame]:
    """
    Generate one N×F matrix per N and return **one dict**.

    Returns
    -------
    data : dict
        Keys are strings "N_<value>", e.g. "N_10"; values are DataFrames.
    """
    rng = np.random.default_rng(rng_seed)
    data = {}

    for N in n_values:
        arr = rng.standard_normal(size=(N, n_features))
        df = pd.DataFrame(arr, columns=[f"x{j}" for j in range(n_features)])
        data[f"N_{N}"] = df

    return data
