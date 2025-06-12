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

import numpy as np
import pandas as pd
from math import ceil
from typing import List, Dict

def generate_simulation_study_data(
    k_values            : List[int],
    n_runs              : int,
    n_range_min         : int,
    n_range_max         : int,
    dstr_unif           : Dict,
    dstr_normal_std     : Dict,
    dstr_normal_wide    : Dict,
    rng_seed            : int,
) -> pd.DataFrame:
    """
    Monte Carlo data for Table 2 with N always divisible by K:
      • For each run_id=1…n_runs
      • For each K in k_values
      • For each distribution in {uniform, normal_std, normal_wide}
        – Compute M_min = ceil(n_range_min / K)
               M_max = floor(n_range_max / K)
        – Draw M ~ UniformInt[M_min, M_max]
        – Set N = M * K
        – Sample an (N × K) matrix
    Returns a DataFrame with columns:
      ['run_id', 'K', 'distribution', 'N', 'stimuli']
    where stimuli is a NumPy array of shape (N, K).
    """
    rng = np.random.default_rng(rng_seed)

    # map distribution names to sampling functions
    dist_map = {
        "uniform":    lambda size: rng.uniform(**dstr_unif, size=size),
        "normal_std": lambda size: rng.normal(**dstr_normal_std, size=size),
        "normal_wide":lambda size: rng.normal(**dstr_normal_wide, size=size),
    }
    dist_names = list(dist_map.keys())

    # total number of parameter‐sets = n_runs × len(k_values)
    total = n_runs * len(k_values)
    run_ids = np.repeat(np.arange(1, n_runs + 1), len(k_values))
    ks      = np.tile(k_values, n_runs)
    dist_choices = rng.choice(dist_names, size=total)

    records = []
    for run_id, K_, dist in zip(run_ids, ks, dist_choices):
        # compute range of multiples that fit in [n_range_min, n_range_max]
        M_min = ceil(n_range_min / K_)
        M_max = n_range_max // K_
        if M_min > M_max:
            raise ValueError(
                f"Range [{n_range_min},{n_range_max}] too small to get a multiple of {K_}"
            )
        M = rng.integers(M_min, M_max + 1)
        N = M * K_

        X = dist_map[dist]((N, K_))
        records.append({
            "run_id":      run_id,
            "K":           K_,
            "distribution":dist,
            "N":           N,
            "stimuli":     X,
        })

    df = pd.DataFrame(records)
    print(f"Generated simulation study data with {len(df)} records.")
    print(f"Example of data:\n{df.head(10)}")
    return df
