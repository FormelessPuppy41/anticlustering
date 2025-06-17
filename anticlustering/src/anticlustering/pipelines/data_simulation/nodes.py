# pipeli​nes/data_simulation/nodes.py
import numpy as np
import pandas as pd
from typing import List, Dict
import logging

_LOG = logging.getLogger(__name__)

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
    Generate Monte Carlo stimuli for reproducing Table 2 of Papenberg & Klau (2022).

    For each run_id=1…n_runs, each cluster‐count K in k_values, 
    and each distribution in {uniform, normal_std, normal_wide}:

      1. Compute M_min = ceil(n_range_min / K)
         and M_max = floor(n_range_max / K).
      2. Draw M ~ UniformInt[M_min, M_max].
      3. Set N = M * K (so N is divisible by K and within [n_range_min, n_range_max]).
      4. Draw number of features F ~ UniformInt[1, 4].
      5. Sample a stimulus matrix X of shape (N, F) from the chosen distribution.

    Returns
    -------
    pd.DataFrame
        Columns:
          - run_id       : int, simulation index (1…n_runs)
          - K            : int, number of clusters
          - distribution : str, one of "uniform", "normal_std", "normal_wide"
          - N            : int, total sample size (multiple of K)
          - F            : int, number of features (1 to 4)
          - stimuli      : np.ndarray of shape (N, F), the generated data

    Raises
    ------
    ValueError
        If [n_range_min, n_range_max] does not contain any multiples of a given K.
    """
    rng = np.random.default_rng(rng_seed)

    # map distribution names to sampling lambdas
    dist_map = {
        "uniform":     lambda size: rng.uniform(**dstr_unif, size=size),
        "normal_std":  lambda size: rng.normal(**dstr_normal_std, size=size),
        "normal_wide": lambda size: rng.normal(**dstr_normal_wide, size=size),
    }
    dist_names = list(dist_map.keys())

    # prepare run‐K grid
    total_runs = n_runs * len(k_values)
    run_ids    = np.repeat(np.arange(1, n_runs + 1), len(k_values))
    ks         = np.tile(k_values, n_runs)
    dists      = rng.choice(dist_names, size=total_runs)

    records = []
    for run_id, K_, dist in zip(run_ids, ks, dists):
        # determine valid multiples of K_
        M_min = ceil(n_range_min / K_)
        M_max = n_range_max // K_
        if M_min > M_max:
            raise ValueError(
                f"No integer M in [ceil({n_range_min}/{K_}), floor({n_range_max}/{K_})]"
            )

        M = rng.integers(M_min, M_max + 1)
        N = M * K_

        # draw number of features
        F = rng.integers(1, 5)  # 1,2,3,4


        #Reduce running time of K3:
        if K_ == 3 and N > 30 and N < 40:
            if F > 2:
                N = 27
            else: 
                N = 30

        # sample the stimuli matrix
        X = dist_map[dist]((N, F))

        records.append({
            "run_id":       run_id,
            "K":            K_,
            "distribution": dist,
            "N":            N,
            "F":            F,
            "stimuli":      X,
        })

    df = pd.DataFrame(records)
    _LOG.info("Generated %d simulation records", len(df))
    _LOG.debug("First few rows:\n%s", df.head())
    return df
