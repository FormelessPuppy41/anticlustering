from typing import Dict, List, Union, Any
import pandas as pd
import numpy as np
from copy import deepcopy

from anticlustering.anticlustering import AntiCluster
from anticlustering.anticlustering.wrapper import SolverResult, SolverResults, _hash_array


# ------------------------------------------------------------------ #
def instantiate_solvers(cfg: Dict) -> List[AntiCluster]:
    """
    Build one AntiCluster instance per entry in cfg['solvers'].
    """
    common = {                     # shared hyper-parameters
        "n_groups"    : cfg["n_groups"],
        "max_iter"    : cfg["max_iter"],
        "random_state": cfg.get("random_state"),
    }
    solvers = []
    for solver_cfg in cfg["solvers"]:
        kwargs = common | solver_cfg           # python ≥3.9
        solvers.append(AntiCluster(**kwargs))
    return solvers


# ------------------------------------------------------------------ #
def solve_all(
    datasets: Dict[str, np.ndarray],               # tag -> X
    solvers: List[AntiCluster],
) -> tuple[List[SolverResults], pd.DataFrame]:

    wrappers, rows = [], []
    for tag, X in datasets.items():
        # --- check if X is a DataFrame -------------
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()                     # convert to ndarray
        # --- run every solver template -----------
        runs = {}
        for solver in solvers:
            name        = solver.solver.__class__.__name__.lower()
            solver      = deepcopy(solver)            # or deepcopy template
            labels      = solver.fit_predict(X)
            score       = solver.score(X)
            runs[name]  = (labels, score)

        # --- wrap & collect ----------------------
        wrapper = SolverResults.from_runs(tag, runs, X)
        wrappers.append(wrapper)
        rows.append(wrapper.to_table().assign(tag=tag))

    big_table = pd.concat(rows).set_index("tag", append=True)
    return wrappers, big_table, big_table
