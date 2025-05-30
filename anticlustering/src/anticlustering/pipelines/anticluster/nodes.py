import time
from typing import Dict, Tuple, List, Any

import pandas as pd
from ...core import get_solver, AntiCluster, BaseConfig, ILPConfig

def benchmark_all(
    data            : Dict[str, pd.DataFrame],
    n_clusters      : int,
    solvers         : list[dict],
    store_models    : bool,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, 'AntiCluster']] | None]:
    """
    Run the solvers on every simulated matrix in *data*.

    Returns
    -------
    table : DataFrame
        Columns = [N, solver, runtime]
    all_models : dict | None
        {N: {solver_label: fitted_solver}}, or None if *store_models* is False
    """
    table_rows: List[Dict[str, Any]] = []
    model_bank: Dict[int, Dict[str, 'AntiCluster']] = {}

    for key, df in data.items():          # key = "N_10", df = DataFrame
        N = int(key.split("_")[1])
        X = df.values
        bucket = {}

        # ---------- decide which solvers to run ----------
        for spec in solvers:
            name = spec['solver_name'].lower()
            spec = {k: v for k, v in spec.items() if k != 'solver_name'}

            if name == "ilp":
                cfg = ILPConfig(n_clusters=n_clusters, **spec)
            else: 
                cfg = BaseConfig(n_clusters=n_clusters, **spec)

            solver = get_solver(name, config=cfg)

            label = (
                "ILP" if name == "ilp" and not spec.get("preclustering")
                else "ILP/precluster" if spec.get("preclustering")
                else "Exchange"
            )
            print(f"Running {label} on N={N}...")
            solver.fit(X)
            
            # Set runtime to nan if it did not finish. That is, labels are not set.
            row = dict(
                N=N,
                solver=label,
                runtime=solver.runtime_,
                score=solver.score_,
                status=solver.status_,
                aborted=solver.status_ != 'ok',
                gap=solver.gap_,
            )
            table_rows.append(row)
            bucket[label] = row

        if store_models:
            model_bank[N] = bucket

    table = (
        pd.DataFrame(table_rows)
        .sort_values(["N", "solver"])
        .reset_index(drop=True)
    )

    return table, (model_bank if store_models else None)
