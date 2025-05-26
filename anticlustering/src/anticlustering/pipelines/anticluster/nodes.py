import time
from typing import Dict, Tuple, List, Any

import pandas as pd
from ...core import get_solver, AntiCluster

def benchmark_all(
    data: Dict[str, pd.DataFrame],
    n_clusters: int,
    ilp_max_n: int,
    precluster_max_n: int,
    store_models: bool,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, 'AntiCluster']] | None]:
    """
    Run three solvers on every simulated matrix in *data*.

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
        configs = [
            ("ilp", {}),                       # Exact ILP
            ("ilp", {"preclustering": True}),  # ILP + preclustering
            ("cluster_editing", {"method": "exchange"}),  # Exchange
        ]

        for name, kwargs in configs:
            # --- skip according to N limits ---
            if name == "ilp" and not kwargs.get("preclustering") and N > ilp_max_n:
                continue
            if name == "ilp" and kwargs.get("preclustering") and N > precluster_max_n:
                continue

            label = (
                "ILP" if name == "ilp" and not kwargs.get("preclustering")
                else "ILP/precluster" if kwargs.get("preclustering")
                else "Exchange"
            )

            t0 = time.perf_counter()
            solver = get_solver(name, n_clusters=n_clusters, **kwargs)
            solver.fit(X)
            runtime = time.perf_counter() - t0

            table_rows.append(dict(N=N, solver=label, runtime=runtime))
            bucket[label] = solver

        if store_models:
            model_bank[N] = bucket

    table = (
        pd.DataFrame(table_rows)
        .sort_values(["N", "solver"])
        .reset_index(drop=True)
    )

    return table, (model_bank if store_models else None)
