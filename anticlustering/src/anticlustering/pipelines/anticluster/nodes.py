import time
from pathlib import Path
from typing import Dict, Tuple, List, Any
import logging

import pandas as pd
import matplotlib.pyplot as plt

from ...core import get_solver, AntiCluster, BaseConfig, ILPConfig, ExchangeConfig, OnlineConfig
from ...visualisation import PartitionVisualizer

from ...metrics.dissimilarity_matrix import within_group_distance, get_dissimilarity_matrix

_LOG = logging.getLogger(__name__)

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
            elif name == "ilp_precluster":
                cfg = ILPConfig(n_clusters=n_clusters,**spec)
            elif name == "exchange":
                cfg = ExchangeConfig(n_clusters=n_clusters, **spec)
            elif name == "online":
                cfg = OnlineConfig(n_clusters=n_clusters, **spec)
            else: 
                cfg = BaseConfig(n_clusters=n_clusters, **spec)

            solver = get_solver(name, config=cfg)

            label = (
                "ILP" if name == "ilp" and not spec.get("preclustering")
                else "ILP_Precluster" if spec.get("preclustering")
                else "Exchange"
            )
           
            solver.fit(X)

            # Check try value of score:
            D = get_dissimilarity_matrix(X)
            true_score = within_group_distance(D, solver.labels_)
            if abs(true_score - solver.score_) > 1e-6:
                _LOG.warning(
                    "Solver %s returned a score of %.2f, but the true score is %.2f. "
                    "This may indicate an issue with the solver implementation.",
                    label, solver.score_, true_score
                )
            else:
                _LOG.info(
                    "Solver %s returned a correct score of %.2f.",
                    label, solver.score_
                )

            # Set runtime to nan if it did not finish. That is, labels are not set.
            try:
                ilp_solve = solver.runtime_ilp_
                pre_solve = solver.runtime_pre_
            except:
                ilp_solve = pre_solve = float('nan')

            row = dict(
                N=N,
                solver=label,
                runtime=solver.runtime_,
                runtime_ilp=ilp_solve,
                runtime_pre=pre_solve,
                score=solver.score_,
                status=solver.status_,
                aborted=solver.status_ != 'ok',
                gap=solver.gap_,
            )
            table_rows.append(row)
            #bucket[label] = row
            bucket[label] = solver

        if store_models:
            model_bank[N] = bucket

    table = (
        pd.DataFrame(table_rows)
        .sort_values(["N", "solver"])
        .reset_index(drop=True)
    )

    fig = PartitionVisualizer.plot_scores_with_gaps(
        table,
        n_clusters=n_clusters,
        log_y=True,  # log scale for better visibility
        pct_gap=False  # absolute gap
    )
    fig.tight_layout()

    return table, fig, (model_bank if store_models else pd.DataFrame())




