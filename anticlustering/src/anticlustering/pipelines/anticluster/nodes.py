import time
from pathlib import Path
from typing import Dict, Tuple, List, Any

import pandas as pd
import matplotlib.pyplot as plt

from ...core import get_solver, AntiCluster, BaseConfig, ILPConfig, ExchangeConfig
from ...visualisation import PartitionVisualizer

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
            else: 
                cfg = BaseConfig(n_clusters=n_clusters, **spec)

            solver = get_solver(name, config=cfg)

            label = (
                "ILP" if name == "ilp" and not spec.get("preclustering")
                else "ILP_Precluster" if spec.get("preclustering")
                else "Exchange"
            )
           
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
            #bucket[label] = row
            bucket[label] = solver

        if store_models:
            model_bank[N] = bucket

    table = (
        pd.DataFrame(table_rows)
        .sort_values(["N", "solver"])
        .reset_index(drop=True)
    )

    fig = _plot_scores_with_gaps(
        table,
        n_clusters=n_clusters,
        log_y=True,  # log scale for better visibility
        pct_gap=False  # absolute gap
    )
    fig.tight_layout()

    return table, fig, (model_bank if store_models else None)



def _plot_scores_with_gaps(
    table: pd.DataFrame,
    n_clusters: int,
    *,
    log_y: bool = False,
    pct_gap: bool = False
) -> plt.Figure:
    """
    Two-panel figure:
        – upper: score vs N         (optionally log-scale)
        – lower: gap   vs N         (absolute or % of best score)
    """
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------
    # 1. merge the baseline scores onto every row
    # ---------------------------------------------------------------
    base = (
        table.loc[table["solver"] == "ILP", ["N", "score"]]
             .rename(columns={"score": "baseline"})
    )
    tbl = (
        table.merge(base, on="N", how="left", validate="many_to_one")
             .sort_values(["solver", "N"])
    )

    tbl["gap"] = tbl["score"] - tbl["baseline"]
    if pct_gap:
        tbl["gap"] = 100 * tbl["gap"] / tbl["baseline"]


    # ------------------------------------------------------------------
    # 2. create figure --------------------------------------------------
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1])
    )

    # ––– upper panel ––––––––––––––––––––––––––––––––––––––––––––––––
    for s, grp in tbl.groupby("solver"):
        ax0.plot(grp["N"], grp["score"], marker="o", label=s)
    ax0.set_ylabel("Objective value")
    ax0.set_title(f"Anticlustering objective vs N  (K = {n_clusters})")
    if log_y:
        ax0.set_yscale("log")
    ax0.grid(alpha=.3)
    ax0.legend(title="Solver", loc="upper left")

    # ––– lower panel ––––––––––––––––––––––––––––––––––––––––––––––––
    for s, grp in tbl.groupby("solver"):
        ax1.plot(grp["N"], grp["gap"], marker="o", label=s)
    ax1.set_xlabel("Problem size N")
    ax1.set_ylabel("Gap" + (" [%]" if pct_gap else ""))
    ax1.grid(alpha=.3)

    return fig
