import time
from pathlib import Path
from typing import Dict, Tuple, List, Any
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ...core import get_solver, AntiCluster, BaseConfig, ILPConfig, ExchangeConfig, OnlineConfig
from ...core._config import MatchingConfig, KMeansConfig, RandomConfig
from ...visualisation import PartitionVisualizer

from ...metrics.dissimilarity_matrix import get_dissimilarity_matrix, diversity_objective

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
        D = get_dissimilarity_matrix(X)
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
            true_score = diversity_objective(D, solver.labels_)
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


# ------- Simulation BenchMark ----------------------

def _compute_M(X: np.ndarray, labels: np.ndarray) -> float:
    # mean difference across clusters, averaged over features
    df = pd.DataFrame(X)
    means = df.groupby(labels).mean().values  # shape (K, n_features)
    return np.mean(np.ptp(means, axis=0))

def _compute_SD(X: np.ndarray, labels: np.ndarray) -> float:
    # std‐difference across clusters, averaged over features
    df = pd.DataFrame(X)
    stds = df.groupby(labels).std(ddof=1).values
    return np.mean(np.ptp(stds, axis=0))


def benchmark_simulation(
    simulation_data : pd.DataFrame,
    solvers        : List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Run each solver on every simulation run and aggregate into Table 2.

    Simulation details as in Papenberg & Klau (2019): 10 000 runs
    (5 000 with K=2; 5 000 with K=3); N∈[10,100] multiple of K; F∈[1,4];
    dist ∈ {uniform[0,1], N(0,1), N(0,2)} :contentReference[oaicite:0]{index=0}.
    Results are binned by N∈{10–20,21–40,42–100} as in Table 2 :contentReference[oaicite:1]{index=1}.
    """
    rows = []
    for _, r in simulation_data.iterrows():
        run = int(r["run_id"])
        K   = int(r["K"])
        N   = int(r["N"])
        X   = np.array(r["stimuli"])
        D   = get_dissimilarity_matrix(X)

        # run each solver
        for spec in solvers:
            name = spec["solver_name"].lower()
            specs = {k: v for k, v in spec.items() if k != 'solver_name'}

            if "random_state" in specs and specs["random_state"] is not None:
                solver_idx = solvers.index(spec)
                specs["random_state"] = run * 100 + solver_idx

            # skip matching if K!=2
            if name == "matching" and K != 2:
                continue
            # ILP only for N == solver_limits["ilp_max_n"]
            if name == "ilp" and N > 20:
                continue
            # ILP+precluster only for N == solver_limits["precluster_max_n"]
            if name == "ilp_precluster" and N > 40:
                continue

            # build config
            if name == "ilp":
                cfg = ILPConfig(n_clusters=K, **specs)
            elif name == "ilp_precluster":
                cfg = ILPConfig(n_clusters=K, **specs)
            elif name == "exchange":
                cfg = ExchangeConfig(n_clusters=K, **specs)
            elif name == "matching":
                cfg = MatchingConfig(n_clusters=K, **specs)
            elif name == "kmeans":
                cfg = KMeansConfig(n_clusters=K, **specs)
            elif name == "random":
                cfg = RandomConfig(n_clusters=K, **specs)
            else:
                cfg = BaseConfig(n_clusters=K, **specs)

            cfg.n_clusters = K  # ensure K is set correctly

            solver = get_solver(name, config=cfg)
            solver.fit(X, D=D)

            # record raw score + diagnostics
            score = solver.score_
            labels = solver.labels_
            M_val = _compute_M(X, labels)
            SD_val = _compute_SD(X, labels)

            rows.append({
                "run": run,
                "K": K,
                "N": N,
                "solver": name,
                "score": score,
                "M": M_val,
                "SD": SD_val,
            })

    df = pd.DataFrame(rows)

    # bin N into the three ranges
    df["N_range"] = pd.cut(
        df["N"],
        bins=[9, 20, 40, 100],
        labels=["10–20","21–40","42–100"]
    )

    # compute percent of best per run/K
    # Define which solver is “optimal” in each N‐bin
    benchmark_map = {
        "10–20": "ilp",
        "21–40": "ilp_precluster",
        "42–100": "exchange",
    }

    # Extract the benchmark scores for each (run, K, N_range)
    # 1. Filter df down to only those benchmark rows
    bench_df = df[
        df["solver"] == df["N_range"].map(benchmark_map)
    ][["run", "K", "N_range", "score"]].rename(columns={"score": "best_score"})

    # 2. Merge best_score back onto the full df
    df = df.merge(
        bench_df,
        on=["run", "K", "N_range"],
        how="left"
    )

    # 3. Compute percent
    df["percent"] = df["score"] / df["best_score"] * 100

    # aggregate means & SDs
    table2 = (
        df
        .groupby(["K", "N_range", "solver"], observed=True)
        .agg(
            pct_D_within = ("percent", "mean"),
            avg_delta_M  = ("M",       "mean"),
            avg_delta_SD = ("SD",      "mean"),
        )
        .reset_index()
        .sort_values(["N_range", "K", "solver"], ascending=[True, True, True])
    )

    return table2
