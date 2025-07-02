
from __future__ import annotations

"""
This is a boilerplate pipeline 'online_anticluster'
generated using Kedro 0.19.13
"""
"""
pipelines/simulate_stream.py
============================

Pipeline #3 ― *Simulate the arrival / departure stream*

Inputs
------
* ``loans_raw``      : List[LoanRecord]     (from ingest pipeline)

Parameters (conf/*/parameters.yml)
----------------------------------
stream_start_date        : "YYYY-MM-DD"  # null ⇒ min(issue_d)
stream_end_date          : "YYYY-MM-DD"  # null ⇒ until last loan departs

Outputs
-------
* ``stream_monthly_events`` : pandas.DataFrame  (one row per calendar month)

Schema of *stream_monthly_events*
---------------------------------
date            datetime64[ns]  (month-end, always the 1st of month)
arrivals_ids    object          (Python list[str])
departures_ids  object          (Python list[str])

Author
------
Your Name  <your.email@example.com>
"""

import ast
import datetime as _dt
import logging
from typing import Dict, List, Sequence, Optional, Any

import pandas as pd
import numpy as np

from ...core.loans.loan import LoanRecord, LoanRecordFeatures
from ...core.loans.vectorizer import LoanVectorizer
from ...core.online._registry import get_online_solver
from ...streaming.stream import StreamEngine
from ...core.streaming.stream_manager import AnticlusterManager
from ...streaming.quality_metrics import (
    balance_score_categorical,
    within_group_variance,
)


_LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                               Node functions                                #
# --------------------------------------------------------------------------- #


def simulate_stream(
    loans: List[LoanRecord],
    stream_start_date: str | None = None,
    stream_end_date: str | None = None,
) -> pd.DataFrame:
    """
    Run `StreamEngine` month-by-month and emit an **event log** DataFrame.

    Parameters
    ----------
    loans
        Parsed LoanRecord list.
    stream_start_date, stream_end_date
        ISO-date strings (``YYYY-MM-DD``) or *null*.

    Returns
    -------
    pd.DataFrame
        One row per calendar month with two list-columns: arrivals_ids,
        departures_ids.  These are consumed by the *update_anticluster*
        pipeline.
    """
    start: _dt.date | None = (
        _dt.date.fromisoformat(stream_start_date) if stream_start_date else None
    )
    end: _dt.date | None = (
        _dt.date.fromisoformat(stream_end_date) if stream_end_date else None
    )

    _LOG.info(
        "simulate_stream: Simulating stream from %s to %s over %d loans",
        start or "min(issue_d)",
        end or "final departure",
        len(loans),
    )
    _LOG.info("simulate_stream: Example loans: %s", loans[0] if loans else "No loans provided")

    engine = StreamEngine(loans, start_date=start, end_date=end)

    records: List[dict] = []
    for date, arrivals, departures in engine.run():
        records.append(
            {
                "date": pd.Timestamp(date),
                "arrivals_ids": [lo.loan_id for lo in arrivals],
                "departures_ids": [lo.loan_id for lo in departures],
            }
        )

    df_events = pd.DataFrame(records)
    _LOG.info("simulate_stream: Generated %d monthly event rows", len(df_events))
    _LOG.info("simulate_stream: Sample events:\n%s", df_events.head(48))
    return df_events






from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ...core.online._config import OnlineExchangeConfig, OnlineGreedyConfig, OnlineDenStreamConfig
from ...core.loans.vectorizer import LoanVectorizerConfig

_LOG = logging.getLogger(__name__)


def update_anticlusters(
    loans: List[LoanRecord],
    events_df: pd.DataFrame,
    k: int,
    kaggle_cols: Dict[str, List[str]],
    metrics_cat_cols: List[str],
    hard_balance_cols: List[str] | None = None,
) -> List[pd.DataFrame]:
    """
    Processes a stream of monthly events, returning:
      - df_assign: long table of (date, loan_id, group)
      - df_metrics: wide table of metrics per date
    """

    # Map loan_id → record
    loan_map = {ln.loan_id: ln for ln in loans}
    vector_config = LoanVectorizerConfig(
        kaggle_columns=kaggle_cols,
        num_scaler=StandardScaler(),
        cat_encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False),
    )
    online_config = OnlineGreedyConfig(n_clusters=k)
    online_config = OnlineExchangeConfig(n_clusters=k)
    
    # Initialize manager with chosen solver
    # online_exchange
    # online_greedy
    solver = get_online_solver("online_exchange", config=online_config)  # or pass in solver_name via params
    mgr = AnticlusterManager(
        solver=solver,
        vectorizer_config=vector_config,
        hard_balance_cols=hard_balance_cols
    )

    assignments_rows: List[Dict[str,Any]] = []
    metrics_rows:     List[Dict[str,Any]] = []
    prev_assignment: Dict[str,int] = {}

    # Process each event date in order
    for _, row in events_df.sort_values("date").iterrows():
        date = LoanRecord._parse_date(row["date"])

        # parse arrivals/departures lists
        raw_arr = row["arrivals_ids"]
        arr_ids = ast.literal_eval(raw_arr) if isinstance(raw_arr, str) else row["arrivals_ids"]
        raw_dep = row["departures_ids"]
        dep_ids = ast.literal_eval(raw_dep) if isinstance(raw_dep, str) else row["departures_ids"]

        arrivals   = [loan_map[lid] for lid in arr_ids]
        departures = [loan_map[lid] for lid in dep_ids]

        # feed departures
        mgr.on_departure(departures)

        # feed arrivals
        mgr.on_arrival(arrivals)

        # rebalance if needed
        if arrivals or departures:
            mgr.on_rebalance()

        # snapshot & record any moves
        current = mgr.get_assignments()  # Dict[str,int]
        for lid, grp in current.items():
            old = prev_assignment.get(lid)
            if old is None or old != grp:
                assignments_rows.append({"date": date, "loan_id": lid, "group": grp})
        prev_assignment = current.copy()

        # compute metrics row
        row_metrics: Dict[str,Any] = {
            "date": date,
            "group_sizes": mgr.group_sizes,
            "within_var": within_group_variance(mgr, loan_map),
            "group_centroids": mgr.centroids,
        }
        for cat in metrics_cat_cols:
            row_metrics[f"balance_{cat}"] = balance_score_categorical(mgr, loan_map, cat)

        metrics_rows.append(row_metrics)

    # assemble outputs exactly as before
    df_assign = pd.DataFrame(assignments_rows)
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics["group_sizes"] = df_metrics["group_sizes"].apply(np.array)

    return [df_assign, df_metrics]



from ...core.streaming.random.random_data_store import RandomStreamingDataStore
from ...core.streaming.random.random_simulator import RandomFeatureStreamSimulator
from ...core.streaming.random.random_stream_manager import StreamingExperimentManager

from ...core.online.offline_baseline import OfflineExchangeSolver, ExchangeConfig
from ...core.online.online_base import BaseOnlineSolver, OnlineBaseConfig
from ...core.online.online_greedy import OnlineGreedySolver, OnlineGreedyConfig
from ...core.online.online_exchange import OnlineExchangeSolver, OnlineExchangeConfig

from ...metrics.dissimilarity_matrix import get_dissimilarity_matrix

from ...core.online._registry import get_online_solver

def _sample_N_by_interval(
    breakpoints: List[int],
    samples_per_bin: int,
    rng: np.random.Generator
) -> List[int]:
    """
    For each consecutive pair (lo, hi) in `breakpoints`, draw
    `samples_per_bin` integers uniformly in [lo, hi], and return
    the flat list of all draws.
    """
    Ns = []
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:]):
        # draw integers in [lo, hi]
        draws = rng.integers(lo, hi + 1, size=samples_per_bin)
        Ns.extend(draws.tolist())
    return Ns

def simulate_online_data(
) -> Dict[str, RandomFeatureStreamSimulator]:
    """
    Generate a dictionary of RandomFeatureStreamSimulator’s over a small design grid.
    Keys encode the scenario so you can easily match results later.

    Returns
    -------
    sims : dict
      {
        "N100_D2_A1.0_R0.05_normal": simulator,
        "N100_D2_A1.0_R0.05_uniform": simulator,
        …
      }
    """
    n_steps_list:       List[int]           = [10, 50, 100, 150]
    feature_dims:       List[int]           = [1, 2, 3, 4]
    arrival_rates:      List[float]         = [1.0, 2.0]
    retentions:         List[float]         = [0.05, 0.1]
    distributions:      List[str]           = ["normal", "normal_wide", "uniform"]
    dist_params:        Optional[Dict[str, dict]] = None
    random_state:       Optional[int]       = 42

    rng = np.random.default_rng(random_state)
    sims: Dict[str, RandomFeatureStreamSimulator] = {}

    n_steps_random_list = _sample_N_by_interval(breakpoints=n_steps_list, samples_per_bin=5, rng=rng)

    for n_steps in n_steps_random_list: # 5x
        dim = int(rng.choice(feature_dims))
        rate = float(rng.choice(arrival_rates))
        ret = float(rng.choice(retentions))
        dist = str(rng.choice(distributions))
        params = (dist_params or {}).get(dist, {})

        sim = RandomFeatureStreamSimulator(
            n_steps=n_steps,
            feature_dim=dim,
            arrival_rate=rate,
            retention=ret,
            distribution=dist,
            dist_params=params,
            random_state=random_state,
        )
        key = f"N{n_steps}_D{dim}_A{rate}_R{ret}_{dist}"
        sims[key] = sim

    return sims

def simulate_online_solvers(
    sims:    Dict[str, RandomFeatureStreamSimulator],
) -> Dict[str, dict]:
    """
    Run each of your online solvers across every simulator generated above.

    Parameters
    ----------
    sims : dict of RandomFeatureStreamSimulator
      Output of simulate_online_data().
    n_clusters : int
      # of clusters for each solver.
    size_delta : int
      balance slack for the online methods.
    collect_metrics : bool
      If True, returns whatever manager.run() collects (per-step histories).

    Returns
    -------
    results : dict
      {
        scenario_key: {
          'baseline':  <metrics dict>,
          'greedy':    <metrics dict>,
          'exchange':  <metrics dict>
        },
        …
      }
    """
    n_clusters:     List[int]   = [2]
    size_delta:     int   = 5
    random_state:   int   = 42
    collect_metrics: bool = True

    i = 0
    results: Dict[str, dict] = {}
    for key, sim in sims.items():
        print(f"Current simulation: {i}")
        i += 1
        # 1) New, empty data‐store
        ds = RandomStreamingDataStore(feature_dim=sim.feature_dim)

        objective = "diversity" # diversity or variance
        # 2) Instantiate your solvers
        solvers: List[BaseOnlineSolver] = []
        for K in n_clusters:
            solvers.extend([
                OfflineExchangeSolver(ExchangeConfig(K, random_state, None, objective)),
                OnlineGreedySolver(OnlineGreedyConfig(K, size_delta, objective)),
                OnlineExchangeSolver(OnlineExchangeConfig(K, size_delta, objective)),
                OnlineGreedySolver(OnlineGreedyConfig(K, size_delta, objective, rebalance_method="incremental")),
                OnlineExchangeSolver(OnlineExchangeConfig(K, size_delta, objective, rebalance_method="incremental")),
            ])

        # 3) Fire up the manager
        manager = StreamingExperimentManager(
            simulator=sim,
            data_store=ds,
            solvers=solvers
        )

        # 4) Run it
        summary_df, _ = manager.run(collect_metrics=collect_metrics)

        # add the scenario key so we can extract N later
        summary_df = summary_df.copy()
        summary_df["scenario"] = key

        # store each solver's summary_df
        by_solver = { name: df for name, df in summary_df.groupby("solver") }

        results[key] = by_solver

    return results

def aggregate_results_by_bins(
    results: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Flatten the results dict, bin by N, and compute the Table 2 aggregates.
    """
    # 1) collect all summary rows
    records = []
    for scenario, solver_dfs in results.items():
        for solver_name, df in solver_dfs.items():
            # each df here is the scenario's summary_df filtered by solver
            # it has one row per scenario (since manager.run builds one summary row)
            row = df.iloc[0]
            records.append({
                "scenario":   scenario,
                "solver":     solver_name,
                "final_%D":   row["final_%D"],
                "AUC_ΔM":     row["AUC_ΔM"],
                "AUC_ΔSD":    row["AUC_ΔSD"],
                "avg_total_solve_time": row["total_solve_time"],
                "p(95%)_%D": row["p(95%)_%D"],
                "p(99%)_%D": row["p(99%)_%D"],
                "K":         row["K"],
            })

    summary_all = pd.DataFrame.from_records(records)

    # 2) extract N
    summary_all["N"] = (
        summary_all["scenario"]
        .str.extract(r"N(\d+)_")  # capture digits after the leading "N"
        .astype(int)
    )

    # 3) define and assign bins
    bins   = [10, 50, 100, 150]
    labels = ["10–50", "51-100", "101–150"]
    summary_all["N_bin"] = pd.cut(summary_all["N"], bins=bins, labels=labels)

    # 4) group and aggregate
    table2 = (
        summary_all
        .groupby(["solver", "N_bin", "K"], observed=True)
        .agg(
            pct_D    = ("final_%D",  "mean"),
            AUC_Delta_M  = ("AUC_ΔM",    "mean"),
            AUC_Delta_SD = ("AUC_ΔSD",   "mean"),
            runs     = ("scenario",  "size"),
            avg_solve_time = ("avg_total_solve_time", "mean"),
            avg_95pct_d = ("p(95%)_%D", "mean"),
            avg_99pct_d = ("p(99%)_%D", "mean")
        )
        .reset_index()
    )

    
    return table2



def simulate_solvers(
    n_steps: int = 150,
    feature_dim: int = 2,
    arrival_rate: float = 2.0,
    retention: float = 0.05,
    distribution: str = "normal",
    dist_params: Optional[dict] = None,
    random_state: Optional[int] = 42,
    n_clusters: int = 2,
    size_delta: int = 5,
    collect_metrics: bool = True,
) -> Dict[str, List[float]]:
    """
    Run a streaming anticlustering comparison between the Greedy and Exchange solvers.

    Parameters
    ----------
    n_steps
        Number of time‐steps to simulate.
    feature_dim
        Dimensionality of each random feature vector.
    arrival_rate
        Mean new arrivals per step (Poisson).
    retention
        If <1: per‐step departure probability. If >=1: fixed window size.
    distribution
        "normal" or "uniform".
    dist_params
        Extra kwargs for the distribution:
          - normal: {"loc":…, "scale":…}
          - uniform: {"low":…, "high":…}
    random_state
        RNG seed.
    n_clusters
        # of clusters for each solver.
    size_delta
        Allowed size imbalance.
    collect_metrics
        If True, returns per‐step objective values for each solver.

    Returns
    -------
    metrics : dict (solver_name → list of objective values)
        If `collect_metrics=False`, returns an empty dict.
    """
    # 1) Build simulator
    sim = RandomFeatureStreamSimulator(
        n_steps=n_steps,
        feature_dim=feature_dim,
        arrival_rate=arrival_rate,
        retention=retention,
        distribution=distribution,
        dist_params=dist_params,
        random_state=random_state,
    )

    # 2) Initialize data store
    ds = RandomStreamingDataStore(feature_dim=0)

    objective = "diversity"  # "variance" or "diversity"

    # 3) Instantiate solvers
    baseline = OfflineExchangeSolver(
        config=ExchangeConfig(
            n_clusters=n_clusters,
            random_state=random_state,
            time_limit=None,
            objective=objective
        )
    )

    greedy = OnlineGreedySolver(
        config= OnlineGreedyConfig(
            n_clusters=n_clusters, 
            size_delta=size_delta,
            objective=objective
        )
    )
    exchange = OnlineExchangeSolver(
        config= OnlineExchangeConfig(
            n_clusters=n_clusters,
            size_delta=size_delta,
            objective=objective
        )
    )

    solvers: List[BaseOnlineSolver] = [baseline, greedy, exchange]

    # 4) Create experiment manager
    manager = StreamingExperimentManager(
        simulator=sim,
        data_store=ds,
        solvers=solvers
    )

    # 5) Run simulation
    metrics, obj_metric = manager.run(collect_metrics=collect_metrics)
    _LOG.info(
        "simulate_solvers: Completed %d steps with %d clusters",
        n_steps,
        n_clusters
    )
    # 6) Return collected metrics (empty if collect_metrics=False)
    return metrics
