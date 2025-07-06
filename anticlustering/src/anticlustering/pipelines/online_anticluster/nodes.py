
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

    _LOG.debug(
        "simulate_stream: Simulating stream from %s to %s over %d loans",
        start or "min(issue_d)",
        end or "final departure",
        len(loans),
    )
    _LOG.debug("simulate_stream: Example loans: %s", loans[0] if loans else "No loans provided")

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
    _LOG.debug("simulate_stream: Generated %d monthly event rows", len(df_events))
    _LOG.debug("simulate_stream: Sample events:\n%s", df_events.head(48))
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


import time

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
    samples_per_bin: Union[int, List[int]],
    rng: np.random.Generator
) -> List[int]:
    """
    For each consecutive pair (lo, hi) in `breakpoints`, draw a number of
    samples in [lo, hi] and return the flat list of all draws.

    Parameters
    ----------
    breakpoints : List[int]
        Sorted list of bin boundaries.
    samples_per_bin : int or List[int]
        If int, number of samples to draw in each bin.
        If list, must have length len(breakpoints)-1, specifying the number
        of samples to draw for each corresponding interval.
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    List[int]
        Flattened list of sampled sizes.
    """
    # Determine how many samples to draw per interval
    num_bins = len(breakpoints) - 1
    if isinstance(samples_per_bin, int):
        counts = [samples_per_bin] * num_bins
    else:
        if len(samples_per_bin) != num_bins:
            raise ValueError(
                f"samples_per_bin list must have length {num_bins}, "
                f"got {len(samples_per_bin)}"
            )
        counts = samples_per_bin

    Ns: List[int] = []
    # Draw for each interval
    for (lo, hi), count in zip(zip(breakpoints[:-1], breakpoints[1:]), counts):
        draws = rng.integers(lo, hi + 1, size=count)
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

    n_steps_random_list = _sample_N_by_interval(breakpoints=n_steps_list, samples_per_bin=500, rng=rng)

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
    size_delta:     int   = 1
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
    results: Dict[str, Dict[str, pd.DataFrame]],
    bins: List[int] = [9, 50, 100, 150],
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
    labels = [str(bins[i]+1) + "-" + str(bins[i+1]) for i in range(len(bins)-1)]
    
    #labels = ["10–50", "51-100", "101–150"]
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







def sample_solve_aggregate(
    loans: List[LoanRecord],
    k: int,
    kaggle_cols: Dict[str, List[str]],
    metrics_cat_cols: List[str],
    hard_balance_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    1) For each N in n_steps_list:
       • sample N loans with replacement,
       • simulate events,
       • run anticlustering,
       • collapse to one summary row,
       • store under results[f"N{N}"]["online_exchange"].

    2) Call aggregate_results_by_bins(results) to get the final table.
    """
    # describe the loans set:
    _LOG.debug(
        "sample_solve_aggregate: Starting with %d loans",
        len(loans)
    )
    n_steps: List[int] = [10, 50, 100, 150]  # or any other list of N values you want to sample 
    # 50, 200, 
    sample_per_bin: List[int] = 500
    random_state: int = 42
    size_delta: int = 1  # balance slack for online methods
    objective: str = "diversity"  # or "variance"

    n_steps_list = _sample_N_by_interval(
        breakpoints=n_steps,
        samples_per_bin=sample_per_bin,  # number of samples per bin
        rng=np.random.default_rng(random_state)
    )

    i = 0
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for N in n_steps_list:
        print(f"Current itteration: {i}, sample size N={N}")
        i += 1
        # ————————————————————————————————————————————
        # 1a) sample with replacement
        sampled_loans = list(np.random.choice(loans, size=N))

        # 1b) simulate the stream
        events_df = simulate_stream(sampled_loans)

        solvers = []
        solvers.extend([
            OfflineExchangeSolver(ExchangeConfig(k, random_state, None, objective)),
            OnlineGreedySolver(OnlineGreedyConfig(k, size_delta, objective)),
            OnlineExchangeSolver(OnlineExchangeConfig(k, size_delta, objective)),
            OnlineGreedySolver(OnlineGreedyConfig(k, size_delta, objective, rebalance_method="incremental")),
            OnlineExchangeSolver(OnlineExchangeConfig(k, size_delta, objective, rebalance_method="incremental")),
        ])
        solvers_by_name = {solver.name: solver for solver in solvers}

        # 1c) solve via online-exchange anticlustering
        raw_metrics = update_anticlusters_multi(
            loans=sampled_loans,
            events_df=events_df,
            solvers=solvers,
            kaggle_cols=kaggle_cols,
            metrics_cat_cols=metrics_cat_cols,
            hard_balance_cols=hard_balance_cols,
        )

        baseline_name = solvers[0].name
        baseline_obj = (
            raw_metrics[baseline_name]
            .set_index("step")["objective"]
        )


        summary_dict: Dict[str, pd.DataFrame] = {}
        for solver_name, df in raw_metrics.items():
            df_idx = df.set_index("step")
            baseline_copy = baseline_obj.copy()

            # 1) steps both series actually have
            common = df_idx.index.intersection(baseline_copy.index)

            # 2) mask out steps where baseline==0 or current==0 or either is NaN
            base_valid = baseline_copy.loc[common].notna() & (baseline_copy.loc[common] != 0)
            curr_valid =  df_idx["objective"].loc[common].notna()  & (df_idx["objective"].loc[common] != 0)
            mask       = base_valid & curr_valid

            if mask.sum() == 0:
                # no valid comparison at all → fallback
                final_pct = 100.0
                p95       = 100.0
                p99       = 100.0
            else:
                good_steps = common[mask]
                current    = df_idx["objective"].loc[good_steps]
                baseline   = baseline_copy.loc[good_steps]

                pct_of_off = (current / baseline) * 100
                # drop infinities just in case, then NaNs
                pct_of_off = pct_of_off.replace([np.inf, -np.inf], np.nan).dropna()

                final_pct = pct_of_off.iloc[-1]
                cutoff = pct_of_off.quantile(0.75)
                avg_pct   = pct_of_off.mean()
                p95       = (pct_of_off >= 95).mean()
                p99       = (pct_of_off >= 99).mean()

            total_time = (df["solver_remove_time"] + df["solver_assign_time"]).sum()

            row = {
                "final_%D":         avg_pct,
                "AUC_ΔM":           df["M"].mean(),
                "AUC_ΔSD":          df["SD"].mean(),
                "total_solve_time": total_time,
                "p(95%)_%D":        p95,
                "p(99%)_%D":        p99,
                "K":                solvers_by_name[solver_name].config.n_clusters,
            }
            summary_dict[solver_name] = pd.DataFrame([row])

        results[f"N{N}_{i}"] = summary_dict

    df =  aggregate_results_by_bins(results, bins=n_steps)
    print(df)
    return df



import time
import ast
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from ...metrics.dissimilarity_matrix import _compute_M, _compute_SD

def update_anticlusters_multi(
    loans: List[LoanRecord],
    events_df: pd.DataFrame,
    solvers: List[BaseOnlineSolver],
    kaggle_cols: Dict[str, List[str]],
    metrics_cat_cols: List[str],
    hard_balance_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the same loan‐stream through multiple solvers and return, for each solver,
    a DataFrame of raw monthly metrics with exactly the fields:

      - step
      - data_remove_time
      - data_add_time
      - solver_remove_time
      - solver_assign_time
      - objective
      - M
      - SD
      - assignments (dict loan_id→cluster)

    This matches the synthetic StreamingExperimentManager’s raw_metrics output.
    """
    # Prepare the loan lookup & shared vectorizer
    loan_map = {ln.loan_id: ln for ln in loans}
    vec_cfg = LoanVectorizerConfig(
        kaggle_columns=kaggle_cols,
        num_scaler=StandardScaler(),
        cat_encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    # Initialize a list to collect records per solver
    raw_metrics: Dict[str, List[Dict[str, Any]]] = {}

    # 1) Instantiate one AnticlusterManager per solver, track name & assignment
    managers: Dict[str, AnticlusterManager] = {}
    for solver in solvers:
        name = solver.name  # assume each solver sets .name uniquely
        managers[name] = AnticlusterManager(
            solver=solver,
            vectorizer_config=vec_cfg,
            hard_balance_cols=hard_balance_cols,
        )
        raw_metrics[name] = []

    # 2) Step through the event stream
    step = 0
    for _, row in events_df.sort_values("date").iterrows():
        step += 1
        date = LoanRecord._parse_date(row["date"])

        # parse arrivals & departures
        arr = (ast.literal_eval(row["arrivals_ids"])
               if isinstance(row["arrivals_ids"], str)
               else row["arrivals_ids"])
        dep = (ast.literal_eval(row["departures_ids"])
               if isinstance(row["departures_ids"], str)
               else row["departures_ids"])
        arrivals   = [loan_map[i] for i in arr]
        departures = [loan_map[i] for i in dep]

        # for each solver, do the three‐stage update & time each piece
        for name, mgr in managers.items():
            # 2a) departures: data remove + solver.remove_old
            t0 = time.perf_counter()
            mgr.store.remove_loans(dep)
            data_remove_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            mgr.assignments = mgr.solver.remove_old(mgr.store, mgr.assignments, dep)
            solver_remove_time = time.perf_counter() - t1

            mgr._rebuild_group_states()  # keep group state in sync

            # 2b) arrivals: data add + solver.assign_new
            t2 = time.perf_counter()
            new_ids = [lo.loan_id for lo in arrivals]
            mgr.store.add_loans(arrivals)
            data_add_time = time.perf_counter() - t2

            t3 = time.perf_counter()
            mgr.assignments = mgr.solver.assign_new(mgr.store, mgr.assignments, new_ids)
            solver_assign_time = time.perf_counter() - t3

            mgr._rebuild_group_states()

            # 2c) rebalance
            t4 = time.perf_counter()
            mgr.assignments = mgr.solver.rebalance(mgr.store, mgr.assignments)
            solver_rebalance_time = time.perf_counter() - t4

            mgr._rebuild_group_states()

            # 3) compute “objective” and dissimilarity metrics
            #    assume your solver implements objective_value(...)
            obj_val = mgr.solver.objective_value(
                mgr.store, mgr.assignments, objective=mgr.solver.config.objective
            )

            labels = np.array([mgr.assignments[lid] for lid in mgr.store.ids], dtype=int)
            
            M_val  = _compute_M(mgr.store.features, labels)
            SD_val = _compute_SD(mgr.store.features, labels)

            # 4) record the row exactly like raw_metrics in the synthetic manager
            raw_metrics[name].append({
                "step":               step,
                "date":               date,
                "data_remove_time":   data_remove_time,
                "data_add_time":      data_add_time,
                "solver_remove_time": solver_remove_time,
                "solver_assign_time": solver_assign_time + solver_rebalance_time,
                "objective":          obj_val,
                "M":                  M_val,
                "SD":                 SD_val,
                "assignments":        mgr.assignments.copy(),
            })

    # 5) Convert each list-of-dicts into a DataFrame
    return { name: pd.DataFrame(records) for name, records in raw_metrics.items() }
    




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
