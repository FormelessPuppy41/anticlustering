# anticlustering/streaming/manager.py

from typing import Dict, List, Any, Tuple

import logging
import time

import numpy as np
import pandas as pd

from .random_simulator import RandomFeatureStreamSimulator
from .random_data_store import RandomStreamingDataStore
from ...online.online_base import BaseOnlineSolver

from ....metrics.dissimilarity_matrix import _compute_M, _compute_SD
# You can import your offline baseline here:
# from anticlustering.solvers.offline.exchange import OfflineExchangeSolver

_LOG = logging.getLogger(__name__)



class StreamingExperimentManager:
    """
    Drives a full streaming experiment:
      1. step through simulator
      2. update data_store
      3. call each solver's assign/remove
      4. (optionally) collect objective metrics at each step
    """
    def __init__(
        self,
        simulator: RandomFeatureStreamSimulator,
        data_store: RandomStreamingDataStore,
        solvers: List[BaseOnlineSolver],             # e.g. [OnlineGreedySolver(cfg), OnlineExchangeSolver(cfg)]
    ):
        self.sim = simulator
        self.ds: RandomStreamingDataStore = data_store
        self.solvers = solvers

        
        # track assignments per solver: name -> {id -> cluster}
        self.assignments: Dict[str, Dict[str,int]] = {
            solver.name: {} for solver in solvers
        }

    def run(self, collect_metrics: bool = False) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Execute the full stream. Returns metrics if requested.
        """
        raw_metrics: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in self.assignments
        }
        
        for new_ids, X_new, old_ids in self.sim:
            _LOG.debug(
                "run -manager: Processing step with %d new IDs and %d old IDs. \nExample new IDs: %s.\nExample old IDs: %s",
                len(new_ids), len(old_ids), new_ids[:5], old_ids[:5]
            )
            step = self.sim.current_step
            # 1) update store
            if old_ids:
                t0 = time.time()
                self.ds.remove_ids(old_ids)
                data_remove_time = time.time() - t0
            else:
                data_remove_time = 0.0

            if new_ids:
                t0 = time.time()
                self.ds.add_features(new_ids, X_new)
                data_add_time = time.time() - t0
            else:
                data_add_time = 0.0
            

            # dispatch solvers
            for solver in self.solvers:
                name = solver.name

                # --- measure removal runtime ---
                t0 = time.time()
                self.assignments[name] = solver.remove_old(
                    self.ds, self.assignments[name], old_ids
                )
                solver_remove_time = time.time() - t0

                # --- measure assignment runtime ---
                t0 = time.time()
                self.assignments[name] = solver.assign_new(
                    self.ds, self.assignments[name], new_ids
                )
                solver_assign_time = time.time() - t0

                # optionally collect a snapshot of the objective
                if collect_metrics:
                    val = solver.objective_value(self.ds, self.assignments[name], objective='diversity')
                    labels = np.ndarray(
                        shape=(len(self.ds.ids),),
                        dtype=int,
                        buffer=np.array(list(self.assignments[name].values()))
                    )
                    M_val = _compute_M(self.ds.features, labels)
                    SD_val = _compute_SD(self.ds.features, labels)

                    raw_metrics[name].append({
                        'step': step,
                        'data_remove_time': data_remove_time,
                        'data_add_time':    data_add_time,
                        'solver_remove_time': solver_remove_time,
                        'solver_assign_time': solver_assign_time,
                        'objective': val,
                        'M': M_val,
                        'SD': SD_val,
                        'assignments': self.assignments[name].copy()
                    })
            
                    _LOG.debug(
                        "%s @ step %d: obj=%.4f | "
                        "data_remove=%.4f data_add=%.4f | "
                        "rem_old=%.4f asg_new=%.4f",
                        name, step, val,
                        data_remove_time, data_add_time,
                        solver_remove_time, solver_assign_time
                    )
            


        # build a DataFrame for each solver
        metrics_dfs: Dict[str, pd.DataFrame] = {}
        for name, records in raw_metrics.items():
            if records:
                metrics_dfs[name] = pd.DataFrame(records)
            else:
                # empty DataFrame with the right columns
                metrics_dfs[name] = pd.DataFrame(columns=[
                    'step','data_remove_time','data_add_time',
                    'solver_remove_time','solver_assign_time',
                    'objective', 'M', 'SD', 'assignments','aborted'
                ])
        
        obj_dict = {
            key: metrics_dfs[key]['objective'].values
            for key in metrics_dfs
        }

        baseline_df   = metrics_dfs[self.solvers[0].name].set_index('step')
        baseline_obj  = baseline_df['objective']

        tolerances = [95, 99]

        # 2) build summary rows
        summary_rows = []
        for solver_name, df in metrics_dfs.items():
            df_idx      = df.set_index('step')
            current_obj = df_idx['objective']
            pct_of_off  = (current_obj / baseline_obj) * 100
            pct_of_off.dropna(inplace=True)

            
            # number of steps
            S = len(df)

            # items per step (extract length of assignments dict)
            items_per_step = df['assignments'].apply(len)
            avg_items = items_per_step.mean()
            max_items = items_per_step.max()

            # solving time per step
            solve_time = df['solver_remove_time'] + df['solver_assign_time'] + df['data_remove_time'] + df['data_add_time']
            total_solve = solve_time.sum()
            avg_solve   = solve_time.mean()

            # final‐step metrics
            final_obj   = current_obj.iloc[-1]
            final_pct   = pct_of_off.iloc[-1]
            cutoff     = pct_of_off.quantile(0.75)
            avg_pct     = pct_of_off.mean()

            # start building the row
            row = {
                'solver':         solver_name,
                'n_steps':        S,
                'avg_items':      avg_items,
                'max_items':      max_items,
                'final_objective': final_obj,
                'final_%D':        avg_pct,
                'total_solve_time': total_solve,
                'avg_solve_time':   avg_solve,
                "K": solver.config.n_clusters
            }

            # now your p() and T() columns
            for tol in tolerances:
                row[f"p({tol}%)_%D"] = (pct_of_off >= tol).mean()
                
            # ΔM / ΔSD summaries
            row['AUC_ΔM']  = df['M'].mean()
            row['AUC_ΔSD'] = df['SD'].mean()

            

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        
        obj_df = pd.DataFrame(obj_dict)

        return summary_df, obj_df
