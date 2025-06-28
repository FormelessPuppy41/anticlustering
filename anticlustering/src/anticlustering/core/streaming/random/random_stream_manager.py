# anticlustering/streaming/manager.py

from typing import Dict, List, Any

import logging

import numpy as np
import pandas as pd

from .random_simulator import RandomFeatureStreamSimulator
from .random_data_store import RandomStreamingDataStore
from ...online.online_base import BaseOnlineSolver
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
            solver.__class__.__name__: {} for solver in solvers
        }

    def run(self, collect_metrics: bool = False):
        """
        Execute the full stream. Returns metrics if requested.
        """
        metrics = {name: [] for name in self.assignments} if collect_metrics else None

        for new_ids, X_new, old_ids in self.sim:
            _LOG.debug(
                "run -manager: Processing step with %d new IDs and %d old IDs. \nExample new IDs: %s.\nExample old IDs: %s",
                len(new_ids), len(old_ids), new_ids[:5], old_ids[:5]
            )
            # 1) update store
            if old_ids:
                self.ds.remove_ids(old_ids)
            if new_ids:
                self.ds.add_features(new_ids, X_new)
            

            # 2) dispatch solvers
            for solver in self.solvers:
                name = solver.__class__.__name__

                # process departures
                self.assignments[name] = solver.remove_old(
                    self.ds, self.assignments[name], old_ids
                )
                
                # assign arrivals
                self.assignments[name] = solver.assign_new(
                    self.ds, self.assignments[name], new_ids
                )
                

                # optionally collect a snapshot of the objective
                if collect_metrics:
                    
                    #    ^ you may choose a different default for missing IDs
                    _LOG.debug(
                        "run -manager: assignments for %s: %s",
                        name, self.assignments[name]
                    )
                    val = solver.objective_value(self.ds, self.assignments[name])
                    metrics[name].append(val)
                    _LOG.info(
                        "run -manager: %s objective value at step %d: %.4f",
                        name, self.sim.current_step, val
                    )

        # Step 3: build DataFrame
        df = pd.DataFrame(metrics or {})

        # Step 4: add elementwise comparison columns as strings
        df['greedy>exchange'] = np.where(
            df['OnlineGreedySolver'] > df['OnlineExchangeSolver'],
            'greedy>onlineExchange',
            'onlineExchange>greedy'
        )
        df['greedy>baseline'] = np.where(
            df['OnlineGreedySolver'] > df['OfflineExchangeSolver'],
            'greedy>offlineExchange',
            'offlineExchange>greedy'
        )
        df['onlineExchange>baseline'] = np.where(
            df['OnlineExchangeSolver'] > df['OfflineExchangeSolver'],
            'onlineExchange>offlineExchange',
            'offlineExchange>onlineExchange'
        )

        return df
