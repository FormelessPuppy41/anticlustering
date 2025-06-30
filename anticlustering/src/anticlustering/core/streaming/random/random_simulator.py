# anticlustering/streaming/simulation.py

import uuid
from collections import deque
from typing import (
    Deque, Dict, Generator, List, Optional, Tuple, Union
)

import numpy as np


class RandomFeatureStreamSimulator:
    """
    Generate a streaming sequence of arrival/departure events with random feature vectors.

    Yields:
        new_ids: List[str]                 # UUIDs of newly arriving items
        X_new: np.ndarray                  # shape (len(new_ids), feature_dim)
        old_ids: List[str]                 # IDs departing at this step
    """
    def __init__(
        self,
        n_steps: int,
        feature_dim: int,
        arrival_rate: float = 1.0,
        retention: Union[int, float] = 100,
        distribution: str = "normal",
        dist_params: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_steps = n_steps
        self.feature_dim = feature_dim
        self.arrival_rate = arrival_rate
        self.retention = retention
        self.distribution = distribution
        self.dist_params = dist_params or {}
        self.rng = np.random.default_rng(random_state)
        # keep (id, birth_step)
        self._queue: Deque[Tuple[str,int]] = deque()

        self.current_step = 0

    def __iter__(self) -> Generator[
        Tuple[List[str], np.ndarray, List[str]], None, None
    ]:
        for step in range(self.n_steps):
            # --- arrivals ---
            n_new = self.rng.poisson(self.arrival_rate)
            new_ids: List[str] = []
            X_new = np.zeros((0, self.feature_dim))
            if n_new > 0:
                new_ids = [str(uuid.uuid4()) for _ in range(n_new)]
                if self.distribution == "normal":
                    loc = self.dist_params.get("loc", 0.0)
                    scale = self.dist_params.get("scale", 1.0)
                    X_new = self.rng.normal(loc, scale, size=(n_new, self.feature_dim))
                elif self.distribution == "normal_wide":
                    loc = self.dist_params.get("loc", 0.0)
                    scale = self.dist_params.get("scale", 2.0)
                    X_new = self.rng.normal(loc, scale, size=(n_new, self.feature_dim))
                elif self.distribution == "uniform":
                    low = self.dist_params.get("low", 0.0)
                    high = self.dist_params.get("high", 1.0)
                    X_new = self.rng.uniform(low, high, size=(n_new, self.feature_dim))
                else:
                    raise ValueError(f"Unknown distribution {self.distribution!r}")

                for lid in new_ids:
                    self._queue.append((lid, step))

            # --- departures ---
            old_ids: List[str] = []
            if isinstance(self.retention, int):
                while self._queue and (step - self._queue[0][1] >= self.retention):
                    lid, _ = self._queue.popleft()
                    old_ids.append(lid)
            else:
                survivors = deque()
                for lid, birth in self._queue:
                    if self.rng.random() < self.retention:
                        old_ids.append(lid)
                    else:
                        survivors.append((lid, birth))
                self._queue = survivors

            self.current_step += 1
            
            yield new_ids, X_new, old_ids
