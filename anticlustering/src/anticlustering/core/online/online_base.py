# anticlustering/solvers/online/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from ...streaming.data_store import StreamingDataStore
from ...core.online._config import OnlineBaseConfig

class BaseOnlineSolver(ABC):
    """
    Stateless solver interface.
    All data lives in StreamingDataStore; solver only receives snapshots.
    """
    def __init__(self, config: OnlineBaseConfig) -> None:
        """
        Initialize with configuration parameters.
        """
        self.config = config

    @abstractmethod
    def assign_new(
        self,
        data: StreamingDataStore,
        prev_assignments: Dict[str,int],
        new_ids: List[str]
    ) -> Dict[str,int]:
        """
        Given the updated data store, plus previous loan→cluster map,
        assign the new_ids to clusters and return an updated map.
        """

    @abstractmethod
    def remove_old(
        self,
        data: StreamingDataStore,
        assignments: Dict[str,int],
        old_ids: List[str]
    ) -> Dict[str,int]:
        """
        Given the updated data store, plus previous loan→cluster map,
        remove the old_ids from the map and return an updated map.
        """

    @abstractmethod
    def rebalance(
        self,
        data: StreamingDataStore,
        assignments: Dict[str,int]
    ) -> Dict[str,int]:
        """
        If size‐drift > delta, run R offline restarts on the current data,
        return a new loan→cluster map.
        """

    @abstractmethod
    def finalize(self) -> None:
        """
        Optional cleanup.
        """
