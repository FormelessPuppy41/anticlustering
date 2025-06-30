# anticlustering/solvers/online/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from ...streaming.data_store import LoanStreamingDataStore
from ...core.online._config import OnlineBaseConfig

class BaseOnlineSolver(ABC):
    """
    Stateless solver interface.
    All data lives in LoanStreamingDataStore; solver only receives snapshots.
    """
    def __init__(self, config: OnlineBaseConfig) -> None:
        """
        Initialize with configuration parameters.
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def assign_new(
        self,
        data: LoanStreamingDataStore,
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
        data: LoanStreamingDataStore,
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
        data: LoanStreamingDataStore,
        assignments: Dict[str,int]
    ) -> Dict[str,int]:
        """
        If size‐drift > delta, run R offline restarts on the current data,
        return a new loan→cluster map.
        """

    @abstractmethod
    def objective_value(
        self,
        data: LoanStreamingDataStore,
        assignments: Dict[str, int]
    ) -> float:
        """
        Compute the objective value for the current assignments.
        This is optional and can be overridden by subclasses.
        """
    

    @abstractmethod
    def finalize(self) -> None:
        """
        Optional cleanup.
        """
