"""
High-level interface: select solver by name and run it.
"""
import numpy as np
from .solver_factory import get_solver

class AntiCluster:
    """
    Orchestrator that picks a solver at runtime.

    Example
    -------
    >>> ac = AntiCluster('kmeans', n_groups=4, max_iter=5, random_state=42)
    >>> labels = ac.fit_predict(X)
    """
    def __init__(
        self,
        solver_name: str,
        n_groups: int,
        max_iter: int = 1,
        random_state: int = None
    ):
        self.solver = get_solver(
            solver_name,
            n_groups=n_groups,
            max_iter=max_iter,
            random_state=random_state
        )

    def fit(self, X: np.ndarray) -> 'AntiCluster':
        """
        Fit the chosen solver on data X.
        """
        self.solver.fit(X)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and return group labels.
        """
        return self.solver.fit_transform(X)

    def score(self, X: np.ndarray) -> float:
        """
        Return objective score of the fitted labels on X.
        """
        return self.solver.score(X)
