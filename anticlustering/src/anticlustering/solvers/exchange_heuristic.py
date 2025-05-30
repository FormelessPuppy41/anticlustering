
import numpy as np
from scipy.spatial.distance import squareform, pdist


from ..core.config import ExchangeConfig

import logging


class ExchangeHeuristic:
    """
    A class that implements the exchange heuristic for anticlustering.
    This heuristic iteratively exchanges elements between clusters to improve the objective function.
    """

    def __init__(
            self, 
            D : np.ndarray,
            K : int,
            config: ExchangeConfig
            ):
        self.cfg = config

        if self.cfg.random_state is not None:
            np.random.seed(self.cfg.random_state)
        else:
            np.random.seed(42)

        if self.cfg.verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    #  API
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    def solve(self, X: np.ndarray) -> np.ndarray:
        """
        Main entry: returns cluster labels (shape (n,))
        """
        rng = np.random.default_rng(self.cfg.random_state)
        n, _ = X.shape

        # 1. pre-compute pairwise distance matrix  (upper-tri condense → square)
        D_cond = pdist(X, metric=self.cfg.metric)  # shape (n*(n-1)/2,)
        D = squareform(D_cond)                     # shape (n,n)

        # 2. balanced random initial assignment
        labels = self._balanced_initialisation(n, rng)

        # 3. pre-compute intra-cluster sums
        intra_sum = self._compute_all_intra(labels, D)

        # 4. local-search by swaps
        current_obj = intra_sum.sum()
        best_obj = current_obj
        no_imp = 0
        status = "solved"

        for sweep in range(1, self.cfg.max_sweeps + 1):
            improved = False

            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = labels[i], labels[j]
                    if ci == cj:
                        continue

                    delta = self._swap_gain(i, j, ci, cj, labels, D)
                    if delta > 0:             # accept best-improving
                        # update bookkeeping
                        self._apply_swap(
                            i, j, ci, cj, labels, intra_sum, D
                        )
                        current_obj += delta
                        improved = True

            if self.cfg.verbose:
                logging.info(
                    f"Sweep {sweep:3d} | "
                    f"Objective = {current_obj:,.4f} "
                    f"{'↑' if improved else ''}"
                )

            if improved:
                best_obj = current_obj
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.cfg.patience:
                    status = "stopped"
                    if self.cfg.verbose:
                        logging.info("Early-stopping: no improvement.")
                    break
        
        return labels, best_obj, status

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    #  Helpers
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
    def _balanced_initialisation(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Returns an array of length n with exactly ⌊n/k⌋ or ⌈n/k⌉ points per cluster.
        """
        k = self.cfg.n_clusters
        base_size, remainder = divmod(n, k)
        sizes = np.array([base_size + 1] * remainder + [base_size] * (k - remainder))
        labels = np.hstack([np.full(sz, c, dtype=int) for c, sz in enumerate(sizes)])
        rng.shuffle(labels)
        return labels

    def _compute_all_intra(self, labels: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Intra-cluster sums for every cluster (upper-triangle, no double counting).
        """
        k = self.cfg.n_clusters
        intra_sum = np.zeros(k, dtype=float)
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            intra = D[np.ix_(idx, idx)]
            intra_sum[c] = np.sum(np.triu(intra, 1))
        return intra_sum

    def _swap_gain(
        self,
        i: int,
        j: int,
        ci: int,
        cj: int,
        labels: np.ndarray,
        D: np.ndarray,
    ) -> float:
        """
        Δ objective if we put i→cj, j→ci (they are currently in ci, cj).
        """
        # Members currently in ci / cj (excluding i/j)
        members_ci = np.where(labels == ci)[0]
        members_cj = np.where(labels == cj)[0]

        # remove i/j themselves for "old" contribution
        members_ci = members_ci[members_ci != i]
        members_cj = members_cj[members_cj != j]

        # OLD sums: i with its own cluster + j with its cluster
        old_sum = D[i, members_ci].sum() + D[j, members_cj].sum()

        # NEW sums after swap
        new_sum = D[i, members_cj].sum() + D[j, members_ci].sum()

        return new_sum - old_sum

    def _apply_swap(
        self,
        i: int,
        j: int,
        ci: int,
        cj: int,
        labels: np.ndarray,
        intra_sum: np.ndarray,
        D: np.ndarray,
    ):
        """
        Execute swap i↔j and update intra_sum in **O(n)**.
        """
        # contribution of i & j to their current clusters
        members_ci = np.where(labels == ci)[0]
        members_cj = np.where(labels == cj)[0]

        # remove i and j themselves
        members_ci = members_ci[members_ci != i]
        members_cj = members_cj[members_cj != j]

        # old contributions
        old_ci = D[i, members_ci].sum()
        old_cj = D[j, members_cj].sum()

        # new contributions
        new_ci = D[j, members_ci].sum()
        new_cj = D[i, members_cj].sum()

        # update sums
        intra_sum[ci] += (new_ci - old_ci)
        intra_sum[cj] += (new_cj - old_cj)

        # finally swap labels
        labels[i], labels[j] = cj, ci