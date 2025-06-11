from __future__ import annotations

"""partition_visualizer.py

A utility module that provides the :class:`PartitionVisualizer` for exploratory
analysis and quick visual inspection of anticlustering (or any other
partitioning) results.

The class is intentionally lightweight and dependency–minimal (only requires
``pandas`` and ``matplotlib``) so that it can be copied into a thesis repository
without extra hassle.

Example
-------
>>> import pandas as pd
>>> from anticlust import anticlustering
>>> from partition_visualizer import PartitionVisualizer
>>>
>>> X = pd.read_csv("stimuli.csv")  # numeric stimulus features
>>> labels = anticlustering(X, K=3)  # or any other partitioning
>>>
>>> viz = PartitionVisualizer(X, labels)
>>> viz.summary_table()
>>> viz.plot_feature_distributions(kind="box")
>>> viz.plot_parallel_coordinates()
>>> viz.show_heatmap()

"""

from collections.abc import Sequence
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
from matplotlib.axes import Axes

import logging

_LOG = logging.getLogger(__name__)

class PartitionVisualizer:
    """Visualise feature distributions across partitions.

    Parameters
    ----------
    data : pandas.DataFrame
        A *tidy* table where each **row** represents an observation (e.g.,
        stimulus, participant) and each **column** is a numeric feature used
        during partitioning.
    labels : Sequence[int] | pandas.Series | numpy.ndarray
        Cluster/partition assignments; must be 1‑dimensional and of equal
        length as *data*.
    label_name : str, optional (default="cluster")
        Name to use for the partition column in generated tables/plots.

    Notes
    -----
    The class focuses on **descriptive** visualisations. Inferential statistics
    (e.g., ANOVA) are intentionally left out so that users can plug in their own
    preferred statistical workflow.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels: Sequence[int],
        *,
        label_name: str = "method",
    ) -> None:
        if len(data) != len(labels):
            raise ValueError("'data' and 'labels' must be of the same length.")
        self.data: pd.DataFrame = data.reset_index(drop=True)
        self.labels: pd.Series = pd.Series(labels, name=label_name).reset_index(
            drop=True
        )
        # Merge so we can use groupby convenience later
        self._df: pd.DataFrame = pd.concat([self.data, self.labels], axis=1)
        self.label_name: str = label_name

        solver_order: list[str] = ["ILP", "ILP_Precluster", "Exchange"]

    # ---------------------------------------------------------------------
    # Tabular summaries
    # ---------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        """
        Return a table with per-cluster means and (population) standard
        deviations for every feature.

        Returns
        -------
        pd.DataFrame
        """
        means = (
            self._df.groupby(self.label_name)
            .mean()
            .add_suffix("_mean")
        )
        stds = (
            self._df.groupby(self.label_name)
            .std(ddof=0)
            .add_suffix("_std")
        )
        
        summary = pd.concat([means, stds], axis=1).sort_index()
        summary.index.name = f"cluster by {self.label_name}"


        # ---- safe logging --------------------------------------------------
        #_LOG.info("Summary table (per-cluster mean ± SD):\n%s", summary.round(3).to_string())

        return summary

    # ------------------------------------------------------------------
    #  colour‑blind‑safe palette helper
    # ------------------------------------------------------------------
    @staticmethod
    def _cluster_palette(k: int) -> list[str]:
        """Return ≥ *k* discernible colours, robust for colour‑blind viewers."""
        # 1. Okabe–Ito up to 8 clusters
        okabe_ito = [
            "#E69F00", "#56B4E9", "#009E73", "#F0E442",
            "#0072B2", "#D55E00", "#CC79A7", "#000000",
        ]
        if k <= 8:
            return okabe_ito[:k]

        # 2. try Glasbey from colorcet (≥ 256 colours)
        try:
            import colorcet as cc  # type: ignore

            palette = cc.glasbey[:k]
            return palette
        except ModuleNotFoundError:
            pass

        # 3. fallback to matplotlib tab20 (20 colours)
        import matplotlib as mpl

        tab20 = mpl.colormaps.get_cmap("tab20").colors  # type: ignore[attr-defined]
        if k <= 20:
            return list(tab20[:k])

        # 4. deterministic HSV cycle if all else fails
        import colorsys
        return [
            mpl.colors.to_hex(colorsys.hsv_to_rgb(i / k, 0.65, 0.9))
            for i in range(k)
        ]
    

    def plot_scatter_with_centroids(
        self,
        *,
        figsize: tuple[int, int] = (8, 6),
        point_size: int = 75,
    ) -> plt.Figure:
        """
        2-D scatter plot coloured by partition label, with cluster centroids.

        Works only when the data matrix has exactly two numeric features.

        Parameters
        ----------
        figsize : tuple, default ``(8, 6)``
            Figure size passed to :pyfunc:`matplotlib.pyplot.subplots`.
        point_size : int, default ``50``
            Marker size for individual observations.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created Figure so you can further customise or save it.
        """
        if self.data.shape[1] != 2:
            raise ValueError(
                "Scatter plot requires exactly two features; "
                f"found {self.data.shape[1]} dims.")

        fig, ax = plt.subplots(figsize=figsize)

        df = self.data.copy()
        df[self.label_name] = self.labels
        centroids = df.groupby(self.label_name).mean()
        palette = self._cluster_palette(len(centroids))

        for idx, (label, group) in enumerate(df.groupby(self.label_name)):
            colour = palette[idx % len(palette)]

            ax.scatter(
                group.iloc[:, 0],
                group.iloc[:, 1],
                s=point_size,
                c=colour,
                edgecolor="k",
                linewidth=0.4,
                alpha=0.8,
                label=f"Cluster {label}",
            )

            cx, cy = centroids.loc[label, centroids.columns[:2]].to_numpy()
            ax.scatter(
                cx,
                cy,
                marker="X",
                s=250,
                c=colour,
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
                label=f"Centroid {label}",
            )
            ax.text(
                cx,
                cy,
                str(label),
                color="black",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
            )

        ax.set_xlabel(self.data.columns[0])
        ax.set_ylabel(self.data.columns[1])
        ax.set_title(f"2‑D Scatter Plot with Cluster Centroids (Method: {self.label_name})")

        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles, labels_, ncol=2, fontsize=8, frameon=True)
        ax.grid(True, linestyle=":", linewidth=0.5)
        fig.tight_layout()
        return fig

    def centroid_distances(self, metric: str = "euclidean") -> pd.Series:
        """
        Pairwise distance between every centroid.

        Parameters
        ----------
        metric : {'euclidean', 'manhattan'}
            Distance measure.

        Returns
        -------
        pd.Series
            Index is a tuple (cluster_i, cluster_j); value is the distance.
        """
        centroids = self._df.groupby(self.label_name).mean()
        pairs = []
        dists = []
        for i, j in combinations(centroids.index, 2):
            if metric == "euclidean":
                d = np.linalg.norm(centroids.loc[i] - centroids.loc[j])
            elif metric == "manhattan":
                d = np.abs(centroids.loc[i] - centroids.loc[j]).sum()
            else:                       # extend if you need other metrics
                raise ValueError("metric must be 'euclidean' or 'manhattan'")
            pairs.append((i, j))
            dists.append(d)
        return pd.Series(dists, index=pd.MultiIndex.from_tuples(pairs,
                           names=[f"{self.label_name}_1",
                                  f"{self.label_name}_2"]),
                         name="distance")

    # ------------------------------------------------------------------
    # STATIC helper for across-N convergence plots
    # ------------------------------------------------------------------
    @staticmethod
    def plot_convergence(
        visualisers: Sequence["PartitionVisualizer"],
        *,
        labels_for_x: Sequence[int] | None = None,
        metric: str = "euclidean",
        ax: Axes | None = None,
    ) -> Axes:
        """
        Show how average centroid distance shrinks with larger *N*.

        Parameters
        ----------
        visualisers
            A sequence of `PartitionVisualizer` objects, each fitted on a
            different sample size *N* (but same K & feature space).
        labels_for_x
            Values to put on the x-axis (e.g., the actual *N*s). Default:
            uses the order 1..len(visualisers).
        metric
            Passed to :py:meth:`centroid_distances`.
        ax
            Matplotlib axis for re-use; created if *None*.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if labels_for_x is None:
            labels_for_x = list(range(1, len(visualisers) + 1))

        avg_dist = [
            v.centroid_distances(metric=metric).mean()
            for v in visualisers
        ]

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.plot(labels_for_x, avg_dist, marker="o")
        ax.set_xlabel("Sample size $N$")
        ax.set_ylabel(f"Mean {metric} distance between centroids")
        ax.set_title("Convergence of Cluster Centroids")
        ax.grid(True, linestyle=":", linewidth=0.5)
        return ax
    
    @staticmethod
    def plot_scores_with_gaps(
        table: pd.DataFrame,
        n_clusters: int,
        *,
        log_y: bool = False,
        pct_gap: bool = False
    ) -> plt.Figure:
        """
        Two-panel figure:
            – upper: score vs N         (optionally log-scale)
            – lower: gap   vs N         (absolute or % of best score)
        """
        import matplotlib.pyplot as plt

        # ---------------------------------------------------------------
        # 1. merge the baseline scores onto every row
        # ---------------------------------------------------------------
        base = (
            table.loc[table["solver"] == "ILP", ["N", "score"]]
                .rename(columns={"score": "baseline"})
        )
        tbl = (
            table.merge(base, on="N", how="left", validate="many_to_one")
                .sort_values(["solver", "N"])
        )

        tbl["gap"] = tbl["score"] - tbl["baseline"]
        if pct_gap:
            tbl["gap"] = 100 * tbl["gap"] / tbl["baseline"]

        # ---------------------------------------------------------------
        #  enforce solver drawing order + colour map
        #     (ILP  »  ILP_Precluster  »  Exchange)
        # ---------------------------------------------------------------
        solver_order = ["ILP", "ILP_Precluster", "Exchange"]
        palette      = dict(zip(
            solver_order,
            plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(solver_order)],
        ))

        # avoid misleading gaps where no ILP baseline exists (N > max_n)
        tbl.loc[tbl["baseline"].isna(), "gap"] = np.nan


        # ------------------------------------------------------------------
        # 2. create figure --------------------------------------------------
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True,
            gridspec_kw=dict(height_ratios=[2, 1])
        )

        # ––– upper panel ––––––––––––––––––––––––––––––––––––––––––––––––
        for z, s in enumerate(reversed(solver_order), start=1):   # back → front
            if s not in tbl["solver"].unique():
                continue
            grp = tbl[tbl["solver"] == s]
            ax0.plot(
                grp["N"], grp["score"], marker="o",
                label=s, color=palette[s], zorder=z + 3
            )
        
        ax0.set_ylabel("Objective value")
        ax0.set_title(f"Anticlustering objective vs N  (K = {n_clusters})")
        if log_y:
            ax0.set_yscale("log")
        ax0.grid(alpha=.3)
        ax0.legend(title="Solver", loc="upper left")

        # ––– lower panel ––––––––––––––––––––––––––––––––––––––––––––––––
        for z, s in enumerate(reversed(solver_order), start=1):
            if s not in tbl["solver"].unique():
                continue
            grp = tbl[tbl["solver"] == s]
            ax1.plot(
                grp["N"], grp["gap"], marker="o",
                label=s, color=palette[s], zorder=z + 3
            )

        ax1.set_xlabel("Problem size N")
        ax1.set_ylabel("Gap" + (" [%]" if pct_gap else ""))
        ax1.grid(alpha=.3)

        return fig
    
    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<PartitionVisualizer: n={len(self._df)}, features={len(self.data.columns)}, "
            f"clusters={self._df[self.label_name].nunique()}>"
        )

