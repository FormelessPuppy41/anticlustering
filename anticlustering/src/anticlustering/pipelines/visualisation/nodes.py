from __future__ import annotations

"""visualization_nodes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kedro node utilities that generate diagnostic artefacts (CSV
summaries + Matplotlib figures) for *the first* `number_of_Ns` sample
sizes contained in a *model bank* produced by the benchmarking
pipeline.

The node is **solver‑aware** and now supports

* either a single solver name or a list/tuple of names;
* *flexible* matching modes – `exact`, `contains`, or `regex` – so the
  user can provide shorthand like `"ilp"` and still catch
  `"ILP_precluster"` or `"ILP_k4"` keys in the nested dictionary.
* a `number_of_Ns` parameter (default =`2`).

Returned artefacts are two dictionaries that plug straight into two
`PartitionedDataset` catalog entries – one for tables, one for figures –
so Kedro saves everything without the node touching the file‑system.
"""

from datetime import datetime, timezone
from typing import Dict, Sequence, Tuple, Any, List
from itertools import zip_longest
from scipy.spatial.distance import pdist
from collections import defaultdict

import logging
import re

import matplotlib.pyplot as plt
import pandas as pd

from ...core import AntiCluster
from ...visualisation import PartitionVisualizer


_LOG = logging.getLogger(__name__)

SolverLike = str | Sequence[str]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _as_list(obj: SolverLike) -> list[str]:
    """Return *obj* as a list (copy if already list/tuple, wrap if str)."""
    return [obj] if isinstance(obj, str) else list(obj)


def _match_keys(
    available: Sequence[str], patterns: Sequence[str], mode: str = "exact"
) -> list[str]:
    """Return the subset of *available* that satisfies *patterns*.

    Parameters
    ----------
    available
        The list of keys in ``model_bank[N]``.
    patterns
        User‑supplied solver names (already lower‑cased).
    mode
        * ``"exact"``    – case‑insensitive equality.
        * ``"contains"`` – substring match.
        * ``"regex"``    – full regular‑expression search.
    """
    selected: list[str] = []
    for pat in patterns:
        for key in available:
            ok = False
            if mode == "exact":
                ok = key.lower() == pat
            elif mode == "contains":
                ok = pat in key.lower()
            elif mode == "regex":
                ok = re.search(pat, key, flags=re.IGNORECASE) is not None
            if ok:
                selected.append(key)
    # preserve order / uniqueness
    return list(dict.fromkeys(selected))


def _numeric_N(n_key: int | str) -> int:
    """Convert keys like ``10`` or ``"N_10"`` to integer 10."""
    if isinstance(n_key, int):
        return n_key
    if n_key.startswith("N_"):
        return int(n_key.split("_", 1)[1])
    return int(n_key)


# -----------------------------------------------------------------------------
# Public Kedro node
# -----------------------------------------------------------------------------

def visualise_first_partitions(
    data: Dict[str, pd.DataFrame],
    model_bank: Dict[int | str, Dict[str, "AntiCluster"]],
    main_solver: SolverLike,
    number_of_Ns: int = -1,
    match_mode: str = "exact",
) -> Tuple[dict[str, pd.DataFrame], dict[str, plt.Figure]]:
    """Create summary tables + 3 plots for each *(N, solver)* pair.

    Parameters
    ----------
    data
        Mapping ``{"N_10": df, "N_12": df, …}`` (from the simulation node).
    model_bank
        Mapping ``{10: {"ilp": AntiCluster, "exchange": …}, 12: …}``.
    main_solver
        Single solver or list – *case‑insensitive*.
    number_of_Ns
        How many of the smallest *N* values to visualise. Defaults to -1, which means
        all available *N* keys in the *model_bank*.
    match_mode
        ``"exact"`` (default), ``"contains"``, or ``"regex"``.

    Returns
    -------
    tables, figures
        Two dictionaries keyed by *filename* ➜ artefact, ready for
        `PartitionedDataset` saves.
    """
    patterns = [p.lower() for p in _as_list(main_solver)]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    tables: dict[str, pd.DataFrame] = {}
    figures: dict[str, plt.Figure] = {}

    if number_of_Ns < 0:
        # If negative, use all available N keys
        number_of_Ns = len(model_bank)

    # Sort N keys numerically regardless of "N_" prefix
    sorted_Ns = sorted(model_bank, key=_numeric_N)[:number_of_Ns]

    for N_key in sorted_Ns:
        N_int = _numeric_N(N_key)
        df_key = f"N_{N_int}"

        df = data.get(df_key)
        if df is None:
            df = data.get(N_key)

        if df is None:
            _LOG.warning("Data for N=%s not found – skipped.", N_key)
            continue


        available_keys = list(model_bank[N_key].keys())
        selected = _match_keys(available_keys, patterns, match_mode)
        
        if not selected:
            _LOG.warning(
                "None of the requested solvers %s found for N=%s. Available: %s",
                patterns,
                N_key,
                available_keys,
            )
            continue

        for solver_key in selected:
            labels = model_bank[N_key][solver_key].labels_
            viz = PartitionVisualizer(df, labels, label_name=solver_key)

            # ── CSV summary ──────────────────────────────────────────────────
            tables[f"{ts}/N{N_int}/{solver_key}"] = viz.summary_table()


            # ── Figures ─────────────────────────────────────────────────────
            base_folder = f"{ts}/N{N_int}"
            figures[f"{base_folder}/scatter_{solver_key}"]  = viz.plot_scatter_with_centroids()

    return tables, figures


_NUM_RE = re.compile(r"[0-9]+")

def _numeric_N(key: str) -> int:
    """Convert 'N10' or 'N_10' → 10."""
    m = _NUM_RE.search(str(key))
    if not m:
        raise ValueError(f"Cannot parse N from key {key!r}")
    return int(m.group())


def centroid_convergence(                 # ← node entry point
    model_bank: Dict[str, Dict[str, object]],
    data: Dict[str, pd.DataFrame],     # NEW  (comes from catalog)
    main_solver: str | Sequence[str],
    *,
    match_mode: str = "exact",
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Track how the average pair-wise distance between cluster centroids
    *within a solution* shrinks as the problem size *N* grows.

    Parameters
    ----------
    model_bank
        ``{N_key -> {solver_name -> fitted_solver}}`` produced by
        `benchmark_all`.
    simulated_data
        Same keys as *model_bank* with the original feature matrices
        generated by `simulate_all_matrices`.
    main_solver
        Either a single solver name (``"ILP"``) **or** an iterable such as
        ``["ILP", "exchange"]``. Only these solvers are tracked.
    match_mode
        Forwarded to `_match_keys` – one of ``"exact" | "contains" | "regex"``.

    Returns
    -------
    table, figure
        • ``table`` – tidy DataFrame with columns
          ``["N", "solver", "mean_centroid_dist"]``  
        • ``figure`` – Matplotlib Figure (line plot).
    """
    patterns = [p.lower() for p in _as_list(main_solver)]

    rows: list[dict[str, object]] = []
    vis_by_solver: defaultdict[str, list[tuple[int, PartitionVisualizer]]] = defaultdict(list)

    # sort N keys *numerically* regardless of 'N' / 'N_' prefix
    for N_key in sorted(model_bank, key=_numeric_N):
        N_int = _numeric_N(N_key)
        df = data.get(N_key) or data.get(f"N_{N_int}")
        if df is None:
            _LOG.warning("Data for N=%s not found – skipped.", N_key)
            continue

        available = list(model_bank[N_key].keys())
        selected = _match_keys(available, patterns, match_mode)
        if not selected:
            _LOG.warning(
                "None of the requested solvers %s found for N=%s. Available: %s",
                patterns, N_key, available,
            )
            continue

        for solver_key in selected:
            labels = model_bank[N_key][solver_key].labels_
            if len(labels) != len(df):
                _LOG.warning(
                    "Length mismatch (labels=%d, rows=%d) for N=%s, solver=%s – skipped.",
                    len(labels), len(df), N_key, solver_key,
                )
                continue

            pv = PartitionVisualizer(df, labels, label_name=solver_key)
            spread = pv.centroid_distances().mean()          # average pair-wise
            rows.append(
                {"N": N_int, "solver": solver_key, "mean_centroid_dist": spread}
            )
            vis_by_solver[solver_key].append((N_int, pv))

    if not rows:
        raise RuntimeError("No matching solver/data combinations – nothing to plot.")

    table = (
        pd.DataFrame(rows)
          .sort_values(["N", "solver"])
          .reset_index(drop=True)
    )

    # ── build convergence plot ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = PartitionVisualizer._cluster_palette(table["solver"].nunique())

    for colour, (solver, pairs) in zip(palette, vis_by_solver.items()):
        pairs.sort(key=lambda t: t[0])               # sort by N
        Ns, pvs = zip(*pairs)
        PartitionVisualizer.plot_convergence(
            pvs, labels_for_x=Ns, ax=ax
        )
        line = ax.get_lines()[-1]
        line.set_label(solver)
        line.set_color(colour)
        line.set_markerfacecolor(colour)

    ax.legend(title="solver", ncol=2, fontsize=8)
    ax.set_title("Convergence of Centroid Separation")
    ax.set_xlabel("N (problem size)")
    ax.set_ylabel("Mean pair-wise centroid distance")
    fig.tight_layout()

    #print(f"Table with centroid distances:\n{table.to_string(index=False)}")
    fig.show()
    return table, fig