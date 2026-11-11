"""Figures 26-30 and the baseline-comparison panels: structural statistics of releases/baselines."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import COLORS, LABELS, errorbar_line, eps_axis

EPS_METHODS = ("randomized_response", "laplace_topm", "deletion_only")
STATIC_METHODS = ("original", "erdos_renyi_matched", "degree_preserving")
LINESTYLES = {"original": ":", "erdos_renyi_matched": "--", "degree_preserving": "-."}


def fig_degree_distribution(dist: pd.DataFrame):
    """Figure 26: mean degree distribution P(k) of the original and of several releases/baselines.
    ``dist`` columns: series, degree, probability."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    palette = {"original": COLORS["original"], "ER": COLORS["erdos_renyi_matched"],
               "degree-preserving": COLORS["degree_preserving"]}
    rr_cols = ["#9ECAE1", "#4292C6", "#08306B"]
    series = list(dist["series"].unique())
    rr_series = [s for s in series if s.startswith("RR")]
    for ax, cum in zip(axes, (False, True)):
        for s in series:
            d = dist[dist["series"] == s].sort_values("degree")
            y = d["probability"].to_numpy()
            y = np.cumsum(y[::-1])[::-1] if cum else y   # CCDF: P(K >= k)
            if cum:
                y = np.where(y > 1e-9, y, np.nan)
            if s in palette:
                kw = dict(color=palette[s], lw=2.2 if s == "original" else 1.5,
                          ls="-" if s == "original" else ("--" if s == "ER" else ":"))
            elif s.startswith("RR"):
                kw = dict(color=rr_cols[rr_series.index(s) % 3], lw=1.6)
            else:
                kw = dict(color=COLORS["laplace_topm"], lw=1.4, ls="--")
            ax.step(d["degree"], y, where="mid", label=s, **kw)
        ax.set_xlabel("Degree k")
        ax.set_ylabel("P(K ≥ k)" if cum else "P(K = k)")
        ax.set_title("(b) Complementary cumulative distribution" if cum else "(a) Degree distribution")
        if cum:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.5)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Degree distribution: original vs. randomized releases and baselines "
                 "(mean over releases)", y=1.02)
    fig.tight_layout()
    return fig, dist


def fig_stat_vs_epsilon(summary: pd.DataFrame, stats: list[tuple[str, str]], title: str,
                        figsize: tuple[float, float] | None = None, log_y: bool = False):
    """One panel per ``(column, ylabel)``. ``summary`` has ``method``, ``epsilon`` (NaN for methods
    without epsilon) and ``<column>_mean`` / ``<column>_ci95``. Epsilon-free methods are drawn as
    horizontal lines (with a CI band)."""
    n = len(stats)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5.2 * n, 3.9), squeeze=False)
    eps_vals = summary.loc[summary["method"] == "randomized_response", "epsilon"].to_numpy()
    for ax, (col, ylabel) in zip(axes[0], stats):
        for method in EPS_METHODS:
            d = summary[summary["method"] == method].sort_values("epsilon")
            if not d.empty:
                errorbar_line(ax, d["epsilon"], d[f"{col}_mean"], d[f"{col}_ci95"], COLORS[method],
                              LABELS[method], marker="o", ms=4)
        for method in STATIC_METHODS:
            d = summary[summary["method"] == method]
            if d.empty:
                continue
            m, c = float(d[f"{col}_mean"].iloc[0]), float(d[f"{col}_ci95"].iloc[0])
            ax.axhline(m, color=COLORS[method], ls=LINESTYLES[method], lw=1.5, label=LABELS[method])
            if c > 0 and method != "original":
                ax.axhspan(m - c, m + c, color=COLORS[method], alpha=0.12, lw=0)
        eps_axis(ax)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
    axes[0][0].legend(fontsize=7.5, loc="best")
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    cols = ["method", "epsilon", "n_releases"] + [f"{c}_{s}" for c, _ in stats for s in ("mean", "ci95")]
    return fig, summary[cols]
