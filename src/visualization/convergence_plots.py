"""Figures 19-21: NSGA-II convergence, population evolution and diversity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import COLORS


def fig_convergence(gen: pd.DataFrame, grid_hv: float | None = None):
    """Figure 19: hypervolume, non-dominated count and best E[Q] per generation."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    ax = axes[0]
    ax.plot(gen["generation"], gen["hypervolume"], "o-", ms=3, color=COLORS["nsga2"], label="NSGA-II")
    if grid_hv is not None:
        ax.axhline(grid_hv, color=COLORS["grid"], ls="--", label="Dense-grid front")
    ax.set_ylabel("Hypervolume (ref. point (10, 0.1))")
    ax.set_title("(a) Hypervolume (higher = better)")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(gen["generation"], gen["front_size"], "o-", ms=3, color=COLORS["nsga2"])
    ax.set_ylabel("Non-dominated individuals")
    ax.set_title("(b) Rank-0 front size")
    ax = axes[2]
    ax.plot(gen["generation"], gen["best_expected_q"], "o-", ms=3, color=COLORS["nsga2"])
    ax.set_ylabel("Best E[Q] in population")
    ax.set_title("(c) Best utility found")
    for ax in axes:
        ax.set_xlabel("Generation")
    fig.tight_layout()
    return fig, gen[["generation", "n_evaluations", "hypervolume", "front_size", "best_expected_q"]]


def fig_population_evolution(pop: pd.DataFrame, generations=(1, 3, 10, 30)):
    """Figure 20: objective-space snapshots (f1 = epsilon, f2 = -E[Q]) of selected generations."""
    gens = [g for g in generations if g in set(pop["generation"])]
    fig, axes = plt.subplots(1, len(gens), figsize=(3.6 * len(gens), 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, gens):
        d = pop[pop["generation"] == g]
        sc = ax.scatter(d["epsilon"], d["neg_expected_q"], c=d["rank"], cmap="viridis_r", s=22,
                        edgecolor="white", linewidth=0.4, vmin=0, vmax=max(pop["rank"].max(), 1))
        ax.set_title(f"Generation {g}")
        ax.set_xlabel(r"$f_1=\varepsilon$")
    axes[0].set_ylabel(r"$f_2=-\mathbb{E}[Q]$ (minimise)")
    fig.suptitle("Population evolution in objective space (colour = non-dominated rank)", y=1.02)
    fig.tight_layout()
    return fig, pop[pop["generation"].isin(gens)]


def fig_diversity(gen: pd.DataFrame, pop: pd.DataFrame):
    """Figure 21: epsilon spread over generations and crowding distances of the final population."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
    ax = axes[0]
    ax.fill_between(gen["generation"], gen["min_epsilon"], gen["max_epsilon"], alpha=0.2,
                    color=COLORS["nsga2"], label="min–max of population")
    ax.plot(gen["generation"], gen["mean_epsilon"], color=COLORS["nsga2"], label="mean")
    ax2 = ax.twinx()
    ax2.plot(gen["generation"], gen["epsilon_std"], color=COLORS["laplace_topm"], ls="--", label="SD")
    ax2.set_ylabel(r"SD of $\varepsilon$", color=COLORS["laplace_topm"])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Population $\varepsilon$")
    ax.set_title("(a) Decision-space diversity")
    ax.legend(loc="lower right")
    ax = axes[1]
    last = pop[pop["generation"] == pop["generation"].max()].sort_values("epsilon")
    finite = last["crowding_distance"].replace(np.inf, np.nan)
    ax.bar(last["epsilon"], finite.fillna(0), width=0.12, color=COLORS["nsga2"])
    bound = last[~np.isfinite(last["crowding_distance"])]
    ax.scatter(bound["epsilon"], np.zeros(len(bound)), marker="^", color=COLORS["selected"], zorder=3,
               label="boundary points (distance = ∞)")
    ax.set_xlabel(r"$\varepsilon$ of final-population individual")
    ax.set_ylabel("Crowding distance")
    ax.set_title("(b) Final-population crowding distance")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = last[["individual", "epsilon", "expected_q", "rank", "crowding_distance"]].copy()
    return fig, out
