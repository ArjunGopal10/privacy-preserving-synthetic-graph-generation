"""Figures 18, 22, 23 (+ extra): Pareto front, NSGA-II vs grid, selected solutions."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import COLORS, eps_axis


def _front_axes(ax, original_q: float | None):
    ax.set_xlabel(r"$f_1=\varepsilon$ (privacy loss, minimise)")
    ax.set_ylabel(r"Expected modularity $\mathbb{E}[Q]$ (= $-f_2$, maximise)")
    if original_q is not None:
        ax.axhline(original_q, color=COLORS["original"], ls=":", lw=1.2, label=f"Original Q = {original_q:.3f}")


def fig_pareto_front(front: pd.DataFrame, original_q: float, grid: pd.DataFrame | None = None):
    """Figure 18: final NSGA-II Pareto front in (epsilon, E[Q]) space."""
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    if grid is not None:
        ax.plot(grid["epsilon"], grid["expected_q"], color=COLORS["grid"], lw=1.0, alpha=0.8,
                label="Dense grid (500 points)")
    ax.scatter(front["epsilon"], front["expected_q"], s=26, color=COLORS["nsga2"], zorder=3,
               label=f"NSGA-II non-dominated set (n={len(front)})")
    _front_axes(ax, original_q)
    ax.set_title("NSGA-II Pareto front: privacy loss vs. community utility")
    ax.legend(loc="lower right")
    return fig, front[["epsilon", "expected_q", "neg_expected_q"]]


def fig_nsga_vs_grid(front: pd.DataFrame, grid: pd.DataFrame, scratch: pd.DataFrame | None,
                     stats: dict[str, float]):
    """Figure 22: NSGA-II against the dense-grid baseline, plus the utility gap per solution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(grid["epsilon"], grid["expected_q"], color=COLORS["grid"], lw=1.4, label="Dense grid (500 evals)")
    ax.scatter(front["epsilon"], front["expected_q"], s=24, color=COLORS["nsga2"], zorder=3,
               label="NSGA-II / pymoo (1200 evals)")
    if scratch is not None:
        ax.scatter(scratch["epsilon"], scratch["expected_q"], s=34, marker="x", color=COLORS["scratch"],
                   zorder=4, label="NSGA-II / from-scratch (cross-check)")
    _front_axes(ax, None)
    ax.set_title("Same objective, two search strategies")
    ax.legend(loc="lower right")
    ax.text(0.03, 0.96, f"HV ratio (NSGA-II / grid) = {stats['hv_ratio']:.4f}", transform=ax.transAxes,
            va="top", fontsize=8.5)
    ax = axes[1]
    order = np.argsort(grid["epsilon"].to_numpy())
    env = np.maximum.accumulate(grid["expected_q"].to_numpy()[order])
    gap = np.interp(front["epsilon"], grid["epsilon"].to_numpy()[order], env) - front["expected_q"]
    ax.scatter(front["epsilon"], gap, s=22, color=COLORS["nsga2"])
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Grid envelope minus NSGA-II E[Q]")
    ax.set_title("Shortfall vs. grid envelope (≥ 0 = worse)")
    fig.tight_layout()
    out = front[["epsilon", "expected_q"]].copy()
    out["grid_envelope_gap"] = gap.to_numpy()
    for k, v in stats.items():
        out[f"stat_{k}"] = v
    return fig, out


def fig_selected(front: pd.DataFrame, selected: pd.DataFrame, original_q: float):
    """Figure 23: front with annotated operating points (utility thresholds and knee)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(front["epsilon"], front["expected_q"], "-", color=COLORS["nsga2"], lw=1.2, alpha=0.6)
    ax.scatter(front["epsilon"], front["expected_q"], s=14, color=COLORS["nsga2"], label="NSGA-II front")
    markers = ["o", "s", "D", "^", "v"]
    for i, r in enumerate(selected.itertuples()):
        ax.scatter([r.epsilon], [r.expected_q], s=90, marker=markers[i % 5], facecolor="none",
                   edgecolor=COLORS["selected"], linewidth=1.8, zorder=4,
                   label=rf"{r.label}: $\varepsilon$={r.epsilon:.2f}, E[Q]={r.expected_q:.3f}")
    _front_axes(ax, original_q)
    ax.set_title("Selected operating points on the privacy–utility front")
    ax.legend(loc="lower right", fontsize=7.5)
    return fig, selected


def fig_crn_vs_noisy(front_crn: pd.DataFrame, front_noisy: pd.DataFrame, grid: pd.DataFrame,
                     noisy_grid_samples: pd.DataFrame):
    """Extra: effect of evaluation noise (fresh draws per evaluation) on the observed front."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.plot(grid["epsilon"], grid["expected_q"], color=COLORS["grid"], lw=1.4, label="CRN grid (reference)")
    ax.scatter(front_crn["epsilon"], front_crn["expected_q"], s=22, color=COLORS["nsga2"], label="NSGA-II, CRN")
    ax.scatter(front_noisy["epsilon"], front_noisy["expected_q"], s=30, marker="o", facecolor="none",
               edgecolor=COLORS["selected"], label="NSGA-II, fresh noise: value the optimiser saw")
    ax.scatter(front_noisy["epsilon"], front_noisy["expected_q_reeval"], s=22, marker="x",
               color=COLORS["selected"], label="same solutions re-evaluated with CRN")
    _front_axes(ax, None)
    ax.set_title("Fronts under common vs. fresh random numbers")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(noisy_grid_samples["epsilon"], noisy_grid_samples["sd_q"], color=COLORS["selected"])
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("SD of single-release Q (across K releases)")
    ax.set_title("Per-release noise the optimiser must average out")
    fig.tight_layout()
    out = noisy_grid_samples[["epsilon", "sd_q"]].copy()
    return fig, out
