"""Figures 10-17: community-structure preservation versus epsilon."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import COLORS, LABELS, errorbar_line, eps_axis, nice_note


def fig_modularity(fixed: pd.DataFrame, red: pd.DataFrame, q_original: float):
    """Figure 10: modularity under fixed P0 vs modularity of the re-detected partition."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    errorbar_line(ax, fixed["epsilon"], fixed["modularity_fixed_mean"], fixed["modularity_fixed_ci95"],
                  COLORS["fixed"], "Fixed partition P0 (Protocol A)")
    errorbar_line(ax, red["epsilon"], red["modularity_redetected_mean"], red["modularity_redetected_ci95"],
                  COLORS["redetected"], "Louvain re-detected on release (Protocol B)", marker="s", ls="--")
    ax.axhline(q_original, color=COLORS["original"], ls=":", label=f"Original Q = {q_original:.3f}")
    ax.axhline(0, color="#999999", lw=0.8)
    eps_axis(ax)
    ax.set_ylabel("Modularity Q")
    ax.set_title("Modularity of released graphs vs. $\\varepsilon$")
    ax.legend(loc="upper left")
    out = fixed[["epsilon", "modularity_fixed_mean", "modularity_fixed_ci95"]].merge(
        red[["epsilon", "modularity_redetected_mean", "modularity_redetected_ci95"]], on="epsilon")
    out["original_q"] = q_original
    return fig, out


def fig_redetected_metric(red: pd.DataFrame, metric: str, ylabel: str, title: str,
                          ceiling: float | None = None, ceiling_label: str = "",
                          ref_line: float | None = None, ref_label: str = ""):
    """Figures 11-13: NMI / ARI / community count of the re-detected (Louvain) partition."""
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    errorbar_line(ax, red["epsilon"], red[f"{metric}_mean"], red[f"{metric}_ci95"], COLORS["redetected"],
                  "Louvain on release")
    if ceiling is not None:
        ax.axhline(ceiling, color=COLORS["original"], ls=":", label=ceiling_label)
    if ref_line is not None:
        ax.axhline(ref_line, color=COLORS["original"], ls=":", label=ref_label)
    eps_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    out = red[["epsilon", f"{metric}_mean", f"{metric}_sd", f"{metric}_ci95"]].copy()
    return fig, out


def fig_size_distribution(size_df: pd.DataFrame, p0_sizes: list[int]):
    """Figure 14: mean sorted community sizes (rank 1 = largest) at selected epsilons."""
    eps_list = sorted(size_df["epsilon"].unique())
    max_rank = int(size_df["rank"].max())
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    width = 0.8 / (len(eps_list) + 1)
    ranks = np.arange(1, max_rank + 1)
    p0 = np.array(p0_sizes + [0] * (max_rank - len(p0_sizes)))[:max_rank]
    ax.bar(ranks - 0.4 + width / 2, p0, width, color=COLORS["original"], label="Original P0")
    cmap = plt.get_cmap("viridis")
    for i, eps in enumerate(eps_list):
        d = size_df[size_df["epsilon"] == eps].sort_values("rank")
        ax.bar(d["rank"] - 0.4 + width * (i + 1.5), d["mean_size"], width, yerr=d["sd_size"], capsize=1.5,
               color=cmap(i / max(len(eps_list) - 1, 1)), label=rf"$\varepsilon$={eps:g}",
               error_kw={"elinewidth": 0.7})
    ax.set_xticks(ranks)
    ax.set_xlabel("Community rank (1 = largest)")
    ax.set_ylabel("Mean community size (nodes)")
    ax.set_title("Community-size distribution of Louvain partitions on releases (mean ± SD)")
    ax.legend(ncol=2, loc="upper right")
    return fig, size_df


def fig_conductance(fixed: pd.DataFrame, red: pd.DataFrame, orig_fixed: float):
    """Figure 15: mean conductance (lower = better separated communities)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    errorbar_line(ax, fixed["epsilon"], fixed["conductance_fixed_mean"], fixed["conductance_fixed_ci95"],
                  COLORS["fixed"], "Fixed P0 on release")
    errorbar_line(ax, red["epsilon"], red["conductance_redetected_mean"], red["conductance_redetected_ci95"],
                  COLORS["redetected"], "Louvain re-detected on release", marker="s", ls="--")
    ax.axhline(orig_fixed, color=COLORS["original"], ls=":", label=f"Original (P0) = {orig_fixed:.3f}")
    eps_axis(ax)
    ax.set_ylabel("Mean conductance")
    ax.set_title("Conductance of communities vs. $\\varepsilon$")
    nice_note(ax, "lower = better-separated communities", "lower right")
    ax.legend(loc="upper right")
    out = fixed[["epsilon", "conductance_fixed_mean", "conductance_fixed_ci95"]].merge(
        red[["epsilon", "conductance_redetected_mean", "conductance_redetected_ci95"]], on="epsilon")
    out["original_conductance_fixed"] = orig_fixed
    return fig, out


def fig_intra_inter(fixed: pd.DataFrame, orig_intra: int, orig_inter: int):
    """Figure 16: mean intra- vs inter-community edge counts under fixed P0 (stacked)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    labels = ["Orig."] + [f"{e:g}" for e in fixed["epsilon"]]
    x = np.arange(len(labels))
    intra = np.r_[orig_intra, fixed["intra_edges_fixed_mean"]]
    inter = np.r_[orig_inter, fixed["inter_edges_fixed_mean"]]
    ax = axes[0]
    ax.bar(x, intra, color="#0072B2", label="Intra-community edges")
    ax.bar(x, inter, bottom=intra, color="#D55E00", label="Inter-community edges")
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.set_ylabel("Mean number of edges")
    ax.set_title("(a) Edge counts under fixed P0")
    ax.legend(loc="upper right")
    ax = axes[1]
    errorbar_line(ax, fixed["epsilon"], fixed["intra_ratio_fixed_mean"], fixed["intra_ratio_fixed_ci95"],
                  "#0072B2", "Intra-community ratio")
    errorbar_line(ax, fixed["epsilon"], fixed["inter_ratio_fixed_mean"], fixed["inter_ratio_fixed_ci95"],
                  "#D55E00", "Inter-community ratio", marker="s")
    ax.axhline(orig_intra / (orig_intra + orig_inter), color="#0072B2", ls=":", lw=1)
    eps_axis(ax)
    ax.set_ylabel("Fraction of edges")
    ax.set_title("(b) Ratios (dotted = original intra ratio)")
    ax.legend(loc="center right")
    fig.tight_layout()
    out = fixed[["epsilon", "intra_edges_fixed_mean", "inter_edges_fixed_mean", "intra_ratio_fixed_mean",
                 "inter_ratio_fixed_mean", "intra_ratio_fixed_ci95", "inter_ratio_fixed_ci95"]].copy()
    out["original_intra_edges"], out["original_inter_edges"] = orig_intra, orig_inter
    return fig, out


def fig_algorithm_comparison(alg: pd.DataFrame):
    """Figure 17: detection algorithms compared on re-detected metrics (summary has `method` col)."""
    spec = [("modularity_redetected", "Modularity of detected partition"),
            ("nmi", "NMI vs. reference P0"), ("ari", "ARI vs. reference P0"),
            ("n_communities", "Number of communities")]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8))
    for ax, (m, ylabel) in zip(axes.ravel(), spec):
        for method, d in alg.groupby("method"):
            errorbar_line(ax, d["epsilon"], d[f"{m}_mean"], d[f"{m}_ci95"], COLORS[method],
                          LABELS[method], marker="o", ms=4)
        eps_axis(ax)
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Community-detection algorithm comparison on randomized releases "
                 "(Protocol B; mean ± 95% CI)", fontsize=11, y=1.0)
    fig.tight_layout()
    cols = ["method", "epsilon", "n_releases"] + [f"{m}_{s}" for m, _ in spec for s in ("mean", "ci95")]
    return fig, alg[cols]


def fig_partition_scores(fixed: pd.DataFrame, red: pd.DataFrame):
    """Extra figure: coverage, performance (fixed P0) and assignment consistency (re-detected)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    for ax, (df, col, ttl) in zip(axes, [(fixed, "coverage_fixed", "Coverage (fixed P0)"),
                                         (fixed, "performance_fixed", "Performance (fixed P0)"),
                                         (red, "assignment_consistency",
                                          "Assignment consistency (Louvain vs. P0)")]):
        errorbar_line(ax, df["epsilon"], df[f"{col}_mean"], df[f"{col}_ci95"], COLORS["fixed"])
        eps_axis(ax)
        ax.set_ylabel(ttl)
        ax.set_title(ttl)
    fig.tight_layout()
    out = fixed[["epsilon", "coverage_fixed_mean", "coverage_fixed_ci95", "performance_fixed_mean",
                 "performance_fixed_ci95"]].merge(
        red[["epsilon", "assignment_consistency_mean", "assignment_consistency_ci95"]], on="epsilon")
    return fig, out
