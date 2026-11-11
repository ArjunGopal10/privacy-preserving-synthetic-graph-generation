"""Figures 1-8: privacy indicators versus epsilon."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.privacy.privacy_metrics import analytic_rr_metrics
from .plot_style import COLORS, errorbar_line, eps_axis, nice_note

DENSE_EPS = np.linspace(0.05, 10.0, 300)


def dense_analytic(n_pairs: int, n_edges: int) -> pd.DataFrame:
    """Analytic plug-in curves on a dense epsilon grid (for the dashed reference lines)."""
    return pd.DataFrame([{"epsilon": e, **analytic_rr_metrics(n_pairs, n_edges, e)} for e in DENSE_EPS])


# metric -> (summary column prefix, analytic column, y label, title, interpretation note)
METRIC_SPECS = {
    "flip_probability": ("flip_fraction", "flip_probability", "Flip probability $p$",
                         r"Randomized-response flip probability $p=1/(1+e^{\varepsilon})$",
                         "formal parameter; lower p = weaker privacy"),
    "precision": ("precision", "precision", "Edge precision",
                  "Edge reconstruction precision vs. $\\varepsilon$",
                  "empirical indicator; higher = more disclosure"),
    "recall": ("recall", "recall", "Edge recall", "Edge reconstruction recall vs. $\\varepsilon$",
               "empirical indicator; higher = real edges survive"),
    "f1": ("f1", "f1", "Edge F1-score", "Edge reconstruction F1-score vs. $\\varepsilon$",
           "empirical indicator; higher = closer to original"),
    "jaccard": ("jaccard", "jaccard", "Jaccard similarity", "Edge-set Jaccard similarity vs. $\\varepsilon$",
                "empirical indicator; higher = more overlap"),
    "retention": ("retention", "retention", "Edge retention",
                  "Fraction of original edges retained vs. $\\varepsilon$",
                  "empirical indicator (equals recall)"),
    "edge_count_ratio": ("edge_count_ratio", "edge_count_ratio", r"$|E_H|/|E_G|$",
                         "Released-to-original edge count vs. $\\varepsilon$", "1 = same edge count"),
    "density": ("density", "density", "Graph density", "Density of released graph vs. $\\varepsilon$",
                "original density shown dashed"),
}


def _panel(ax, summary: pd.DataFrame, analytic: pd.DataFrame, key: str, original_density: float | None,
           show_note: bool = True) -> pd.DataFrame:
    col, acol, ylabel, title, note = METRIC_SPECS[key]
    ax.plot(analytic["epsilon"], analytic[acol], ls="--", color=COLORS["analytic"], lw=1.4,
            label="Analytic (plug-in)")
    errorbar_line(ax, summary["epsilon"], summary[f"{col}_mean"], summary[f"{col}_ci95"],
                  COLORS["randomized_response"], label="Empirical mean ± 95% CI")
    if key == "density" and original_density is not None:
        ax.axhline(original_density, color=COLORS["original"], ls=":", lw=1.2, label="Original")
    if key == "edge_count_ratio":
        ax.axhline(1.0, color=COLORS["original"], ls=":", lw=1.2, label="Original")
    eps_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_note:
        nice_note(ax, note, "upper left" if key in ("precision", "recall", "f1", "jaccard", "retention")
                  else "upper right")
    out = summary[["epsilon", f"{col}_mean", f"{col}_ci95"]].copy()
    out.columns = ["epsilon", "empirical_mean", "ci95_halfwidth"]
    out["analytic"] = np.interp(out["epsilon"], analytic["epsilon"], analytic[acol])
    return out


def fig_metric_vs_epsilon(summary: pd.DataFrame, analytic: pd.DataFrame, key: str,
                          original_density: float | None = None):
    """Single-metric figure (figures 1-6)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    data = _panel(ax, summary, analytic, key, original_density)
    ax.legend(loc="center right" if key != "flip_probability" else "upper right")
    return fig, data


def fig_dashboard(summary: pd.DataFrame, analytic: pd.DataFrame, original_density: float):
    """Figure 7: all indicators in one 2x4 dashboard."""
    keys = ["flip_probability", "precision", "recall", "f1", "jaccard", "retention",
            "edge_count_ratio", "density"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    frames = []
    for ax, key in zip(axes.ravel(), keys):
        d = _panel(ax, summary, analytic, key, original_density, show_note=False)
        ax.set_title(METRIC_SPECS[key][3].replace(" vs. $\\varepsilon$", ""), fontsize=9)
        if key == "edge_count_ratio":
            ax.set_yscale("log")
            ax.set_yticks([1, 1.5, 2, 3])
            ax.set_yticklabels(["1", "1.5", "2", "3"])
            ax.minorticks_off()
        d.insert(0, "metric", key)
        frames.append(d)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Privacy-metric dashboard (randomized response, Karate Club). "
                 "Only $\\varepsilon$ and $p$ are formal DP quantities; the rest are empirical indicators.",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    return fig, pd.concat(frames, ignore_index=True)


def fig_budget_comparison(budget: pd.DataFrame, releases=(1, 2, 5, 10)):
    """Figure 8: formal privacy-budget comparison across epsilon values."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    x = np.arange(len(budget))
    labels = [f"{e:g}" for e in budget["epsilon"]]
    ax = axes[0]
    ax.bar(x, budget["likelihood_ratio_bound_exp_eps"], color=COLORS["randomized_response"])
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.set_ylabel(r"DP bound $e^{\varepsilon}$ (log scale)")
    ax.set_title("(a) Max. output-probability ratio per pair")
    ax = axes[1]
    ax.plot(x, budget["posterior_belief_observed_edge"], "o-", color=COLORS["randomized_response"],
            label="Observed edge → P(real)")
    ax.plot(x, budget["posterior_belief_observed_non_edge"], "s-", color=COLORS["laplace_topm"],
            label="Observed non-edge → P(real)")
    ax.axhline(budget["prior_edge_probability"].iloc[0], color=COLORS["original"], ls=":",
               label="Prior (density)")
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.set_ylabel("Adversary posterior")
    ax.set_title("(b) Posterior belief about one pair\n(prior-dependent illustration)")
    ax.legend(loc="upper left")
    ax = axes[2]
    for k, c in zip(releases, ["#0072B2", "#009E73", "#E69F00", "#D55E00"]):
        ax.plot(x, budget[f"total_epsilon_{k}_releases"], "o-", color=c, label=f"{k} release(s)")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Per-release $\varepsilon$")
    ax.set_ylabel(r"Total $\varepsilon$ (basic composition)")
    ax.set_title("(c) Budget after repeated independent releases")
    ax.legend(loc="upper left")
    fig.suptitle("Privacy-budget comparison (formal quantities)", fontsize=11, y=1.02)
    fig.tight_layout()
    cols = ["epsilon", "flip_probability", "likelihood_ratio_bound_exp_eps", "prior_edge_probability",
            "posterior_belief_observed_edge", "posterior_belief_observed_non_edge"] + \
           [f"total_epsilon_{k}_releases" for k in releases]
    return fig, budget[cols]
