"""Epsilon sweeps: empirical vs analytic privacy indicators, plus the privacy-budget table."""
from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from src.graph_generation.graph_preprocessing import graph_to_pair_vector, num_possible_pairs
from src.graph_generation.randomized_response import (
    draw_uniforms, flip_probability, randomized_response_vectors)
from .differential_privacy import (
    basic_composition, likelihood_ratio_bound, max_log_likelihood_ratio, posterior_edge_belief)
from .privacy_metrics import all_edge_metrics, analytic_rr_metrics

METRICS = ("flip_fraction", "precision", "recall", "f1", "jaccard", "retention",
           "edge_count_ratio", "density", "edges_added", "edges_deleted", "n_edges")


def mean_sd_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean, sample SD and half-width of the t-based confidence interval."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]                      # undefined values (e.g. empty-graph conductance) are skipped
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if x.size < 2:
        return mean, 0.0, 0.0
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2, x.size - 1) * sd / np.sqrt(x.size))
    return mean, sd, half


def summarize(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    """Per-group mean / sd / 95% CI half-width columns ``<col>_mean``, ``<col>_sd``, ``<col>_ci95``."""
    rows = []
    for key, grp in df.groupby(list(group_cols), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["n_releases"] = len(grp)
        for col in value_cols:
            m, s, c = mean_sd_ci(grp[col].to_numpy())
            row[f"{col}_mean"], row[f"{col}_sd"], row[f"{col}_ci95"] = m, s, c
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_privacy(G: nx.Graph, epsilons: Sequence[float], n_releases: int, seed: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run RR at each epsilon with common random numbers; return (per-release, summary) frames.

    The summary contains empirical means/SD/CI plus analytic plug-in values for comparison.
    """
    vec = graph_to_pair_vector(G)
    n_pairs, m = num_possible_pairs(G.number_of_nodes()), int(vec.sum())
    U = draw_uniforms(n_pairs, n_releases, seed)
    rows = []
    for eps in epsilons:
        releases = randomized_response_vectors(vec, eps, U)
        for k, rel in enumerate(releases):
            rows.append({"epsilon": eps, "release": k, **all_edge_metrics(vec, rel)})
    per_release = pd.DataFrame(rows)
    summary = summarize(per_release, ["epsilon"], list(METRICS))
    prior = m / n_pairs
    ana = pd.DataFrame([{"epsilon": e, **{f"{k}_analytic": v for k, v in
                                          analytic_rr_metrics(n_pairs, m, e).items()},
                         "posterior_edge_belief_analytic": posterior_edge_belief(prior, e, True)}
                        for e in summary["epsilon"]])
    return per_release, summary.merge(ana, on="epsilon")


def privacy_budget_table(epsilons: Sequence[float], n_pairs: int, n_edges: int,
                         releases: Sequence[int] = (1, 2, 5, 10)) -> pd.DataFrame:
    """Formal privacy-budget view: flip probability, e^eps bound, composition, posterior belief."""
    prior = n_edges / n_pairs
    rows = []
    for e in epsilons:
        row = {"epsilon": e, "flip_probability": flip_probability(e),
               "likelihood_ratio_bound_exp_eps": likelihood_ratio_bound(e),
               "verified_max_log_likelihood_ratio": max_log_likelihood_ratio(e) if e > 0 else 0.0,
               "prior_edge_probability": prior,
               "posterior_belief_observed_edge": posterior_edge_belief(prior, e, True),
               "posterior_belief_observed_non_edge": posterior_edge_belief(prior, e, False)}
        for k in releases:
            row[f"total_epsilon_{k}_releases"] = basic_composition([e] * k)
        rows.append(row)
    return pd.DataFrame(rows)
