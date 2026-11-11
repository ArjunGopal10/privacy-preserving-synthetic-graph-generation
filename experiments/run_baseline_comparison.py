"""Baseline comparison and general structure figures (figures 24-30 + baseline panels).

Methods: original, randomized response, Laplace top-m (simplified TmF-style), matched G(n,m),
degree-preserving rewiring, deletion-only. Only randomized response and the Laplace top-m filter
carry a formal edge-DP guarantee (see src/evaluation/baselines.py for the full table).

Usage:  python experiments/run_baseline_comparison.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import (base_parser, check_overwrite, dataframe_to_markdown, ensure_results_dirs,  # noqa: E402
                        parse_epsilons, write_metadata)
from src.evaluation.baselines import BASELINES, generate  # noqa: E402
from src.evaluation.comparison import compare_graphs  # noqa: E402
from src.evaluation.graph_metrics import degree_histogram_normalized  # noqa: E402
from src.experiment_setup import load_inputs, run_cli, write_reference_partition  # noqa: E402
from src.graph_generation.graph_preprocessing import graph_to_pair_vector, pair_vector_to_graph  # noqa: E402
from src.graph_generation.randomized_response import draw_uniforms, randomized_response_vectors  # noqa: E402
from src.privacy.privacy_analysis import summarize  # noqa: E402
from src.visualization import graph_visualization as gv  # noqa: E402
from src.visualization import structure_plots as sp  # noqa: E402
from src.visualization.plot_style import FigureSaver, apply_style  # noqa: E402

KEYS = {"method", "epsilon", "release"}


def pick(values: list[float], target: float) -> float:
    return min(values, key=lambda v: abs(v - target))


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    eps = parse_epsilons(args.epsilons)
    d = ensure_results_dirs(args.results_dir)
    files = {"per": d["csv"] / "baseline_per_graph.csv", "sum": d["csv"] / "baseline_summary.csv",
             "reg": d["csv"] / "baseline_registry.csv", "md": d["tables"] / "baseline_comparison.md",
             "meta": d["reports"] / "baseline_comparison_metadata.json"}
    check_overwrite(files.values(), args.overwrite)
    apply_style()

    G, P0, p0_src = load_inputs(args)
    n = G.number_of_nodes()
    write_reference_partition(d["reports"] / "reference_partition.json", P0, args.overwrite)
    vec = graph_to_pair_vector(G)
    U = draw_uniforms(len(vec), args.n_releases, args.seed)
    lo, mid, hi = pick(eps, 0.5), pick(eps, 2.0), pick(eps, 5.0)
    deg_max = n - 1
    deg_acc: dict[str, list[np.ndarray]] = {}
    rows, t0 = [], time.time()

    def record(method: str, e: float | None, k: int, H, series: str | None = None) -> None:
        rec = compare_graphs(G, H, P0, detection_seed=args.seed)
        rows.append({"method": method, "epsilon": np.nan if e is None else e, "release": k, **rec})
        if series:
            deg_acc.setdefault(series, []).append(degree_histogram_normalized(H, deg_max))

    record("original", None, 0, G, "original")
    for mi, (name, b) in enumerate(BASELINES.items()):
        if name == "original":
            continue
        grid = eps if b.uses_epsilon else [None]
        for ei, e in enumerate(grid):
            for k in range(args.n_releases):
                if name == "randomized_response":                  # coupled releases (CRN) across epsilon
                    H = pair_vector_to_graph(randomized_response_vectors(vec, e, U[k:k + 1])[0], n)
                else:
                    H = generate(name, G, np.random.default_rng([args.seed, mi, ei, k]), e)
                series = None
                if name == "randomized_response" and e in (lo, mid, hi):
                    series = f"RR ε={e:g}"
                elif name == "laplace_topm" and e == mid:
                    series = f"Laplace top-m ε={e:g}"
                elif name == "erdos_renyi_matched":
                    series = "ER"
                elif name == "degree_preserving":
                    series = "degree-preserving"
                record(name, e, k, H, series)
        print(f"  {name} done ({time.time() - t0:.1f}s)")
    per = pd.DataFrame(rows)
    value_cols = [c for c in per.columns if c not in KEYS and per[c].dtype.kind in "fiu"]
    summ = summarize(per, ["method", "epsilon"], value_cols)
    registry = pd.DataFrame([{"method": b.name, "uses_epsilon": b.uses_epsilon,
                              "formal_edge_dp": b.formal_edge_dp, "note": b.note} for b in BASELINES.values()])
    per.to_csv(files["per"], index=False)
    summ.to_csv(files["sum"], index=False)
    registry.to_csv(files["reg"], index=False)

    sel = summ[(summ["epsilon"].isna()) | (summ["epsilon"] == mid)]
    tbl = sel[["method", "epsilon", "n_releases", "precision_mean", "jaccard_mean", "modularity_fixed_mean",
               "nmi_mean", "degree_ks_mean", "stat_avg_clustering_mean", "stat_n_components_mean"]].copy()
    tbl = tbl.merge(registry[["method", "formal_edge_dp"]], on="method")
    order_m = {m: i for i, m in enumerate(BASELINES)}
    tbl = tbl.sort_values("method", key=lambda c: c.map(order_m)).reset_index(drop=True)
    tbl.columns = ["method", "epsilon", "releases", "edge precision", "Jaccard", "Q under P0", "NMI vs P0",
                   "degree KS", "clustering", "components", "formal edge-DP"]
    files["md"].write_text(f"<!-- epsilon-based methods shown at epsilon={mid:g}; means over releases. "
                           "Edge precision/Jaccard are empirical indicators, not DP metrics. -->\n"
                           + dataframe_to_markdown(tbl), encoding="utf-8")

    # ---- figures ----
    saver = FigureSaver(d["figures"], d["figure_data"], args.overwrite)
    pos = gv.fixed_layout(G, args.seed)
    fig, data = gv.fig_original(G, P0, pos)
    saver.save(fig, "fig24_original_karate_club_graph", data)
    rel = {e: pair_vector_to_graph(randomized_response_vectors(vec, e, U[:1])[0], n) for e in (lo, mid, hi)}
    fig, data = gv.fig_epsilon_panels(G, rel, P0, pos, "Randomized graphs at low, medium and high privacy budget "
                                      "(one release each; colour = fixed partition P0)")
    saver.save(fig, "fig25_randomized_graphs_low_medium_high_epsilon", data)

    dist = pd.DataFrame([{"series": s, "degree": k, "probability": float(v)}
                         for s, hs in deg_acc.items() for k, v in enumerate(np.mean(hs, axis=0))])
    order = ["original", f"RR ε={lo:g}", f"RR ε={mid:g}", f"RR ε={hi:g}", f"Laplace top-m ε={mid:g}", "ER",
             "degree-preserving"]
    dist["series"] = pd.Categorical(dist["series"], categories=order, ordered=True)
    dist = dist.sort_values(["series", "degree"]).reset_index(drop=True)
    dist["series"] = dist["series"].astype(str)
    fig, data = sp.fig_degree_distribution(dist)
    saver.save(fig, "fig26_degree_distribution_comparison", data)
    fig, data = sp.fig_stat_vs_epsilon(summ, [("stat_avg_clustering", "Average clustering coefficient")],
                                       "Clustering coefficient: original vs. releases and baselines")
    saver.save(fig, "fig27_clustering_coefficient_comparison", data)
    fig, data = sp.fig_stat_vs_epsilon(
        summ, [("stat_avg_path_length_lcc", "Avg. shortest path (largest component)"),
               ("stat_lcc_fraction", "Fraction of nodes in largest component")],
        "Path length (largest connected component) and its coverage")
    saver.save(fig, "fig28_shortest_path_length_comparison", data)
    fig, data = sp.fig_stat_vs_epsilon(summ, [("stat_density", "Graph density")],
                                       "Graph density: original vs. releases and baselines")
    saver.save(fig, "fig29_graph_density_comparison", data)
    fig, data = sp.fig_stat_vs_epsilon(summ, [("stat_n_components", "Connected components")],
                                       "Connected components: original vs. releases and baselines")
    saver.save(fig, "fig30_connected_components_comparison", data)
    fig, data = sp.fig_stat_vs_epsilon(
        summ, [("precision", "Edge precision"), ("jaccard", "Edge Jaccard")],
        "Baselines: empirical edge-reconstruction indicators (not formal DP metrics)")
    saver.save(fig, "figX3_baseline_privacy_indicators", data)
    fig, data = sp.fig_stat_vs_epsilon(
        summ, [("modularity_fixed", "Modularity under fixed P0"), ("nmi", "NMI (Louvain re-detected vs. P0)")],
        "Baselines: community-structure utility")
    saver.save(fig, "figX4_baseline_community_utility", data)
    fig, data = sp.fig_stat_vs_epsilon(summ, [("degree_ks", "Degree-distribution KS distance"),
                                              ("abs_clustering_diff", "|Δ clustering|")],
                                       "Baselines: structural distance to the original (lower = closer)")
    saver.save(fig, "figX5_baseline_structural_distance", data)

    (d["reports"] / "baseline_figures_manifest.json").write_text(json.dumps(saver.manifest, indent=2))
    write_metadata(files["meta"], "experiments/run_baseline_comparison.py",
                   {**vars(args), "epsilons": eps, "low_mid_high_eps": [lo, mid, hi]},
                   {"reference_partition_source": p0_src, "baselines": registry.to_dict("records"),
                    "caveat": "Only randomized_response and laplace_topm carry formal edge-DP guarantees."})
    print(f"Saved {len(saver.manifest)} figures; total time {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli(main))
