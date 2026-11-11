"""Community-structure preservation versus epsilon (figures 9-17 + extra).

Two evaluation protocols are kept SEPARATE:
  A. fixed reference partition P0 (Louvain on the original graph, computed once) scored on each release;
  B. communities RE-DETECTED on each release by several algorithms and compared with P0.

Usage:  python experiments/run_community_comparison.py [--algorithms louvain,greedy_modularity,...]
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
from src.community.community_detection import METHODS, leiden_available  # noqa: E402
from src.community.community_metrics import (fixed_partition_report, intra_inter_edges,  # noqa: E402
                                             redetected_report)
from src.experiment_setup import load_inputs, run_cli, write_reference_partition  # noqa: E402
from src.graph_generation.graph_preprocessing import graph_to_pair_vector, pair_vector_to_graph  # noqa: E402
from src.graph_generation.randomized_response import draw_uniforms, randomized_response_vectors  # noqa: E402
from src.privacy.privacy_analysis import summarize  # noqa: E402
from src.visualization import community_plots as cp  # noqa: E402
from src.visualization.graph_visualization import fig_epsilon_panels, fixed_layout  # noqa: E402
from src.visualization.plot_style import FigureSaver, apply_style  # noqa: E402

FIXED_COLS = ["modularity_fixed", "coverage_fixed", "performance_fixed", "conductance_fixed",
              "intra_edges_fixed", "inter_edges_fixed", "intra_ratio_fixed", "inter_ratio_fixed"]
RED_COLS = ["modularity_redetected", "nmi", "ari", "assignment_consistency", "n_communities",
            "largest_community", "conductance_redetected", "coverage_redetected", "performance_redetected"]


def nearest(values: list[float], targets=(0.5, 2.0, 5.0)) -> list[float]:
    """Distinct members of ``values`` closest to each target (low / medium / high epsilon)."""
    picked = []
    for t in targets:
        v = min(values, key=lambda x: abs(x - t))
        if v not in picked:
            picked.append(v)
    return picked


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--algorithms", type=str, default=",".join(METHODS),
                   help=f"Comma-separated subset of {METHODS}; leiden is skipped if not installed.")
    p.add_argument("--gn-releases", type=int, default=5,
                   help="Girvan-Newman is slow on dense noisy graphs: only the first N releases are used.")
    args = p.parse_args(argv)
    eps = parse_epsilons(args.epsilons)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in algos:
        if a not in METHODS:
            raise ValueError(f"Unknown algorithm {a!r}; choose from {METHODS}.")
    if "leiden" in algos and not leiden_available():
        print("NOTE: leiden dependencies (python-igraph, leidenalg) not installed - skipping Leiden.")
        algos.remove("leiden")

    d = ensure_results_dirs(args.results_dir)
    outputs = {k: d["csv"] / f for k, f in {
        "fixed": "community_fixed_partition_per_release.csv", "red": "community_redetected_per_release.csv",
        "fixed_s": "community_fixed_partition_summary.csv", "red_s": "community_redetected_summary.csv",
        "sizes": "community_size_distribution.csv", "orig_ref": "community_original_reference.csv"}.items()}
    outputs["md"] = d["tables"] / "community_summary.md"
    outputs["md2"] = d["tables"] / "community_algorithm_comparison.md"
    outputs["meta"] = d["reports"] / "community_comparison_metadata.json"
    check_overwrite(outputs.values(), args.overwrite)
    apply_style()

    G, P0, p0_src = load_inputs(args)
    n = G.number_of_nodes()
    vec = graph_to_pair_vector(G)
    write_reference_partition(d["reports"] / "reference_partition.json", P0, args.overwrite)
    orig_fixed = fixed_partition_report(G, P0)
    orig_intra, orig_inter = intra_inter_edges(G, P0)
    print(f"P0 from {p0_src}: sizes={sorted(map(len, P0), reverse=True)}, Q0={orig_fixed['modularity_fixed']:.4f}")

    U = draw_uniforms(len(vec), args.n_releases, args.seed)
    fixed_rows, red_rows, size_rows, example = [], [], [], {}
    t0 = time.time()
    for e in eps:
        releases = randomized_response_vectors(vec, e, U)
        for k, rel in enumerate(releases):
            H = pair_vector_to_graph(rel, n)
            if k == 0:
                example[e] = H
            fixed_rows.append({"epsilon": e, "release": k, **fixed_partition_report(H, P0)})
            for a in algos:
                if a == "girvan_newman" and k >= args.gn_releases:
                    continue
                r = redetected_report(H, P0, a, seed=args.seed)
                sizes = r.pop("sizes")
                red_rows.append({"method": a, "epsilon": e, "release": k, **r})
                if a == "louvain":
                    for rank, s in enumerate(sizes + [0] * (8 - len(sizes)), start=1):
                        if rank <= 8:
                            size_rows.append({"epsilon": e, "release": k, "rank": rank, "size": s})
        print(f"  epsilon={e:g} done ({time.time() - t0:.1f}s)")
    fixed, red = pd.DataFrame(fixed_rows), pd.DataFrame(red_rows)

    # Reference: the same algorithms re-run on the ORIGINAL graph (seeds unrelated to P0's seed).
    ref_rows = []
    for a in algos:
        seeds = [None] if a in ("greedy_modularity", "girvan_newman") else [1000 + s for s in range(args.n_releases)]
        for s in seeds:
            r = redetected_report(G, P0, a, seed=s)
            r.pop("sizes")
            ref_rows.append({"method": a, "seed": s, **r})
    orig_ref = pd.DataFrame(ref_rows)

    fixed_s = summarize(fixed, ["epsilon"], FIXED_COLS)
    red_s = summarize(red, ["method", "epsilon"], RED_COLS)
    sizes_df = pd.DataFrame(size_rows).groupby(["epsilon", "rank"]).agg(
        mean_size=("size", "mean"), sd_size=("size", "std")).reset_index().fillna(0.0)

    fixed.to_csv(outputs["fixed"], index=False)
    red.to_csv(outputs["red"], index=False)
    fixed_s.to_csv(outputs["fixed_s"], index=False)
    red_s.to_csv(outputs["red_s"], index=False)
    sizes_df.to_csv(outputs["sizes"], index=False)
    orig_ref.to_csv(outputs["orig_ref"], index=False)

    lou = red_s[red_s["method"] == "louvain"].reset_index(drop=True)
    tbl = fixed_s[["epsilon", "modularity_fixed_mean", "coverage_fixed_mean", "conductance_fixed_mean"]].merge(
        lou[["epsilon", "modularity_redetected_mean", "nmi_mean", "ari_mean", "n_communities_mean"]], on="epsilon")
    tbl.columns = ["epsilon", "Q under P0", "coverage (P0)", "conductance (P0)", "Q re-detected",
                   "NMI vs P0", "ARI vs P0", "# communities"]
    outputs["md"].write_text("<!-- Louvain; mean over releases; Protocol A columns use fixed P0, "
                             "Protocol B columns re-detect. -->\n" + dataframe_to_markdown(tbl), encoding="utf-8")
    tbl2 = red_s[red_s["epsilon"].isin(nearest(eps))][["method", "epsilon", "n_releases", "modularity_redetected_mean",
                                                      "nmi_mean", "ari_mean", "n_communities_mean"]]
    outputs["md2"].write_text(dataframe_to_markdown(tbl2), encoding="utf-8")

    # ---- figures ----
    saver = FigureSaver(d["figures"], d["figure_data"], args.overwrite)
    pos = fixed_layout(G, args.seed)
    ex_eps = nearest(eps, (2.0,))[0]
    fig, data = fig_epsilon_panels(G, {ex_eps: example[ex_eps]}, P0, pos,
                                   "Original vs. synthetic graph (one randomized release; colour = fixed partition P0)")
    saver.save(fig, "fig09_original_vs_synthetic_graph", data)
    fig, data = cp.fig_modularity(fixed_s, lou, orig_fixed["modularity_fixed"])
    saver.save(fig, "fig10_epsilon_vs_modularity", data)
    nmi_ceiling = float(orig_ref[orig_ref["method"] == "louvain"]["nmi"].mean())
    ari_ceiling = float(orig_ref[orig_ref["method"] == "louvain"]["ari"].mean())
    fig, data = cp.fig_redetected_metric(
        lou, "nmi", "NMI (re-detected vs. P0)", "NMI between re-detected and reference partition",
        nmi_ceiling, f"Louvain re-run on original (other seeds): {nmi_ceiling:.2f}")
    saver.save(fig, "fig11_epsilon_vs_nmi", data)
    fig, data = cp.fig_redetected_metric(
        lou, "ari", "ARI (re-detected vs. P0)", "Adjusted Rand Index between re-detected and reference partition",
        ari_ceiling, f"Louvain re-run on original (other seeds): {ari_ceiling:.2f}")
    saver.save(fig, "fig12_epsilon_vs_ari", data)
    fig, data = cp.fig_redetected_metric(
        lou, "n_communities", "Number of communities", "Detected community count vs. $\\varepsilon$",
        ref_line=len(P0), ref_label=f"Reference P0 ({len(P0)} communities)")
    saver.save(fig, "fig13_epsilon_vs_number_of_communities", data)
    fig, data = cp.fig_size_distribution(sizes_df[sizes_df["epsilon"].isin(nearest(eps, (0.5, 2.0, 5.0, 8.0)))],
                                         sorted(map(len, P0), reverse=True))
    saver.save(fig, "fig14_community_size_distribution", data)
    fig, data = cp.fig_conductance(fixed_s, lou, orig_fixed["conductance_fixed"])
    saver.save(fig, "fig15_conductance_comparison", data)
    fig, data = cp.fig_intra_inter(fixed_s, orig_intra, orig_inter)
    saver.save(fig, "fig16_intra_vs_inter_community_edges", data)
    fig, data = cp.fig_algorithm_comparison(red_s)
    saver.save(fig, "fig17_community_detection_algorithm_comparison", data)
    fig, data = cp.fig_partition_scores(fixed_s, lou)
    saver.save(fig, "figX1_coverage_performance_consistency", data)

    (d["reports"] / "community_figures_manifest.json").write_text(json.dumps(saver.manifest, indent=2))
    write_metadata(outputs["meta"], "experiments/run_community_comparison.py",
                   {**vars(args), "epsilons": eps, "algorithms_run": algos},
                   {"reference_partition_source": p0_src,
                    "reference_partition_sizes": sorted(map(len, P0), reverse=True),
                    "original_modularity_under_P0": orig_fixed["modularity_fixed"],
                    "protocol_note": "Protocol A = fixed P0 on release; Protocol B = re-detected, compared to P0.",
                    "louvain_nmi_ceiling_on_original_other_seeds": nmi_ceiling})
    print(f"Saved {len(saver.manifest)} figures; total time {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli(main))
