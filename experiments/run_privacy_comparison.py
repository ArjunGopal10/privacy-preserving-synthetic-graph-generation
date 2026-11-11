"""Privacy-metric comparison across epsilon (figures 1-8).

Formal quantities (epsilon, flip probability, e^epsilon bound, composition) are kept separate from
empirical reconstruction indicators (precision, recall, F1, Jaccard, retention, density).

Usage:  python experiments/run_privacy_comparison.py [--epsilons 0.1,0.5,1,2,3,5,8] [--n-releases 20]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (base_parser, check_overwrite, dataframe_to_markdown, ensure_results_dirs,  # noqa: E402
                        parse_epsilons, write_metadata)
from src.experiment_setup import load_inputs, run_cli  # noqa: E402
from src.graph_generation.graph_preprocessing import graph_to_pair_vector, num_possible_pairs  # noqa: E402
from src.privacy.privacy_analysis import privacy_budget_table, sweep_privacy  # noqa: E402
from src.visualization import privacy_plots as pp  # noqa: E402
from src.visualization.plot_style import FigureSaver, apply_style  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    eps = parse_epsilons(args.epsilons)
    d = ensure_results_dirs(args.results_dir)
    outputs = [d["csv"] / "privacy_per_release.csv", d["csv"] / "privacy_summary.csv",
               d["csv"] / "privacy_budget_table.csv", d["tables"] / "privacy_summary.md",
               d["reports"] / "privacy_comparison_metadata.json"]
    check_overwrite(outputs, args.overwrite)
    apply_style()

    G, P0, p0_src = load_inputs(args)
    vec = graph_to_pair_vector(G)
    n_pairs, m = num_possible_pairs(G.number_of_nodes()), int(vec.sum())
    print(f"Graph: {G.number_of_nodes()} nodes, {m} edges, {n_pairs} possible pairs; "
          f"epsilons={eps}; releases/epsilon={args.n_releases}; seed={args.seed}")

    per_release, summary = sweep_privacy(G, eps, args.n_releases, args.seed)
    budget = privacy_budget_table(eps, n_pairs, m)
    per_release.to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    budget.to_csv(outputs[2], index=False)

    show = summary[["epsilon", "flip_probability_analytic", "precision_mean", "recall_mean", "f1_mean",
                    "jaccard_mean", "edge_count_ratio_mean", "density_mean",
                    "posterior_edge_belief_analytic"]].copy()
    show.columns = ["epsilon", "flip prob p", "precision", "recall", "F1", "Jaccard", "|E_H|/|E_G|",
                    "density", "posterior (observed edge)"]
    outputs[3].write_text(
        "<!-- Empirical means over releases; only epsilon and p are formal DP quantities. -->\n"
        + dataframe_to_markdown(show), encoding="utf-8")

    saver = FigureSaver(d["figures"], d["figure_data"], args.overwrite)
    analytic = pp.dense_analytic(n_pairs, m)
    dens0 = m / n_pairs
    for name, key in [("fig01_epsilon_vs_flip_probability", "flip_probability"),
                      ("fig02_epsilon_vs_edge_precision", "precision"),
                      ("fig03_epsilon_vs_edge_recall", "recall"),
                      ("fig04_epsilon_vs_edge_f1", "f1"),
                      ("fig05_epsilon_vs_jaccard_similarity", "jaccard"),
                      ("fig06_epsilon_vs_edge_retention", "retention")]:
        fig, data = pp.fig_metric_vs_epsilon(summary, analytic, key, dens0)
        saver.save(fig, name, data)
    fig, data = pp.fig_dashboard(summary, analytic, dens0)
    saver.save(fig, "fig07_privacy_metric_dashboard", data)
    fig, data = pp.fig_budget_comparison(budget)
    saver.save(fig, "fig08_privacy_budget_comparison", data)

    (d["reports"] / "privacy_figures_manifest.json").write_text(json.dumps(saver.manifest, indent=2))
    write_metadata(outputs[4], "experiments/run_privacy_comparison.py",
                   {**vars(args), "epsilons": eps},
                   {"reference_partition_source": p0_src, "n_nodes": G.number_of_nodes(),
                    "n_edges": m, "n_pairs": n_pairs,
                    "note": "Empirical metrics are indicators, not formal DP metrics."})
    print(f"Saved {len(saver.manifest)} figures, 3 CSVs, 1 table to {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli(main))
