"""Run the full pipeline: privacy -> community -> NSGA-II -> baselines, then write a summary report.

Usage:  python experiments/run_complete_experiment.py [--overwrite] [--seed 42] [--n-releases 20]
All shared options (epsilons, seed, releases, results dir, edge list, partition file) are forwarded.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import (base_parser, check_overwrite, dataframe_to_markdown, ensure_results_dirs,  # noqa: E402
                        write_metadata)
from src.experiment_setup import run_cli  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baseline_comparison, run_community_comparison, run_nsga2, run_privacy_comparison  # noqa: E402,E401

STEPS = [("privacy comparison", run_privacy_comparison), ("community comparison", run_community_comparison),
         ("NSGA-II", run_nsga2), ("baseline comparison", run_baseline_comparison)]


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    d = ensure_results_dirs(args.results_dir)
    report = d["reports"] / "experiment_report.md"
    meta = d["reports"] / "complete_experiment_metadata.json"
    check_overwrite([report, meta], args.overwrite)

    fwd = ["--seed", str(args.seed), "--epsilons", args.epsilons, "--n-releases", str(args.n_releases),
           "--results-dir", str(args.results_dir)]
    if args.edge_list:
        fwd += ["--edge-list", str(args.edge_list)]
    if args.partition_file:
        fwd += ["--partition-file", str(args.partition_file)]
    if args.overwrite:
        fwd.append("--overwrite")
    for i, (name, mod) in enumerate(STEPS, 1):
        print(f"\n=== [{i}/{len(STEPS)}] {name} ===")
        code = mod.main(fwd)
        if code != 0:
            print(f"Step '{name}' failed with exit code {code}.", file=sys.stderr)
            return code

    parts = ["# Experiment report (auto-generated)\n",
             "All numbers below were produced by the scripts in `experiments/` with the recorded seed; "
             "see `*_metadata.json` for configuration and library versions.\n",
             "Only epsilon and the flip probability are formal differential-privacy quantities. "
             "Precision, recall, F1, Jaccard, modularity, NMI, etc. are empirical indicators.\n"]
    for title, f in [("Privacy indicators", "privacy_summary.md"), ("Community structure (Louvain)", "community_summary.md"),
                     ("Selected NSGA-II operating points", "nsga2_selected_solutions.md"),
                     ("Baselines", "baseline_comparison.md")]:
        parts.append(f"\n## {title}\n\n" + (d["tables"] / f).read_text(encoding="utf-8"))
    diag = pd.read_csv(d["csv"] / "nsga2_pareto_diagnostics.csv")
    parts.append("\n## NSGA-II diagnostics\n\n" + dataframe_to_markdown(diag.T.reset_index().rename(
        columns={"index": "quantity", 0: "value"}), "{:.4f}"))
    report.write_text("\n".join(parts), encoding="utf-8")
    write_metadata(meta, "experiments/run_complete_experiment.py", vars(args))
    print(f"\nComplete. Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli(main))
