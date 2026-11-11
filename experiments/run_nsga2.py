"""NSGA-II over epsilon: minimise [epsilon, -E[Q(H; P0)]] (figures 18-23 + extra).

Runs (1) the pymoo NSGA-II, (2) a from-scratch NSGA-II cross-check, (3) a dense grid-search baseline,
and (4) a fresh-noise run that shows the effect of NOT using common random numbers.

IMPORTANT: epsilon is the only decision variable, so this is a one-dimensional search; the dense grid
is an (almost) exact reference and NSGA-II is not expected to beat it. Selecting epsilon with the
original graph in the loop is a data-dependent step and is NOT covered by the single-release
epsilon-DP guarantee.
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
                        write_metadata)
from src.experiment_setup import load_inputs, run_cli, write_reference_partition  # noqa: E402
from src.optimization.fitness_functions import ExpectedModularityEvaluator  # noqa: E402
from src.optimization.nsga2_optimizer import (NSGA2Config, grid_search, run_nsga2,  # noqa: E402
                                              run_nsga2_scratch)
from src.optimization.pareto_analysis import (compare_with_grid, hypervolume_2d,  # noqa: E402
                                              select_solutions)
from src.visualization import convergence_plots as cv  # noqa: E402
from src.visualization import pareto_plots as pr  # noqa: E402
from src.visualization.plot_style import FigureSaver, apply_style  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--pop-size", type=int, default=40)
    p.add_argument("--n-gen", type=int, default=30)
    p.add_argument("--crossover-prob", type=float, default=0.9)
    p.add_argument("--crossover-eta", type=float, default=15.0)
    p.add_argument("--mutation-eta", type=float, default=20.0)
    p.add_argument("--eps-min", type=float, default=0.1)
    p.add_argument("--eps-max", type=float, default=10.0)
    p.add_argument("--grid-points", type=int, default=500)
    args = p.parse_args(argv)
    cfg = NSGA2Config(args.pop_size, args.n_gen, args.seed, args.crossover_prob, args.crossover_eta,
                      args.mutation_eta, args.eps_min, args.eps_max, args.n_releases, True)

    d = ensure_results_dirs(args.results_dir)
    files = {k: d["csv"] / f for k, f in {
        "front": "nsga2_pareto_front.csv", "gen": "nsga2_generation_history.csv",
        "pop": "nsga2_population_history.csv", "grid": "nsga2_grid_search.csv",
        "sel": "nsga2_selected_solutions.csv", "cmp": "nsga2_vs_grid_summary.csv",
        "scratch": "nsga2_from_scratch_front.csv", "noisy": "nsga2_fresh_noise_front.csv",
        "diag": "nsga2_pareto_diagnostics.csv"}.items()}
    files["md"] = d["tables"] / "nsga2_selected_solutions.md"
    files["meta"] = d["reports"] / "nsga2_metadata.json"
    check_overwrite(files.values(), args.overwrite)
    apply_style()

    G, P0, p0_src = load_inputs(args)
    write_reference_partition(d["reports"] / "reference_partition.json", P0, args.overwrite)
    mk = lambda crn=True: ExpectedModularityEvaluator(G, P0, cfg.n_releases, cfg.seed, crn)  # noqa: E731
    q0 = mk().original_modularity
    print(f"Q0 under P0 = {q0:.4f}; config = {cfg.to_dict()}")

    t = time.time()
    res = run_nsga2(mk(), cfg)
    print(f"pymoo NSGA-II: {res.n_evaluations} evaluations, {len(res.front)} non-dominated solutions "
          f"({time.time() - t:.2f}s)")
    scratch = run_nsga2_scratch(mk(), cfg)
    grid = grid_search(mk(), cfg.eps_min, cfg.eps_max, args.grid_points)
    stats = compare_with_grid(res.front, grid, cfg.hv_ref_point)
    stats_scratch = compare_with_grid(scratch.front, grid, cfg.hv_ref_point)
    grid_front = grid[grid["on_pareto_front"]]
    grid_hv = stats["hv_grid"]
    selected = select_solutions(res.front, q0)

    # Fresh-noise run: every evaluation draws new releases (no CRN) -> noisy objective.
    noisy = run_nsga2(mk(crn=False), cfg)
    ev_crn = mk()
    nf = noisy.front.copy()
    nf["expected_q_reeval"] = [ev_crn.expected_modularity(e) for e in nf["epsilon"]]

    # Diagnostics on the shape of the true (CRN) trade-off curve
    q_sorted = grid.sort_values("epsilon")["expected_q"].to_numpy()
    n_dips = int((np.diff(q_sorted) < -1e-12).sum())
    sat_eps = float(grid_front["epsilon"].max())
    diag = pd.DataFrame([{
        "grid_points": len(grid), "grid_points_on_pareto_front": int(grid["on_pareto_front"].sum()),
        "fraction_on_front": float(grid["on_pareto_front"].mean()),
        "largest_pareto_epsilon_on_grid": sat_eps, "non_monotone_steps_in_E[Q]_curve": n_dips,
        "nsga2_unique_nondominated": len(res.front), "nsga2_final_population": cfg.pop_size,
        "nsga2_evaluations": res.n_evaluations, "grid_evaluations": args.grid_points,
        "nsga2_hypervolume": stats["hv_nsga2"], "scratch_hypervolume": stats_scratch["hv_nsga2"],
        "fresh_noise_hypervolume_on_reeval": hypervolume_2d(
            np.column_stack([nf["epsilon"], -nf["expected_q_reeval"]]), cfg.hv_ref_point)}])

    res.front.to_csv(files["front"], index=False)
    res.generations.to_csv(files["gen"], index=False)
    res.populations.to_csv(files["pop"], index=False)
    grid.to_csv(files["grid"], index=False)
    selected.to_csv(files["sel"], index=False)
    pd.DataFrame([{"method": "pymoo_nsga2", **stats}, {"method": "from_scratch_nsga2", **stats_scratch}]
                 ).to_csv(files["cmp"], index=False)
    scratch.front.to_csv(files["scratch"], index=False)
    nf.to_csv(files["noisy"], index=False)
    diag.to_csv(files["diag"], index=False)
    files["md"].write_text("<!-- Selection compares against the ORIGINAL graph's modularity: data-dependent, "
                           "not covered by the single-release DP guarantee. -->\n"
                           + dataframe_to_markdown(selected), encoding="utf-8")

    saver = FigureSaver(d["figures"], d["figure_data"], args.overwrite)
    fig, data = pr.fig_pareto_front(res.front, q0, grid)
    saver.save(fig, "fig18_pareto_front", data)
    fig, data = cv.fig_convergence(res.generations, grid_hv)
    saver.save(fig, "fig19_generation_convergence", data)
    snaps = tuple(g for g in (1, 3, 10, cfg.n_gen) if g <= cfg.n_gen)
    fig, data = cv.fig_population_evolution(res.populations, snaps)
    saver.save(fig, "fig20_objective_space_population_evolution", data)
    fig, data = cv.fig_diversity(res.generations, res.populations)
    saver.save(fig, "fig21_crowding_distance_and_diversity", data)
    fig, data = pr.fig_nsga_vs_grid(res.front, grid, scratch.front, stats)
    saver.save(fig, "fig22_nsga2_vs_grid_search", data)
    fig, data = pr.fig_selected(res.front, selected, q0)
    saver.save(fig, "fig23_selected_solutions_tradeoff", data)
    fig, data = pr.fig_crn_vs_noisy(res.front, nf, grid, grid[["epsilon", "sd_q"]])
    saver.save(fig, "figX2_common_vs_fresh_random_numbers", data)

    (d["reports"] / "nsga2_figures_manifest.json").write_text(json.dumps(saver.manifest, indent=2))
    write_metadata(files["meta"], "experiments/run_nsga2.py", {**vars(args), "nsga2": cfg.to_dict()},
                   {"reference_partition_source": p0_src, "original_modularity_under_P0": q0,
                    "comparison_with_grid": stats, "diagnostics": diag.iloc[0].to_dict(),
                    "privacy_note": "Epsilon selection uses the original graph; not end-to-end DP."})
    print(f"HV ratio vs grid: {stats['hv_ratio']:.4f}; saved {len(saver.manifest)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli(main))
