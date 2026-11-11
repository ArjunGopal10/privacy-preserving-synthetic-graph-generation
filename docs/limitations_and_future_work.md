# Limitations, issues found, and future work

## Issues found and how they were handled
1. **Reference partition is implementation-sensitive.** Louvain depends on node/edge order and NetworkX version; running Louvain on `nx.karate_club_graph()` directly gives 4 communities (e.g. 14/10/6/4, Q≈0.43) for seed 42, while this repository's relabelled graph gives 17/13/4 (Q₀=0.3854, matching the website). *Handling:* the partition is saved (`reference_partition.json`) and can be loaded with `--partition-file`.
2. **Re-detected modularity is inflated by noise.** Louvain finds Q≈0.12 on pure-noise releases. *Handling:* Protocols A and B are reported separately and NMI/fixed-Q are the primary utility signals.
3. **NMI/ARI have no ceiling of 1 for re-detection.** *Handling:* original-graph reference values are saved and plotted.
4. **NSGA-II adds little for one variable.** The front is the monotone $\mathbb E[Q(\varepsilon)]$ curve. *Handling:* grid baseline included; no optimality claim made.
5. **Evaluation noise.** *Handling:* common random numbers by default; a fresh-noise comparison is provided.
6. **Hypervolume is not strictly monotone** across generations once the first front exceeds the population (crowding truncation). The unit test asserts only final ≥ initial.
7. **"Edge retention" equals recall** by definition; both names are kept for readability.

## Limitations
* One 34-node graph; no ego-Facebook run (dense RR is infeasible at ~4,000 nodes).
* Single-release edge-DP only; no composition-aware selection of ε; no node-DP.
* Empirical indicators depend on the adversary's prior (graph density) and are not guarantees.
* Laplace top-m is a simplified baseline; no other published DP graph generators (e.g. DER, LDPGen, dK-graph mechanisms, HRG-based) are implemented.
* Leiden/igraph are optional dependencies; Girvan–Newman is subsampled.

## Future work
* Scalable edge-DP mechanisms (sparse RR, TmF, generative models) and larger graphs.
* Data-independent or DP-accounted choice of ε; privacy accounting across the search.
* Additional decision variables so that multi-objective search is non-trivial; more objectives (degree/clustering fidelity).
* Stronger adversaries (structural re-identification, link prediction) as empirical risk indicators.
