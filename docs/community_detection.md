# Community detection and community-preservation metrics

## Modularity
$Q=\sum_c[L_c/m-(d_c/2m)^2]$: observed within-community edge fraction minus what a degree-preserving random graph would give. Higher = stronger community structure for a *given* partition. **Limitation:** modularity maximisers find high-$Q$ partitions even in random graphs, so a re-detected $Q$ alone is not evidence that structure survived (see below).

## Algorithms (`src/community/community_detection.py`)
| Algorithm | Notes |
|---|---|
| Louvain | greedy modularity optimisation with node aggregation (Blondel et al., 2008); stochastic, order-dependent |
| Greedy modularity | Clauset–Newman–Moore agglomeration; deterministic |
| Label propagation | very fast, unstable; collapses to 1 community on noisy graphs |
| Girvan–Newman | removes highest-betweenness edges; best-modularity level is returned; slow on dense graphs (only 5 releases per epsilon are used) |
| Leiden | optional (`python-igraph`, `leidenalg`) |

## Two evaluation protocols (kept separate)
* **Protocol A – fixed $P_0$.** $P_0$ = Louvain on the original graph (seed 42), computed once and never re-detected. Each release is scored *under $P_0$*: modularity, coverage, performance, conductance, intra/inter edge ratio. This is the project's utility definition. NMI/ARI are meaningless here (comparing $P_0$ with itself).
* **Protocol B – re-detected.** An algorithm runs on each release; its partition is compared with $P_0$ (NMI, ARI, assignment consistency) and its own properties reported (modularity, count, sizes). Measures whether the groups are *recoverable*.

Why fixed $P_0$? It isolates the effect of the mechanism: the question "do the *same* groups still look like communities?" does not depend on detection randomness. Its weakness is that it cannot reward a release whose structure shifted to a different but equally good partition.

## Metrics
| Metric | Meaning | Better | Main caveat |
|---|---|---|---|
| Modularity | within-group edge excess | higher | resolution limit; noise-fitting when re-detected |
| NMI | information shared by two partitions | higher | biased upward when one partition has many communities (e.g. Girvan–Newman at low ε) |
| ARI | chance-corrected pair agreement | higher | can be negative; sensitive to community-size imbalance |
| Community count / sizes | granularity | closer to $P_0$ | says nothing about membership |
| Conductance | cut / min(vol) per community | lower | undefined for empty/one-community cases (skipped) |
| Coverage | fraction of edges inside communities | higher | trivially 1 for a single community |
| Performance | correctly classified pairs | higher | dominated by non-edges in sparse graphs |
| Intra/inter ratios | same as coverage / 1−coverage | – | |
| Assignment consistency | node accuracy after optimal community matching | higher | penalises splits/merges |

## Reading the results honestly
* At small $\varepsilon$ Louvain still reports $Q\approx0.12$ on pure-noise releases (`results/csv/community_redetected_summary.csv`) while NMI vs $P_0$ is ≈0.09 and $Q$ under fixed $P_0$ is ≈−0.01. Only the latter two indicate loss of the original structure.
* Re-running Louvain on the *original* graph with different seeds gives NMI ≈ 0.77 vs $P_0$ (other methods 0.63–0.69; `community_original_reference.csv`). So NMI = 1 is not an attainable ceiling for re-detection; this is drawn as a dotted line in Fig. 11/12.
* $P_0$ depends on node/edge insertion order and the NetworkX version. With this repository's graph construction and seed 42, NetworkX 3.7 gives communities of 17/13/4 and $Q_0=0.3854$ (matching the website). The partition is saved to `results/reports/reference_partition.json`, and `--partition-file` lets you pin it exactly.
