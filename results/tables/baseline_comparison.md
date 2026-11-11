<!-- epsilon-based methods shown at epsilon=2; means over releases. Edge precision/Jaccard are empirical indicators, not DP metrics. -->
| method | epsilon | releases | edge precision | Jaccard | Q under P0 | NMI vs P0 | degree KS | clustering | components | formal edge-DP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | nan | 1 | 1.000 | 1.000 | 0.385 | 1.000 | 0.000 | 0.571 | 1.000 | False |
| randomized_response | 2.000 | 20 | 0.550 | 0.514 | 0.175 | 0.325 | 0.529 | 0.320 | 1.000 | True |
| laplace_topm | 2.000 | 20 | 0.325 | 0.195 | 0.077 | 0.180 | 0.234 | 0.157 | 1.500 | True |
| erdos_renyi_matched | nan | 20 | 0.141 | 0.076 | -0.024 | 0.131 | 0.257 | 0.133 | 1.150 | False |
| degree_preserving | nan | 20 | 0.327 | 0.196 | -0.033 | 0.102 | 0.000 | 0.361 | 1.000 | False |
| deletion_only | 2.000 | 20 | 1.000 | 0.880 | 0.379 | 0.698 | 0.118 | 0.450 | 1.450 | False |
