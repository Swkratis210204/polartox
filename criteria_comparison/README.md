# Split-criteria comparison on DICES (informal experiment)

Does the choice of split criterion change how much polarization a Polarized-Trees-style tree explains on real data? The notebook plugs
**ten criteria** into one tree skeleton, so the criterion is the only thing that changes, and compares them on DICES-350 and DICES-990.

**The synthetic corpora are not used in this notebook.** The hyperparameters are fixed: the shared tree settings and each criterion's own
threshold come from the earlier tuning on the synthetic corpora A to J (best mean Jaccard of the planted dimensions) and are written
into section 1 of the notebook. Nothing in `polartox/` or `modules/` is modified.

## Contents

| File | What it is |
|---|---|
| `criteria_comparison.ipynb` | the notebook, saved with its outputs |
| `results/dices_results.csv` | residual nDFU in the leaves on DICES-350, DICES-990 and their mean |
| `results/dices_efficiency.csv` | explanation efficiency (explained polarization per split) on the same three groups |
| `results/ari_results.csv` | agreement between the trees of every pair of criteria on DICES |
| `results/synthetic_results.csv` | **not produced by the notebook any more**: left over from the earlier full version (synthetic recovery of the planted dimensions, which is where the thresholds came from) |

## The ten criteria

Every criterion is a function `score(node, groups)` of the rating histogram of a node and of the subgroups a split would create.

| Criterion | Score of a split |
|---|---|
| PEG harmonic, weighted, max, min, mean | the five PEG formulations of `polartox`, from the nDFU of the node and of its subgroups |
| Variance reduction | `Var(node) - sum_g w_g Var(g)` |
| JSD | size-weighted generalised Jensen-Shannon divergence between the subgroups. With size weights this equals the entropy information gain, so it is also the "entropy" criterion |
| Wasserstein | `sum_g w_g W1(subgroup g, node)` on the ordered scale |
| Gain ratio (C4.5) | information gain divided by the split information |
| KL from node | `sum_g w_g KL(subgroup g, node)` with add-one smoothing |

## Setup

- **Skeleton:** the same greedy tree as `polartox.polarized_tree.detect_polarized_subgroups` (multiway split per dimension, a dimension
  used once per branch).
- **Shared settings:** `theta_filter` 0.30, `min_size_frac` 0.02, `max_depth` 4, `theta_stop` 0.10.
- **Fixed thresholds** (from the synthetic tuning): PEG harmonic relative 0.20, PEG weighted relative 0.15, PEG max absolute 0.20,
  PEG min relative 0.40, PEG mean relative 0.40, Variance reduction relative 0.05, JSD absolute 0.02, Wasserstein relative 0.30,
  Gain ratio absolute 0.02, KL from node absolute 0.02. "Relative" means the gain divided by the node's own spread.
- **DICES:** the same preprocessing as `Dices/DICES_polarized_trees_end_to_end.ipynb`. Retained items (nDFU >= 0.30): 28 in DICES-350,
  105 in DICES-990. **Tree sizes are not matched**, so read residual nDFU together with the leaves per tree.
- **Efficiency:** per item, initial nDFU at the root, residual nDFU (annotator-weighted mean over the leaves), explained = initial -
  residual, splits = internal nodes, efficiency = explained / splits, and 0 when the tree never splits or when explained <= 0.

## Data needed

`datasets/dices_350.csv` and `datasets/dices_990.csv` (git-ignored). The overall bias question is `Q3_bias_overall` in the 350 file and
`Q3_unfair_bias_overall` in the 990 file; the notebook reads both as the same role.

## Running it

From the repository root, with an environment that has `polartox` installed (`pip install -r requirements.txt`):

    python scripts/run_notebook.py criteria_comparison/criteria_comparison.ipynb

or open the notebook and Run All. It rewrites the three CSVs in `results/`.

## The result files

`criterion` holds the names used in the notebook: `PEG harmonic`, `PEG weighted`, `PEG max`, `PEG min`, `PEG mean`,
`Variance reduction`, `JSD (= entropy gain)`, `Wasserstein`, `Gain ratio (C4.5)` and `KL from node`.

- **`dices_results.csv`**: one row per criterion and `dataset` (`DICES-350`, `DICES-990`, `both (mean of the two datasets)`):
  `threshold_form`, `threshold`, `n_items`, `leaves_per_tree`, `residual_nDFU_unweighted`, `residual_nDFU_weighted`
  (annotator-weighted, lower is better) and `rank_by_weighted_residual`.
- **`dices_efficiency.csv`**: same rows: `initial_nDFU`, `residual_nDFU`, `explained`, `splits_per_tree`, `efficiency`, `no_split_items`,
  `negative_explained_items` (items whose residual is higher than the root nDFU) and `rank_by_efficiency`. On the `both` rows the
  means are the means of the two datasets and the item counts are sums.
- **`ari_results.csv`**: long form, one row per `dataset`, `measure`, `criterion_row` and `criterion_col`. Measures: `ARI`,
  `first_split_agreement`, `node_level_agreement` and `dimension_set_jaccard`. `n_texts` is the number of polarised items.

## Results

Residual nDFU, annotator-weighted, lower is better, **tree sizes not matched**:

| Rank | Criterion | DICES-350 | DICES-990 | Both | Leaves (350 / 990) |
|---|---|---|---|---|---|
| 1 | JSD | 0.395 | 0.349 | 0.372 | 15.5 / 9.7 |
| 2 | PEG min | 0.402 | 0.347 | 0.374 | 13.2 / 9.1 |
| 3 | KL from node | 0.406 | 0.343 | 0.375 | 13.2 / 9.8 |
| 4 | Gain ratio (C4.5) | 0.449 | 0.371 | 0.410 | 10.3 / 8.9 |
| 5 | PEG weighted | 0.460 | 0.387 | 0.424 | 4.8 / 4.3 |
| 6 | PEG harmonic | 0.444 | 0.406 | 0.425 | 7.5 / 5.3 |
| 7 | Variance reduction | 0.484 | 0.373 | 0.429 | 4.9 / 7.0 |
| 8 | PEG mean | 0.514 | 0.396 | 0.455 | 6.8 / 7.7 |
| 9 | PEG max | 0.499 | 0.440 | 0.469 | 10.4 / 7.2 |
| 10 | Wasserstein | 0.527 | 0.428 | 0.477 | 3.3 / 4.9 |

Explanation efficiency (mean of the two datasets, higher is better):

| Rank | Criterion | Efficiency | Explained | Splits per tree |
|---|---|---|---|---|
| 1 | PEG weighted | 0.0288 | 0.1099 | 2.20 |
| 2 | PEG harmonic | 0.0239 | 0.1085 | 3.39 |
| 3 | PEG min | 0.0239 | 0.1595 | 6.69 |
| 4 | Variance reduction | 0.0226 | 0.1050 | 3.11 |
| 5 | KL from node | 0.0225 | 0.1592 | 6.76 |
| 6 | JSD (= entropy gain) | 0.0224 | 0.1619 | 7.66 |
| 7 | Gain ratio (C4.5) | 0.0192 | 0.1236 | 5.52 |
| 8 | PEG mean | 0.0164 | 0.0790 | 3.91 |
| 9 | Wasserstein | 0.0146 | 0.0563 | 1.96 |
| 10 | PEG max | 0.0134 | 0.0646 | 4.92 |

## Limits to keep in mind

- **Thresholds were tuned on synthetic data, not on DICES.** They are carried over unchanged. Three PEG variants (max, min, mean) chose
  the largest value of the tuning grid, so their best setting may lie beyond it.
- **Tree size.** A bigger tree leaves less polarization. The residual ranking is largely the ranking of tree size (the entropy-family
  criteria grow the biggest trees). Efficiency goes the other way: it rewards cautious criteria that split little, and PEG weighted and
  Wasserstein never split on many items. Read residual, explained, efficiency and splits together; none of them alone says which
  criterion is better.
- **Negative explained.** On some items the residual is higher than the root nDFU (up to 40 of 105 in DICES-990 for PEG max). This is
  likely because annotators with no value on the splitting dimension drop out of the leaves but are counted at the root; this has not
  been checked.
- **Residual nDFU is the quantity PEG is built to reduce,** so it is not a neutral yardstick.
- **Small samples.** DICES-350 has only 28 polarised items and DICES-990 has 105. The notebook has paired Wilcoxon tests on the residual
  (uncorrected for 45 comparisons per dataset); there are no confidence intervals in the CSV files.
