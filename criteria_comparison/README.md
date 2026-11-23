# Split-criteria comparison (informal experiment)

Does the choice of split criterion change how well a Polarized-Trees-style tree recovers the planted dimensions, and how much
polarization it leaves in its leaves? The notebook plugs **ten criteria** into one tree skeleton, so the criterion is the only thing that
changes, and compares them on the synthetic corpora A to J and on DICES-350 / DICES-990.

Nothing in `polartox/` or `modules/` is modified: the notebook only imports from them and reads their data.

## Contents

| File | What it is |
|---|---|
| `criteria_comparison.ipynb` | the notebook, saved with its outputs |
| `results/synthetic_results.csv` | recovery of the planted dimensions, per criterion, per corpus |
| `results/dices_results.csv` | residual nDFU in the leaves on DICES-350, DICES-990 and their mean |
| `results/ari_results.csv` | agreement between the trees of every pair of criteria |

## The ten criteria

Every criterion is a function `score(node, groups)` of the rating histogram of a node and of the subgroups a split would create.

| Criterion | Score of a split |
|---|---|
| PEG harmonic, weighted, max, min, mean | the five PEG formulations of `polartox`, from the nDFU of the node and of its subgroups |
| Variance reduction | `Var(node) - sum_g w_g Var(g)` |
| JSD | size-weighted generalised Jensen-Shannon divergence between the subgroups. With size weights this equals the entropy information gain, so it is also the "entropy" criterion, not a separate family |
| Wasserstein | `sum_g w_g W1(subgroup g, node)` on the ordered scale |
| Gain ratio (C4.5) | information gain divided by the split information (the entropy of how the split divides the annotators) |
| KL from node | `sum_g w_g KL(subgroup g, node)` with add-one smoothing. Without smoothing it is exactly the information gain, i.e. the JSD above; the smoothing makes it behave differently on small subgroups |

## Setup

- **Skeleton:** the same greedy tree as `polartox.polarized_tree.detect_polarized_subgroups` (multiway split per dimension, a dimension
  used once per branch). The notebook asserts that PEG harmonic inside it, and PEG max, min and mean, reproduce the saved results of
  the 800-configuration search exactly.
- **Shared settings** (the selected configuration): `theta_filter` 0.30, `min_size_frac` 0.02, `max_depth` 4, `theta_stop` 0.10.
- **Tuning:** each criterion gets its own threshold, on the same grid of 13 settings (absolute or relative to the node's own spread,
  like `relative_h`), chosen by mean Jaccard with equal weight per corpus on A to J. A random-half check (30 splits: tune on half of the
  texts of every corpus, score on the other half) shows the tuning inflates the Jaccard by less than 0.004.
- **DICES:** the same preprocessing as `Dices/DICES_polarized_trees_end_to_end.ipynb`. Each criterion runs at the threshold tuned on the
  synthetic corpora, carried over unchanged. **Tree sizes are not matched**, so read residual nDFU together with the leaves per tree.

## Data needed

- `modules/benchmarks/benchmark_data/` (corpora A to J, already in the repository);
- `datasets/dices_350.csv` and `datasets/dices_990.csv` (git-ignored). The overall bias question is `Q3_bias_overall` in the 350 file and
  `Q3_unfair_bias_overall` in the 990 file; the notebook reads both as the same role. Without these files the DICES sections are skipped.

## Running it

From the repository root, with an environment that has `polartox` installed (`pip install -r requirements.txt`):

    python scripts/run_notebook.py criteria_comparison/criteria_comparison.ipynb

or open the notebook and Run All. It takes about 10 minutes and rewrites the three CSVs in `results/`.

## The result files

In all three files `criterion` holds the names used in the notebook: `PEG harmonic`, `PEG weighted`, `PEG max`, `PEG min`, `PEG mean`,
`Variance reduction`, `JSD (= entropy gain)`, `Wasserstein`, `Gain ratio (C4.5)` and `KL from node`.

- **`synthetic_results.csv`**: one row per criterion and `group`. `group_type` is `corpus` (the ten corpora), `mean_over_corpora`
  or `true_k` (texts pooled by the true number of active dimensions, `k=1` to `k=4`). Columns: `criterion`, `threshold_form`
  (`absolute` or `relative`), `threshold`, `n_texts`, `jaccard`, `precision`, `recall`, `exact_match`, `leaves_per_tree`,
  `dimensions_found`, and, on the `mean_over_corpora` row only, `heldout_jaccard` and `heldout_jaccard_sd`. On that row Jaccard,
  precision, recall and exact match are equal-weight means over the corpora, while leaves and dimensions found are pooled over all texts.
- **`dices_results.csv`**: one row per criterion and `dataset` (`DICES-350`, `DICES-990`, `both (mean of the two datasets)`):
  `n_items`, `leaves_per_tree`, `residual_nDFU_unweighted`, `residual_nDFU_weighted` (annotator-weighted, lower is better) and
  `rank_by_weighted_residual`.
- **`ari_results.csv`**: long form, one row per `dataset` (`synthetic`, `DICES-350`, `DICES-990`), `measure`, `criterion_row` and
  `criterion_col`. Measures: `ARI` (adjusted Rand index of the groupings of the annotators the two trees make), `first_split_agreement`,
  `node_level_agreement` and `dimension_set_jaccard`. `n_texts` is the number of polarised texts or items averaged over.

## Results

Synthetic recovery (794 polarised texts, equal weight per corpus):

| Criterion | Jaccard | Precision | Recall | Exact match | Held-out Jaccard |
|---|---|---|---|---|---|
| Variance reduction | 0.942 | 0.996 | 0.946 | 0.838 | 0.941 |
| Wasserstein | 0.937 | 0.993 | 0.938 | 0.838 | 0.935 |
| KL from node | 0.920 | 0.976 | 0.944 | 0.776 | 0.919 |
| Gain ratio (C4.5) | 0.916 | 0.985 | 0.929 | 0.782 | 0.915 |
| PEG mean | 0.899 | 0.948 | 0.940 | 0.745 | 0.899 |
| JSD | 0.893 | 0.948 | 0.945 | 0.710 | 0.890 |
| PEG harmonic | 0.889 | 0.942 | 0.931 | 0.722 | 0.887 |
| PEG weighted | 0.886 | 0.953 | 0.930 | 0.710 | 0.884 |
| PEG min | 0.872 | 0.922 | 0.950 | 0.669 | 0.872 |
| PEG max | 0.829 | 0.898 | 0.879 | 0.651 | 0.826 |

DICES residual nDFU at each criterion's tuned threshold (annotator-weighted, lower is better, **tree sizes not matched**):

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

## Limits to keep in mind

- **Edge of the tuning grid.** PEG max (absolute 0.20), PEG min and PEG mean (relative 0.40) chose the largest threshold on the grid, so
  their best setting may lie beyond it and their scores may be understated. The grid also goes further than the paper's search (`h` up
  to 0.20). The other criteria chose values inside the grid.
- **Tree size on DICES.** A bigger tree leaves less polarization. The ranking above is largely the ranking of tree size (the entropy-family
  criteria grow the biggest trees), so it is not a statement about which criterion is better at equal size.
- **Residual nDFU is the quantity PEG is built to reduce,** so it is not a neutral yardstick. Recovery of the planted dimensions on the
  synthetic corpora does not depend on any criterion.
- **Small samples.** DICES-350 has only 28 polarised items and DICES-990 has 105. No confidence intervals are reported in these files;
  the notebook's last analysis section has paired Wilcoxon tests (uncorrected for 45 comparisons per dataset).
- **The synthetic generator makes polarization a difference in mean rating between two camps,** which is exactly what variance-based
  criteria detect. A corpus where polarization changes the shape of the ratings but not their mean has not been tested.
