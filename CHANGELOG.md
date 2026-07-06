## [0.3.2] — 2026-07-06

### Reverted
- `AnnotatorPool.alpha_window` (added in 0.3.0) is **removed**. Narrowing
  co-active dimensions' alpha values together improved recovery metrics,
  but only by making the synthetic data easier to solve, not by improving
  the detection algorithm. If real annotator disagreement has one dominant
  demographic driver and a much weaker secondary one on the same text —
  entirely plausible — the generator should be able to produce that case,
  not quietly exclude it. `_sample_text_config` reverts to fully
  independent per-dimension alpha draws.
- The `intensity_range=(0.6, 1.0)` recommendation from 0.3.0 is likewise
  withdrawn; `DEFAULT_INTENSITY_RANGE=(0.3, 1.0)` remains the reference
  configuration.

### Added
- `polarized_trees.PolarizedTreesPipeline`: new `relative_h` parameter
  (default `False`). When `True`, the splitting-gain threshold `h` is
  compared against `PRG / node's own nDFU` (a fraction of remaining
  disagreement explained) instead of the raw PRG value. Fixes the same
  "absorption" problem `alpha_window` was targeting, but at the actual
  source: the paper's fixed-threshold `h` unfairly penalizes a real-but-
  weaker cause in an already-mostly-resolved subgroup, since the same raw
  PRG value means a small fraction of a large residual or a large fraction
  of a small one, and the original comparison couldn't tell those apart.

### Corpus-level impact (3-dataset baseline, unmodified generator, tree fix only)
| Dataset | Jaccard (v1, absolute h=0.05) | Jaccard (v2, relative h=0.15) |
|---|---|---|
| A — default | 0.812 | 0.863 |
| B — weak signal | 0.846 | 0.882 |
| C — deep (k=3,4 biased) | 0.820 | 0.825 |

Every metric (jaccard, precision, recall, exact match) improved or stayed
flat across all three corpora with `relative_h=True` — no regressions
observed anywhere, unlike every `min_size`-based fix attempted previously,
each of which improved one case only by taking an equal amount away from
another. `relative_h=True, h=0.15` is now the recommended configuration.

### Why this approach over 0.3.0's `alpha_window`
`alpha_window` changed *what the generator is allowed to produce*.
`relative_h` changes *how the tree judges a split* — a fix to the
detection algorithm itself, testable and shown to generalize on the
original, unmodified, harder synthetic data. This is the more honest fix:
if the method's F/C/P output is to be trusted on real data (where there's
no control over how mismatched true causes' strengths are), the
improvement needs to live in the algorithm, not in how forgiving the test
data is.

---

## [0.3.0] — 2026-07-06
**⚠️ Superseded by 0.3.1.** The `alpha_window` parameter and
`intensity_range=(0.6, 1.0)` recommendation below were reconsidered and
reverted — see 0.3.1 for why and what replaced them. Left here for
history.

### Changed
- `AnnotatorPool`: added `alpha_window` parameter (default `0.15`). For
  texts with 2+ active dimensions, alpha is now drawn from a shared
  per-text base value +/- `alpha_window`, instead of fully independent
  draws per dimension. **This changes generated data for the same seed.**
  Fixes an "absorption" failure mode where independent draws could pair
  a strong dimension with a much weaker co-active one; the tree would
  then reliably find only the strong dimension, missing the other
  entirely. Measured on 200 k=2 texts: reduced from 63% to 38% of texts
  showing this exact pattern. Not eliminated -- documented as a known,
  reduced-but-present limitation.

### Recommended defaults updated
- `intensity_range=(0.6, 1.0)` (previously `(0.3, 1.0)`) is now the
  suggested default when validating `polarized_trees`, based on direct
  testing showing it improves recovery at k=3 (jaccard 0.854→0.954) and
  k=4 (0.850→0.897) with no measurable cost elsewhere. `DEFAULT_INTENSITY_RANGE`
  itself is unchanged to avoid a silent behavior change for existing code
  relying on the old default; pass `intensity_range=(0.6, 1.0)` explicitly.

### Corpus-level impact (200-text synthetic validation, both changes combined)
| | Before | After |
|---|---|---|
| Overall jaccard | 0.811 | 0.871 |
| Overall exact match | 0.595 | 0.686 |
| k=2 jaccard | 0.612 | 0.675 |
| k=3 jaccard | 0.854 | 0.954 |
| k=4 jaccard | 0.850 | 0.897 |

## [0.2.0] — 2026-07-06

### Added
- `polartox.polarized_trees`: Polarized Trees detection algorithm (paper
  Steps 1–6), built on the collaborative `ndfu` package. `PolarizedTreesPipeline`
  covers the full pipeline: nDFU-based text filtering, per-text tree
  construction, pole assignment, and corpus-level metrics (Dimension
  Frequency, Subgroup Pole Consistency, Subgroup PRG).
- `diagnostics()`: ground-truth-free corpus statistics (retention rate,
  mean residual nDFU, indeterminate rate, etc.) — usable on real annotation
  data with no known answer to check against.
- `recovery_metrics()`: jaccard/precision/recall/exact-match against known
  ground truth — a validation harness for synthetic data only, kept
  separate from the ground-truth-free metrics above.
- `inspect_tree()`: per-text drill-down with an optional rating-distribution
  histogram at every node.
- `run_full_evaluation()`: single entry point running the whole pipeline;
  returns F/C/P and diagnostics always, plus recovery metrics and
  validation-enriched F/C/P columns (`ever_truly_active`,
  `true_lean_match_rate`, `mean_true_alpha`) when `ground_truth` is supplied.

### Design decisions (empirically validated departures from the paper)
- `min_size_frac` (default 3% of each text's annotator count) replaces a
  fixed `min_size`. The paper's `nmin=2` is a mathematical floor, not a
  reliability threshold: tested directly, `min_size=2` produced a coin-flip
  ("indeterminate") leaf in 100% of texts and 76% larger trees on average,
  versus 8.9% and a meaningfully lower jaccard-vs-ground-truth gap at
  `min_size=50` on the same corpus. A fixed count doesn't generalize across
  datasets with different annotator densities, hence the fraction-based default.
- `variant="beta"` (PRGbeta, harmonic mean of PRGmax and PRGvar) replaces
  the paper's stated primary criterion, PRGmax alone. Found a concrete case
  where PRGmax picked a spurious dimension over a real one because its
  worst-case-only design penalizes a mostly-good split for one bad subgroup;
  PRGbeta corrected this while avoiding PRGvar's tendency to produce values
  too small to clear a sensible `h`.
- `theta_stop`: a node-level pre-filter with no equivalent in the paper's
  pseudocode. Stops a node immediately if its own nDFU is already low,
  without searching for a split. Removing it on one test text increased
  the tree from 6 to 58 leaves, fragmenting on dimensions with no real
  effect on that text.

---

## [0.1.1] — 2026-07-04

### Fixed
- `GeneratedDataset.describe_text()` no longer leaks numpy string reprs (`np.str_(...)`) into printed dimension names; `ground_truth`'s `active_dims` now stores plain Python `str` instead of numpy string objects.
- `describe_text()` now explains what `alpha` means inline (`intensity: 0 = no effect, 1 = fully deterministic pole`) and formats lean/rating-count output more readably.
- `GeneratedDataset.text_ids_by_k()` now prints a truncated, readable summary (count + first N ids) instead of dumping the full list, while still returning the complete list for programmatic use.

---

## [0.1.0] — 2026-07-04

### Changed
- Full rewrite and rename from `toxpol-nlp`. This is a new baseline, not a
  continuation of `toxpol-nlp`'s version history.
- Synthetic data generation mechanism replaced entirely: severity tiers
  (High/Moderate/Low) and geometric-mean-of-weights + median threshold are
  replaced by a k-active-dimensions design, where each active dimension's
  pull is governed by a single continuous intensity parameter (`alpha`),
  and identities' rating distributions are the elementwise product of their
  active-dimension shapes.
- `bias_configs` renamed to `ground_truth`; structure changed accordingly
  (see `data_gen/README.md` migration notes).
- `generate_dataset` now returns a `GeneratedDataset` object (still unpacks
  as `(dataset, ground_truth)` for backward compatibility), adding
  `.head()`, `.tail()`, `.sample()`, `.describe_text()`, and
  `.text_ids_by_k()`.
- nDFU is no longer reimplemented in this package. `polartox` depends on
  the collaborative [`ndfu`](https://github.com/ipavlopoulos/ndfu) package
  (Pavlopoulos & Likas, 2024) directly, installable via the `[ndfu]` extra.

### Removed
- `pool.describe_bias`, `pool.analyze`, `pool.summarize`, `pool.summarize_all`.
- `toxic_range` / `civil_range` / `neutral_range` / `exclude` /
  `high_ratio` / `moderate_ratio` / `low_ratio` / `low_unimodal_share`
  parameters.

### Why
The previous weight-averaging mechanism capped achievable nDFU well below 1
— a Central Limit Theorem artifact of averaging many annotators sampled
from a large pool. The new multiplicative mechanism composes signal
instead of diluting it, reaching the full nDFU range.

---

## Prior history (`toxpol-nlp`, deprecated)
See the [`toxpol-nlp` PyPI page](https://pypi.org/project/toxpol-nlp/) for
changelog entries prior to this rewrite.