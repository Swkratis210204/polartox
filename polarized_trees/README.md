# `polartox.polarized_trees` — Polarized Trees Detection

Given an annotation dataset, recursively partitions annotators by demographic dimension to find the specific subgroups that are most polarized on a given text — a diagnostic tool for understanding which dimensions drive disagreement, and an attribution framework for pinpointing intersectional subgroups whose opinions diverge most strongly.

Implements the paper's full six-step pipeline: compute nDFU per text, filter to meaningfully polarized texts, build one tree per text via recursive PRG-based splitting, assign pole labels to leaves, and aggregate dataset-level metrics (Dimension Frequency, Subgroup Pole Consistency, Subgroup PRG) across the whole corpus.

![Example polarized tree](polarized_tree.png)

A basic example of the method in action: one text's annotators recursively split by demographic dimension until each leaf is a subgroup consistently leaning toward one pole.

## Install

```bash
pip install polartox
```

(`ndfu` is a core dependency — nDFU scoring is provided by the collaborative [`ndfu`](https://github.com/ipavlopoulos/ndfu) package, Pavlopoulos & Likas 2024, not reimplemented here.)

## Quickstart

```python
from polartox.polarized_trees import PolarizedTreesPipeline

pipe = PolarizedTreesPipeline(
    dims=["gender", "politics", "age", "education", "orientation"],
    scale=5,
    theta_filter=0.3,      # Step 2 -- user-defined; how polarized a text must be to analyze at all
    min_size_frac=0.03,    # min subgroup size, as a fraction of each text's annotators
    h=0.15,                # PRG gain threshold -- interpreted as a FRACTION when relative_h=True (see below)
    max_depth=6,
    relative_h=True,       # recommended -- see "Departures from the paper" below
)

# dataset: one row per (text_id, annotator_id), columns text_id, rating, <dims>
results = pipe.run_full_evaluation(dataset)

results["F"]             # dimension frequency by depth
results["C"]             # subgroup pole consistency
results["P"]             # subgroup PRG
results["diagnostics"]   # corpus-level sanity stats, no ground truth needed
```

With known ground truth (synthetic data only), pass it in to additionally get recovery metrics and validation-enriched F/C/P:

```python
results = pipe.run_full_evaluation(dataset, ground_truth=ground_truth)
results["recovery"]   # per-text jaccard / precision / recall / exact_match
```

## How it works (paper Steps 1–6)

1. **Compute nDFU** for every text's full annotation multiset.
2. **Filter** to texts whose nDFU ≥ `theta_filter` — only meaningfully polarized texts proceed.
3. **Set parameters, choose PRG variant.** At each node, every remaining candidate dimension is scored by Polarization Reduction Gain; the best one splits the node, then is removed from consideration for all descendants on that branch.
4. **Construct one tree per text**, recursing until `min_size`, `h`, `max_depth`, or dimension exhaustion stops a branch.
5. **Assign pole labels** to leaves: `toxic` if the fraction of ratings ≥ `theta_pole` exceeds 0.5, `civil` if below, `indeterminate` at exactly 0.5.
6. **Aggregate dataset-level metrics** across every tree: Dimension Frequency (F), Subgroup Pole Consistency (C), Subgroup PRG (P).

## Departures from the paper (empirically validated, not arbitrary)

Four design choices in this implementation diverge from the paper's literal specification. Each was tested directly against the alternative on synthetic ground-truth data before being adopted as the default.

### `relative_h`: PRG measured as a fraction of remaining disagreement, not an absolute amount

The paper's gain threshold `h` compares PRG directly to a fixed number — but a PRG of 0.05 means something very different depending on how much disagreement is left at that node. In a node with 0.8 nDFU remaining, 0.05 is negligible; in a node with only 0.1 remaining, that same 0.05 explains half of what's left. Comparing both cases to the same fixed `h` unfairly penalizes real-but-weaker causes in already-mostly-resolved subgroups — this was found to be the dominant reason texts with two active dimensions of unequal strength ("k=2") often lost the weaker dimension entirely.

With `relative_h=True`, the comparison becomes `(best PRG / node's own nDFU) <= h`, so `h` is interpreted as a **fraction of remaining disagreement explained**, not a raw amount. Tested on identical, unmodified synthetic data (no generator changes): switching from absolute `h=0.05` to relative `h=0.15` improved every corpus-level metric on every one of three differently-configured synthetic corpora, with no regressions anywhere (see the Baseline table below). This is unlike every `min_size`-based adjustment tried beforehand, each of which only improved weak-signal texts by taking an equal amount away from texts with more/stronger true causes. Set `relative_h=False` (default, for backward compatibility) to use the paper's literal absolute comparison.

### `min_size_frac` instead of a fixed `min_size`

The paper's `nmin=2` is a mathematical floor (below 2 annotations, nDFU is undefined) — not a recommendation for reliable output. Testing `min_size=2` directly against `min_size=50` (on a 1,620-annotator pool) found:

| | `min_size=2` | `min_size=50` |
|---|---|---|
| Texts with ≥1 indeterminate (coin-flip) leaf | 100% | 8.9% |
| Mean leaves per tree | 12.0 | 6.8 |
| Mean jaccard vs. ground truth | 0.804 | 0.887 |

`min_size=2` reliably fragments trees on pure sampling noise. Since a fixed absolute count means something completely different depending on how many annotators a dataset has per text (50 is ~3% of 1,620, but ~33% of 150 — unusable), subgroup size is expressed as `min_size_frac`, a fraction of *that specific text's* annotator count, and resolved to an absolute count internally. Pass `min_size` instead for a fixed absolute value if you prefer.

*Note: depth-dependent `min_size_frac` schedules (tightening the threshold only at deeper levels) were also tested as an alternative fix for the weak-dimension problem above. Every schedule tried improved weak-signal texts only by taking an equal amount away from texts with more/stronger true causes — the noise-driven splits in under-signaled texts and the legitimate 3rd/4th-cause splits in richer texts occur at the same depth and subgroup size, so no depth- or size-based threshold can separate them. `relative_h` was adopted instead because it fixes the actual comparison being made, not just where it's applied.*

### PRGbeta as the default splitting criterion, not PRGmax

The paper names PRGmax "the primary splitting criterion." Direct testing found a concrete case where PRGmax's worst-case-only design picked a spurious dimension over a genuine one: a real cause with 2 clean subgroups and 1 imperfect one (PRGmax penalized by the one bad subgroup) lost to a fake cause that merely had one large, coincidental swing (PRGmax = 0.055 for the real cause vs. 0.156 for the fake one). PRGvar (size-weighted average) correctly ranked the real cause far higher (0.331 vs. 0.010) in the same case, but alone tends to produce values too small to clear a sensible `h`. PRGbeta (harmonic mean of both, the paper's own proposed combination) correctly ranked the true cause first (0.093 vs. 0.019) while producing usable magnitudes — hence the default.

### `theta_stop`: a node-level pre-filter not present in the paper

If a node's own nDFU is already below `theta_stop`, the tree stops immediately without searching for a split at all — distinct from `h`, which asks whether a *found* split is good enough to use. Without this check, a node that is already essentially resolved can still fragment further on pure noise: removing it from an otherwise-identical run increased one text's tree from 6 leaves to 58, with the extra branches splitting on demographic dimensions that had no real effect on that text. Set `theta_stop=None` to disable and match the literal paper pseudocode.

## Baseline performance across multiple synthetic datasets

`run_full_evaluation` was run end-to-end on three differently-configured synthetic corpora (100 texts each, `theta_filter=0.3`, `min_size_frac=0.03`, `max_depth=6`, PRGbeta), comparing the default configuration against `relative_h=True`:

### v2 — with `relative_h=True, h=0.1` (current recommended default)

| Dataset | Config | Retention | Mean leaves | Mean residual nDFU | Indeterminate rate | Jaccard | Precision | Recall | Exact match |
|---|---|---|---|---|---|---|---|---|---|
| A — default | `intensity_range=(0.3, 1.0)` | 0.93 | 7.80 | 0.196 | 0.001 | 0.863 | 0.987 | 0.875 | 0.677 |
| B — weak signal | `intensity_range=(0.2, 0.6)` | 0.88 | 9.08 | 0.246 | 0.010 | 0.882 | 0.947 | 0.935 | 0.670 |
| C — deep (biased toward k=3,4) | `depth_weights` shifted toward higher k | 0.92 | 8.45 | 0.178 | 0.003 | 0.825 | 0.993 | 0.832 | 0.587 |

### v1 — with `relative_h=False, h=0.05` (paper's literal absolute comparison)

| Dataset | Jaccard | Precision | Recall | Exact match |
|---|---|---|---|---|
| A — default | 0.812 | 0.950 | 0.862 | 0.570 |
| B — weak signal | 0.846 | 0.917 | 0.929 | 0.614 |
| C — deep | 0.820 | 0.993 | 0.827 | 0.576 |

Every metric improved or stayed flat moving from v1 to v2, on all three corpora, with no regressions observed anywhere. Gains concentrate where expected: **A and B (dominated by k=1/k=2 texts) show the largest jumps** (exact match +10.7 and +5.7 points respectively), since `relative_h` specifically helps recover weaker co-active dimensions; **C (already rich in strong, multiple true causes) barely moves** (+1.1 points), since it had little of that specific problem to fix. This directional consistency — rather than random fluctuation — supports `relative_h=True` as a genuine improvement to the method itself, not an artifact of one dataset.

## API

### `PolarizedTreesPipeline(dims, scale, theta_filter, h, max_depth, min_size=None, min_size_frac=0.03, variant="beta", beta=1.0, theta_pole=None, theta_stop=0.15, relative_h=False)`

| Parameter | Default | Description |
|---|---|---|
| `dims` | required | candidate demographic dimensions (column names) |
| `scale` | required | rating scale; ratings are integers in `[1, scale]` |
| `theta_filter` | required | Step 2. User-defined — no empirical default, since it depends on the scale and what counts as "meaningfully polarized" for your data |
| `h` | required | Step 3. Minimum PRG required to accept a split. Interpreted as a fraction of remaining nDFU when `relative_h=True` (recommend `h≈0.15` in that case, vs `h≈0.05` for the absolute default) |
| `max_depth` | required | Step 3. Maximum recursion depth |
| `min_size` | `None` | fixed absolute minimum subgroup size |
| `min_size_frac` | `0.03` | minimum subgroup size as a fraction of each text's annotator count. Ignored if `min_size` is set |
| `variant` | `"beta"` | `"max"`, `"var"`, or `"beta"` — see above |
| `beta` | `1.0` | only used when `variant="beta"` |
| `theta_pole` | `None` → `scale//2 + 1` | Step 5. rating ≥ `theta_pole` counts as toxic |
| `theta_stop` | `0.15` | node-level pre-filter; `None` disables |
| `relative_h` | `False` | compare PRG to `h` as a fraction of the node's own nDFU, instead of as an absolute amount — see "Departures from the paper" above. Recommended `True`. |

### Methods

| Method | Requires ground truth? | Returns |
|---|---|---|
| `filter_polarized_texts(dataset)` | No | list of retained text_ids (Steps 1–2) |
| `build_all_trees(dataset, text_ids=None)` | No | `{text_id: (leaves, root)}` (Steps 3–5) |
| `dimension_frequency(ground_truth=None)` | Optional | F table; adds `ever_truly_active` column if ground truth given |
| `subgroup_pole_consistency(ground_truth=None)` | Optional | C table; adds `true_lean_match_rate` column if ground truth given |
| `subgroup_prg(ground_truth=None)` | Optional | P table; adds `mean_true_alpha` column if ground truth given |
| `diagnostics()` | No | dict of corpus-level sanity stats — the real deliverable on real data |
| `recovery_metrics(ground_truth)` | **Required** | per-text jaccard/precision/recall/exact_match — synthetic validation only |
| `inspect_tree(text_id, dataset, show_distributions=False)` | No | prints one tree; pass `show_distributions=True` for a rating histogram at every node |
| `run_full_evaluation(dataset, ground_truth=None)` | Optional | runs everything above in one call |

### `diagnostics()` — the ground-truth-free deliverable

Use this on real annotation data (e.g. DICES), where there's no injected answer to check against:

```python
{
  "retention_rate": ...,       # fraction of texts that passed theta_filter
  "mean_leaves": ...,          # average tree size
  "mean_depth": ...,           # average leaf depth
  "mean_residual_ndfu": ...,   # avg. unexplained polarization left at leaves -- lower is better
  "mean_top_split_prg": ...,   # avg. strength of the primary (root) driver
  "indeterminate_rate": ...,   # fraction of leaves with no clear pole -- a coin-flip
  "dims_never_used": [...],    # dimensions that never explained anything, anywhere
}
```

### `recovery_metrics(ground_truth)` — synthetic-only validation

`ground_truth`: `dict[text_id] -> {"active_dims": [...]}`. Returns a per-text DataFrame with `jaccard`, `precision`, `recall`, `exact_match`, `k_true`. This is a validation harness, not part of the method's real output — it only exists because a synthetic generator lets you check the tree's answer against a known one.

## Demo

See [`trees_demo.ipynb`](trees_demo.ipynb) for a full walkthrough: dataset generation with known ground truth, running the pipeline, inspecting one tree with `show_distributions=True`, reading F/C/P, breaking recovery down by depth (k), and running without ground truth for the real-data use case.
