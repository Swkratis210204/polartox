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
    min_size_frac=0.03,    # min subgroup size, as a fraction of each text's annotators -- mandatory-ish default
    h=0.05,                # PRG gain threshold
    max_depth=6,
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

Three design choices in this implementation diverge from the paper's literal specification. Each was tested directly against the alternative on synthetic ground-truth data before being adopted as the default.

### `min_size_frac` instead of a fixed `min_size`

The paper's `nmin=2` is a mathematical floor (below 2 annotations, nDFU is undefined) — not a recommendation for reliable output. Testing `min_size=2` directly against `min_size=50` (on a 1,620-annotator pool) found:

| | `min_size=2` | `min_size=50` |
|---|---|---|
| Texts with ≥1 indeterminate (coin-flip) leaf | 100% | 8.9% |
| Mean leaves per tree | 12.0 | 6.8 |
| Mean jaccard vs. ground truth | 0.804 | 0.887 |

`min_size=2` reliably fragments trees on pure sampling noise. Since a fixed absolute count means something completely different depending on how many annotators a dataset has per text (50 is ~3% of 1,620, but ~33% of 150 — unusable), subgroup size is expressed as `min_size_frac`, a fraction of *that specific text's* annotator count, and resolved to an absolute count internally. Pass `min_size` instead for a fixed absolute value if you prefer.

### PRGbeta as the default splitting criterion, not PRGmax

The paper names PRGmax "the primary splitting criterion." Direct testing found a concrete case where PRGmax's worst-case-only design picked a spurious dimension over a genuine one: a real cause with 2 clean subgroups and 1 imperfect one (PRGmax penalized by the one bad subgroup) lost to a fake cause that merely had one large, coincidental swing (PRGmax = 0.055 for the real cause vs. 0.156 for the fake one). PRGvar (size-weighted average) correctly ranked the real cause far higher (0.331 vs. 0.010) in the same case, but alone tends to produce values too small to clear a sensible `h`. PRGbeta (harmonic mean of both, the paper's own proposed combination) correctly ranked the true cause first (0.093 vs. 0.019) while producing usable magnitudes — hence the default.

### `theta_stop`: a node-level pre-filter not present in the paper

If a node's own nDFU is already below `theta_stop`, the tree stops immediately without searching for a split at all — distinct from `h`, which asks whether a *found* split is good enough to use. Without this check, a node that is already essentially resolved can still fragment further on pure noise: removing it from an otherwise-identical run increased one text's tree from 6 leaves to 58, with the extra branches splitting on demographic dimensions that had no real effect on that text. Set `theta_stop=None` to disable and match the literal paper pseudocode.

## Baseline performance across multiple synthetic datasets

To check the method isn't tuned to one specific data-generating setup, `run_full_evaluation` was run end-to-end on three differently-configured synthetic corpora (100 texts each, default `theta_filter=0.3`, `min_size_frac=0.03`, `h=0.05`, `max_depth=6`, PRGbeta):

| Dataset | Config | Retention | Mean leaves | Mean residual nDFU | Indeterminate rate | Jaccard | Precision | Recall | Exact match |
|---|---|---|---|---|---|---|---|---|---|
| A — default | `intensity_range=(0.3, 1.0)` | 0.93 | 8.00 | 0.214 | 0.004 | 0.812 | 0.950 | 0.862 | 0.570 |
| B — weak signal | `intensity_range=(0.2, 0.6)` | 0.88 | 9.33 | 0.259 | 0.017 | 0.846 | 0.917 | 0.929 | 0.614 |
| C — deep (biased toward k=3,4) | `depth_weights` shifted toward higher k | 0.92 | 8.34 | 0.181 | 0.005 | 0.820 | 0.993 | 0.827 | 0.576 |

Results move in an interpretable, mechanism-consistent direction rather than fluctuating randomly:

- **Weaker overall signal (B)** raises recall (0.929) — with no single dimension dominating as strongly, more true causes get found — but also raises the indeterminate rate (~4x A) and lowers precision slightly, since weaker signal produces more genuinely ambiguous outcomes and occasional spurious splits.
- **More true causes per text (C)** raises precision to near-perfect (0.993) — the method still rarely invents a fake cause — but lowers recall (0.827), since more true causes per text means more chances for the weakest one to be absorbed by stronger co-occurring causes (a pattern documented throughout development).

No configuration collapses or behaves erratically; jaccard stays in a tight 0.81–0.85 band across all three, supporting this as a usable baseline across corpora with varying signal strength and complexity, not just the default settings.

## API

### `PolarizedTreesPipeline(dims, scale, theta_filter, h, max_depth, min_size=None, min_size_frac=0.03, variant="beta", beta=1.0, theta_pole=None, theta_stop=0.15)`

| Parameter | Default | Description |
|---|---|---|
| `dims` | required | candidate demographic dimensions (column names) |
| `scale` | required | rating scale; ratings are integers in `[1, scale]` |
| `theta_filter` | required | Step 2. User-defined — no empirical default, since it depends on the scale and what counts as "meaningfully polarized" for your data |
| `h` | required | Step 3. Minimum PRG required to accept a split |
| `max_depth` | required | Step 3. Maximum recursion depth |
| `min_size` | `None` | fixed absolute minimum subgroup size |
| `min_size_frac` | `0.03` | minimum subgroup size as a fraction of each text's annotator count. Ignored if `min_size` is set |
| `variant` | `"beta"` | `"max"`, `"var"`, or `"beta"` — see above |
| `beta` | `1.0` | only used when `variant="beta"` |
| `theta_pole` | `None` → `scale//2 + 1` | Step 5. rating ≥ `theta_pole` counts as toxic |
| `theta_stop` | `0.15` | node-level pre-filter; `None` disables |

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