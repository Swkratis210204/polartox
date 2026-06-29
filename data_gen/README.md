# `toxpol.datagen` — Synthetic Annotation Dataset Generator

Builds a pool of annotators with explicit demographic identities and generates annotation datasets where every text is independently assigned a **severity tier** (High, Moderate, or Low polarization), with per-text random weights producing genuine intersectional disagreement.

Because the bias configuration is returned alongside the dataset, recovery can be tested directly against ground truth: did the algorithm find the right dimensions, at the right depth, for texts of each severity tier?

## Install

```bash
pip install toxpol-nlp

# with nDFU support (for analysis methods)
pip install "toxpol-nlp[ndfu]"
```

## Quickstart

```python
from toxpol.datagen import AnnotatorPool, DEFAULT_DIMENSIONS

pool = AnnotatorPool(
    dimensions=DEFAULT_DIMENSIONS,
    scale=5,                      # ratings are integers in [1, scale] -- mandatory
    toxic_range=(4, 5),           # rating range for the toxic pole in the High tier -- mandatory
    civil_range=(1, 2),           # rating range for the civil pole in the High tier -- mandatory
    neutral_range=(3, 3),         # reserved for future use -- mandatory
    exclude=None,                 # drop dimension names, e.g. ["education"]
    annotators_per_identity=10,   # annotators replicated per unique identity
)
pool.summary()
# Active dimensions : ['gender', 'politics', 'age', 'education', 'orientation']
# Pool size          : 1620  (162 identities × 10 annotators each)

dataset, bias_configs = pool.generate_dataset(
    n_texts=100,                  # number of texts to generate
    n_annotators_per_text=150,    # sampled per text; must be ≤ pool.pool_size
    noise=0.05,                   # prob. any rating is drawn fully at random
    high_ratio=0.60,              # share of texts with strong, non-overlapping polarization
    moderate_ratio=0.20,          # share with the same mechanism but a softer signal
    low_ratio=0.20,                # share with little to no polarization (negative control)
    low_unimodal_share=0.40,      # within Low, fraction with no demographic split at all
)
# dataset columns: text_id, annotator_id, <dimensions>, rating
# bias_configs: ground-truth tier, sub-case, weights, and threshold per text
```

## How it works

Every text is independently assigned one of three severity tiers, mixed according to `high_ratio` / `moderate_ratio` / `low_ratio` (which must sum to 1.0):

- **High** — every value of every dimension gets a fresh random weight (drawn from `(0.5, 2.0)`). An annotator's combined weight is the geometric mean of their weights across all dimensions. A threshold is computed as the median combined weight across the **full pool**; annotators above it rate from `toxic_range`, below it from `civil_range`. Because every dimension contributes simultaneously, no single dimension is privileged — the combination of dimensions that best explains the disagreement varies from text to text.
- **Moderate** — the same weight mechanism, but with narrower, more overlapping ranges than High, producing a softer signal that should resolve in fewer splits.
- **Low** — a `low_unimodal_share` fraction of Low texts use no demographic split at all (every sampled annotator draws from a single random peak on the scale, with natural spread — the true negative control). The remainder use the weight mechanism with even more heavily overlapping ranges than Moderate, producing only marginal residual signal.

`noise` applies uniformly across all tiers: with that probability, any individual annotator's rating is instead drawn fully at random from the whole scale, regardless of tier or pole.

### Moderate and Low ranges are derived, not separately configured

You only ever specify `toxic_range`/`civil_range` once, for the High tier. Moderate's and Low's ranges are automatically derived from them in `__init__`, shifted toward the center by a step proportional to `scale` — so they scale correctly no matter what rating scale you use, without needing four extra parameters:

```python
pool.moderate_toxic_range   # derived from toxic_range
pool.moderate_civil_range   # derived from civil_range
pool.low_toxic_range        # derived from toxic_range, shifted further
pool.low_civil_range        # derived from civil_range, shifted further
```

For example, with `scale=5, toxic_range=(4,5), civil_range=(1,2)`:

```python
moderate_toxic_range = (4, 5)   # same as High
moderate_civil_range = (1, 3)
low_toxic_range      = (3, 5)
low_civil_range      = (1, 3)
```

## API

### `AnnotatorPool(dimensions, scale, toxic_range, civil_range, neutral_range, ...)`

| Parameter | Default | Description |
|---|---|---|
| `dimensions` | required | `dict[str, list[str]]` — demographic axes and their values |
| `scale` | required | max rating value; ratings are integers in `[1, scale]`. No default — Moderate/Low ranges are derived from it. |
| `toxic_range` | required | rating range for the toxic pole in the High tier. No default — also the basis for Moderate/Low's toxic ranges. |
| `civil_range` | required | rating range for the civil pole in the High tier. No default — also the basis for Moderate/Low's civil ranges. |
| `neutral_range` | required | reserved for future use. No default. |
| `exclude` | `None` | dimension names to drop (for ablations) |
| `annotators_per_identity` | `10` | replication factor per unique identity combination |

`scale`, `toxic_range`, `civil_range`, and `neutral_range` must all be passed explicitly — there is no fallback default for any of them.

### `pool.generate_dataset(...)`

| Parameter | Default | Description |
|---|---|---|
| `n_texts` | `100` | number of texts to generate |
| `n_annotators_per_text` | `100` | annotators sampled per text (without replacement); must be `≤ pool.pool_size` |
| `noise` | `0.05` | probability that any annotator's rating is drawn fully at random |
| `high_ratio` | `0.60` | share of texts assigned the High tier |
| `moderate_ratio` | `0.20` | share of texts assigned the Moderate tier |
| `low_ratio` | `0.20` | share of texts assigned the Low tier |
| `low_unimodal_share` | `0.40` | within Low, the fraction using the true-unimodal sub-case rather than the weighted sub-case |

`high_ratio + moderate_ratio + low_ratio` must sum to 1.0. These are *probabilities* applied independently per text, not exact counts — with 100 texts and `high_ratio=0.60`, expect approximately (not exactly) 60 High-tier texts.

Returns `(dataset, bias_configs)`. Every text's tier and bias config are generated independently — call once for a full dataset with built-in heterogeneity across severity levels.

### Diagnostics

```python
pool.pool_size      # int — total annotators
pool.n_identities    # int — unique demographic combinations
pool.active_dims     # dict — dimensions after applying exclude
pool.summary()        # prints full pool config, including derived Moderate/Low ranges

pool.describe_bias(bias_configs, text_id=0)
# Text 0 -- tier: high
#   threshold (median combined weight): 1.124
#   gender       male=1.87  female=1.80  non-binary=0.82
#   politics     left=0.71  center=1.23  right=0.88
#   ...

pool.summarize(dataset, bias_configs, text_id=0)
# Text 0 (tier: high) -- overall nDFU: 0.735
# gender:
#   male: 0.310
#   female: 0.360
#   ...

results = pool.analyze(dataset, bias_configs)  # raw nDFU scores
# results[text_id]["overall"]      -> float
# results[text_id][dim][value]     -> float

pool.summarize_all(dataset, bias_configs)
# Overall nDFU -- mean: 0.459  median: 0.559  (across 100 texts)
# Tier counts: {'high': 61, 'moderate': 19, 'low': 20}
# high       mean: 0.735  min: 0.485  max: 0.964  (n=61)
# moderate   mean: 0.109  min: 0.000  max: 0.269  (n=19)
# low        mean: 0.090  min: 0.000  max: 0.235  (n=20)
```

`analyze()`, `summarize()`, and `summarize_all()` require `pip install "toxpol-nlp[ndfu]"`.

## Default Dimensions

```python
DEFAULT_DIMENSIONS = {
    "gender":      ["male", "female", "non-binary"],
    "politics":    ["left", "center", "right"],
    "age":         ["<25", "25-50", ">50"],
    "education":   ["low", "medium", "high"],
    "orientation": ["heterosexual", "lgbtq+"],
}
# → 162 unique identities → 1620 annotators (default annotators_per_identity=10)
```

Pass any custom dict to use different axes or values:

```python
pool = AnnotatorPool(
    {"politics": ["left", "center", "right"], "age": ["<25", ">25"]},
    scale=5, toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3),
)
# 6 identities → 60 annotators
```

## Demo

See [`datagen_demo.ipynb`](datagen_demo.ipynb) for an end-to-end walkthrough with visualizations.