# `polartox.datagen` — Synthetic Annotation Dataset Generator

Builds a pool of annotators with explicit demographic identities and generates annotation datasets where every text independently gets **k active dimensions** (0–4) that drive its disagreement — `k=0` is a true unimodal negative control, `k>0` texts have a random subset of dimensions, each with its own random toxic/civil lean split and intensity.

Each active dimension's pull is governed by a single continuous **intensity** parameter, interpolating between the uniform distribution (no signal) and a fully deterministic pole (maximal signal). Identities' final rating distributions are built by taking the **elementwise product** of their active-dimension shapes — signal composes rather than averages away, so the dataset reaches the full nDFU range instead of collapsing toward the middle.

Because the generative config is returned alongside the dataset, recovery can be tested directly against ground truth: did the algorithm find the right dimensions, at the right depth, for texts of varying k and intensity?

## Install

```bash
pip install polartox
```

## Quickstart

```python
from polartox.datagen import AnnotatorPool, DEFAULT_DIMENSIONS, DEFAULT_DEPTH_WEIGHTS, DEFAULT_INTENSITY_RANGE

pool = AnnotatorPool(
    dimensions=DEFAULT_DIMENSIONS,
    scale=5,                              # ratings are integers in [1, scale] -- mandatory
    intensity_range=DEFAULT_INTENSITY_RANGE,  # (alpha_min, alpha_max), each active dim draws its own alpha -- mandatory
    depth_weights=DEFAULT_DEPTH_WEIGHTS,      # P(k active dims) for k = 0..4 -- mandatory
    annotators_per_identity=10,           # annotators replicated per unique identity
)
pool.summary()
# Dimensions        : ['gender', 'politics', 'age', 'education', 'orientation']
# Pool size          : 1620  (162 identities × 10 annotators each)

result = pool.generate_dataset(
    n_texts=100,                  # number of texts to generate
    n_annotators_per_text=None,   # None uses the full pool; pass an int to subsample instead
    noise=0.05,                   # prob. any rating is drawn fully at random
    seed=42,
)

# Unpacks like the classic (dataset, ground_truth) tuple:
dataset, ground_truth = result

# ...or use it directly:
result.head()
result.describe_text(text_id=0)
```

## How it works

For every text, independently:

1. **Sample k** — how many dimensions are active, drawn from `depth_weights` (a distribution over 0..len(dimensions)). `k=0` is the negative control: every sampled annotator draws from a single shared random peak on the scale, with natural spread — no demographic structure at all.
2. **Sample which k dimensions** are active, uniformly at random from the full set — no dimension is privileged across the dataset.
3. For each active dimension, **randomly split its values** into a toxic-leaning group and a civil-leaning group (the split is re-drawn per text — no value is permanently "the toxic one").
4. For each active dimension, **draw an intensity `alpha`** uniformly from `intensity_range`. `alpha=0` means that dimension contributes no signal (identical to uniform); `alpha=1` means it deterministically forces its pole.
5. Each identity's final rating distribution is the **elementwise product** of the shapes implied by its value on every active dimension, renormalized. Agreeing leans sharpen the result; conflicting leans pull it back toward the middle — this is what produces genuine intersectional behavior without hand-specifying every combination.

`noise` applies uniformly regardless of k: with that probability, any individual annotator's rating is instead drawn fully at random from the whole scale.

### Why elementwise product instead of averaging

An earlier version of this generator combined per-dimension signals by taking the geometric mean of per-value weights, then thresholding at the population median. Averaging washes out variance — sampling many annotations from a large, diverse pool collapses toward the aggregate regardless of injected structure (a consequence of the Central Limit Theorem in log-space), capping achievable nDFU well below 1. Multiplying distributions is an *intersection* of constraints, not a lossy aggregate: it composes signal instead of diluting it, which is what lets this generator reach the full nDFU range.

## API

### `AnnotatorPool(dimensions, scale, intensity_range, depth_weights, annotators_per_identity)`

| Parameter | Default | Description |
|---|---|---|
| `dimensions` | required | `dict[str, list[str]]` — demographic axes and their values |
| `scale` | required | max rating value; ratings are integers in `[1, scale]` |
| `intensity_range` | required | `(alpha_min, alpha_max)`, each in `[0, 1]`. Controls how strongly each active dimension pulls toward its pole. |
| `depth_weights` | required | `dict[int, float]`, `P(k active dims)` for `k = 0..len(dimensions)`. Must sum to 1. |
| `annotators_per_identity` | required | replication factor per unique identity combination |

All parameters are mandatory — there is no silent fallback for any of them. If you want the reference configuration, pass the `DEFAULT_*` constants explicitly, as in the quickstart above.

### `pool.generate_dataset(...)`

| Parameter | Default | Description |
|---|---|---|
| `n_texts` | required | number of texts to generate |
| `n_annotators_per_text` | `None` | annotators sampled per text (without replacement). `None`, or any value ≥ `pool.pool_size`, uses the full pool for every text. |
| `noise` | `0.05` | probability that any annotator's rating is drawn fully at random |
| `seed` | `0` | random seed |

Returns a `GeneratedDataset`.

### `GeneratedDataset`

The return value of `generate_dataset`. Unpacks as `(dataset, ground_truth)` for backward compatibility, and also supports:

```python
result.data              # the underlying pd.DataFrame
result.ground_truth       # the underlying dict[int, dict]

result.head(n=5)          # dataset.head(n)
result.tail(n=5)          # dataset.tail(n)
result.sample(n=5, **kw)  # dataset.sample(n, **kw)

result.text_ids_by_k(k)   # list of text_ids with exactly k active dimensions

result.describe_text(text_id)
# Text 3  (n_annotators=1620)
#   k = 2
#   active_dims = ['politics', 'age']
#   politics     alpha=0.74  lean: left=civil, center=civil, right=toxic
#   age          alpha=0.55  lean: <25=toxic, 25-50=toxic, >50=civil
#   rating counts: {1: 210, 2: 188, 3: 240, 4: 401, 5: 581}
```

`ground_truth[text_id]` is a dict:
- for `k=0` texts: `{"active_dims": [], "peak": int, "spread": float}`
- for `k>0` texts: `{"active_dims": [...], "lean": {dim: {value: "toxic"/"civil"}}, "alpha": {dim: float}}`

### Diagnostics

```python
pool.pool_size       # int -- total annotators
pool.n_identities     # int -- unique demographic combinations
pool.summary()         # prints full pool config
```

nDFU scoring is provided by the external [`ndfu`](https://github.com/ipavlopoulos/ndfu) package (install via `pip install "polartox[ndfu]"`):

```python
from ndfu import dfu, pdf

text_data = dataset[dataset["text_id"] == 0]
hist = pdf(text_data["rating"].tolist(), range(1, pool.scale + 1))
score = dfu(hist)
```

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
    dimensions={"politics": ["left", "center", "right"], "age": ["<25", ">25"]},
    scale=5,
    intensity_range=(0.3, 1.0),
    depth_weights={0: 0.1, 1: 0.5, 2: 0.4},
    annotators_per_identity=10,
)
# 6 identities → 60 annotators
```

## Demo

See [`datagen_demo.ipynb`](datagen_demo.ipynb) for an end-to-end walkthrough with visualizations.

## Migrating from `polartox`

This package replaces the earlier severity-tier / weight-threshold mechanism from `polartox` entirely. Key differences:

| Old (`polartox`) | New (`polartox.datagen`) |
|---|---|
| `high_ratio` / `moderate_ratio` / `low_ratio` tiers | `depth_weights` over `k` (number of active dims) |
| `toxic_range` / `civil_range` / `neutral_range` | `intensity_range` (continuous `alpha`) |
| Geometric mean of weights + median threshold | Elementwise product of per-dimension shapes |
| Returns `(dataset, bias_configs)` tuple | Returns `GeneratedDataset` (still unpacks as a tuple) |
| `pool.describe_bias`, `pool.analyze`, `pool.summarize`, `pool.summarize_all` | `result.describe_text(id)`; nDFU via the external `ndfu` package |

The old mechanism is no longer maintained; its known limitation (nDFU capped well below 1 due to CLT-driven averaging) is what motivated this rewrite.