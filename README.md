# Polarized Trees — Synthetic Annotation Dataset

Synthetic data generator for studying **demographic polarization in human annotation**.
Builds a structured pool of annotators with explicit demographic profiles and generates
disagreement patterns governed by a configurable bias model.

---

## What this does

Real annotation studies often show that annotators from different demographic groups
disagree not randomly, but **systematically**: two groups reliably rate the same content
at opposite ends of the scale. This project generates synthetic datasets that mimic that
structure, so that polarization-detection methods (like [Polarized Trees](polarized_trees.pdf))
can be validated against known ground truth.

### Core concept

1. **Dimensions** — demographic axes (e.g. politics, age) each with a set of values.
2. **AnnotatorPool** — the Cartesian product of all dimension values, replicated N times,
   gives a fixed pool of annotators each with a unique demographic profile.
3. **Bias config** — for each dataset instance, every dimension is randomly assigned a role:
   - **polarizing**: values are split into a *toxic pole* and a *civil pole*;
     annotators on the toxic side draw high ratings, civil side draws low ratings.
   - **unimodal**: all annotators converge toward one rating range (toxic / civil / neutral),
     regardless of their demographic value on that dimension.
4. **Annotation** — each annotator's rating is determined by majority vote across all
   polarizing dimensions they belong to, plus a configurable noise term.
5. **nDFU** — the [normalized Disagreement from Uniformity](https://pypi.org/project/ndfu/)
   score quantifies how bimodal a rating distribution is. Higher = more polarized.

---

## File overview

| File | Purpose |
|---|---|
| `polarization_dataset.py` | The library. Import `AnnotatorPool` from here. |
| `Polarization_Dataset.ipynb` | Original notebook — quick demo and nDFU analysis. |
| `polarizedtrees.ipynb` | Polarized Trees model experiments. |
| `notes/datagen_approach.txt` | Design rationale for the data-generation pipeline. |
| `notes/notes.txt` | Known limitations and design decisions. |

---

## Quickstart

```python
from polarization_dataset import AnnotatorPool, DEFAULT_DIMENSIONS

# 1. Define (or reuse) demographic dimensions
dimensions = {
    "gender":      ["male", "female", "non-binary"],
    "politics":    ["left", "center", "right"],
    "age":         ["<25", "25-50", ">50"],
    "education":   ["low", "medium", "high"],
    "orientation": ["heterosexual", "lgbtq+"],
}

# 2. Build the annotator pool
pool = AnnotatorPool(dimensions)
pool.summary()
# Active dimensions : ['gender', 'politics', 'age', 'education', 'orientation']
# Unique identities : 162
# Annotators/identity: 10
# Pool size          : 1620
# Rating scale       : 1–5

# 3. Generate a dataset
dataset, bias_config = pool.generate_dataset(
    n_texts=100,            # number of texts to annotate
    n_annotators_per_text=100,  # annotators sampled per text (≤ pool size)
    noise=0.1,              # prob. of ignoring bias config (random rating)
    polarizing_prob=0.7,    # prob. that each dimension is polarizing
)

print(dataset.head())
# text_id  annotator_id  gender  politics  age  education  orientation  rating
# 0        946           female  right     <25  high       heterosexual  5
# ...

# 4. Inspect the bias config (the ground truth for this dataset)
for dim, config in bias_config.items():
    print(f"{dim}: {config}")
# gender:    {'role': 'unimodal', 'convergence': 'toxic'}
# politics:  {'role': 'polarizing', 'toxic_pole': ['left', 'right'], 'civil_pole': ['center']}
# ...
```

---

## `AnnotatorPool` — full API

### Constructor

```python
AnnotatorPool(
    dimensions,                   # required — see below
    exclude=None,                 # list of dimension names to drop
    annotators_per_identity=10,   # replication factor per unique identity
    scale=5,                      # max rating value (ratings are 1..scale)
    toxic_range=(4, 5),           # rating range for toxic-pole annotators
    civil_range=(1, 2),           # rating range for civil-pole annotators
    neutral_range=(3, 3),         # rating range for neutral unimodal convergence
)
```

### `dimensions` — the only required argument

A `dict` mapping each dimension name (`str`) to a list of its possible values (`list[str]`).

```python
dimensions = {
    "politics": ["left", "center", "right"],   # 3 values
    "age":      ["<25", ">25"],                # 2 values
}
# → 3 × 2 = 6 unique identities → 60 annotators (with annotators_per_identity=10)
```

Minimum: one dimension with at least two values.

### `generate_dataset()`

```python
dataset, bias_config = pool.generate_dataset(
    n_texts=100,
    n_annotators_per_text=100,  # must be ≤ pool.pool_size
    noise=0.1,
    polarizing_prob=0.7,
)
```

**Returns:**

- `dataset` — `pd.DataFrame` with columns:
  `text_id`, `annotator_id`, one column per active dimension, `rating`
- `bias_config` — `dict` describing the polarization structure used for this dataset

**Constraint:** `n_annotators_per_text ≤ pool.pool_size`
(= `n_identities × annotators_per_identity`).

### Properties / diagnostics

```python
pool.pool_size     # total annotators
pool.n_identities  # unique demographic combinations
pool.summary()     # prints all config in one shot
pool.pool          # the full pd.DataFrame of annotators
pool.active_dims   # the dimensions dict after applying exclude
```

---

## Rating scale

| Range | Meaning |
|---|---|
| `toxic_range` (default 4–5) | Annotator is on the **toxic pole** of a polarizing dimension |
| `civil_range` (default 1–2) | Annotator is on the **civil pole** of a polarizing dimension |
| `neutral_range` (default 3–3) | Unimodal dimension converges to "neutral" |

All three ranges are sampled uniformly at random (inclusive bounds). `scale` sets the
upper limit for noise ratings.

---

## How the bias config works

```
bias_config = {
  "politics": {
    "role": "polarizing",
    "toxic_pole": ["left", "right"],   # annotators with these values → high ratings
    "civil_pole": ["center"]           # annotators with these values → low ratings
  },
  "age": {
    "role": "unimodal",
    "convergence": "toxic"             # all age groups converge to high ratings
  },
  ...
}
```

For each annotator, polarizing dimensions cast a vote ("toxic" or "civil").
The majority vote wins; the annotator then draws a rating from the winning range.
If no polarizing dimension applies, the unimodal convergence is used as a fallback.

---

## Known limitations

- **Balanced splits by construction.** The random pole assignment tends to produce
  roughly balanced toxic/civil splits; real polarization is often asymmetric (80/20).
- **No interaction effects.** Dimensions contribute independently (additive in the
  majority-vote scheme). A specific *combination* of values (e.g. male + right-wing)
  cannot carry disproportionate weight beyond what each dimension contributes alone.
- **Single bias config per dataset.** All texts in one `generate_dataset()` call share
  the same bias config. Real corpora contain many topics with different polarization
  structures; call `generate_dataset()` separately for each topic to vary this.

---

## Dependencies

```
numpy
pandas
ndfu       # pip install ndfu  (for nDFU computation in the notebook)
```
