# `polartox.datagen` --- Synthetic Annotation Dataset Generator

`polartox.datagen` generates synthetic annotation datasets where the
source of polarization is known by construction. This makes it possible
to test polarization-attribution methods against known ground truth.

## Install

``` bash
pip install polartox
```

## How it works

The generator has two main stages:

1.  **Build an annotator population.**\
    Demographic dimensions are combined into intersectional identities,
    and each identity is replicated into the requested number of
    annotators.

2.  **Generate texts independently.**\
    For each text, the generator samples the number of active dimensions
    `k`, selects the active dimensions, assigns their polarization
    profiles, and combines their effects to generate the observed
    ratings.

The resulting annotations are returned together with the generation
configuration used as ground truth.

## Quickstart

``` python
from polartox.datagen import (
    AnnotatorPool,
    DEFAULT_DIMENSIONS,
    DEFAULT_DEPTH_WEIGHTS,
    DEFAULT_INTENSITY_RANGE,
)

pool = AnnotatorPool(
    dimensions=DEFAULT_DIMENSIONS,
    scale=5,
    intensity_range=DEFAULT_INTENSITY_RANGE,
    depth_weights=DEFAULT_DEPTH_WEIGHTS,
    annotators_per_identity=10,
    alpha_window=0.15,
)

pool.summary()

result = pool.generate_dataset(
    n_texts=100,
    n_annotators_per_text=None,
    noise=0.05,
    seed=42,
)

dataset, ground_truth = result
```

## The generation process

For every text:

-   `k` determines how many SCD dimensions are active.
-   The active dimensions are selected from the available dimensions.
-   Each active dimension receives a random toxic/civil subgroup lean.
-   Each active dimension receives an intensity `alpha`.
-   The active-dimensional distributions are combined by elementwise
    product and normalized.
-   Annotators then receive ratings from the resulting distributions.
-   `noise` can replace an annotation with a random rating.

A text with `k=0` is a non-demographic negative control. When `k>0`, the
active dimensions are the ground truth that Polarized Trees can later
try to recover.

## Ground truth

For each text, the generator retains the configuration that produced its
ratings.

For `k > 0`, this includes:

``` python
{
    "active_dims": [...],
    "lean": {...},
    "alpha": {...},
}
```

The most important field for evaluating Polarized Trees is:

``` python
ground_truth[text_id]["active_dims"]
```

because it identifies the dimensions that were deliberately activated.

## Measuring polarization

The generated ratings can be summarized using nDFU:

``` python
from ndfu import dfu, pdf

text_data = dataset[dataset["text_id"] == 0]

hist = pdf(
    text_data["rating"].tolist(),
    range(1, pool.scale + 1),
)

score = dfu(hist)
print("nDFU:", score)
```

This lets us check the polarization produced by the generator before
testing a detection method.

## Main API

### `AnnotatorPool`

``` python
AnnotatorPool(
    dimensions,
    scale,
    intensity_range,
    depth_weights,
    annotators_per_identity,
    alpha_window=0.15,
)
```

-   `dimensions`: demographic dimensions and their possible values.
-   `scale`: maximum rating value.
-   `intensity_range`: range from which polarization strength is
    sampled.
-   `depth_weights`: probability of each possible number of active
    dimensions.
-   `annotators_per_identity`: annotators created for each identity.
-   `alpha_window`: limits the difference in strength between co-active
    dimensions.

The package provides reference configurations through
`DEFAULT_DIMENSIONS`, `DEFAULT_DEPTH_WEIGHTS`, and
`DEFAULT_INTENSITY_RANGE`.

### `generate_dataset`

``` python
pool.generate_dataset(
    n_texts,
    n_annotators_per_text=None,
    noise=0.05,
    seed=0,
)
```

`n_annotators_per_text=None` uses the full annotator pool.

The returned `GeneratedDataset` can be unpacked as:

``` python
dataset, ground_truth = result
```

It also provides convenient inspection methods:

``` python
result.head()
result.tail()
result.sample(5, random_state=0)
result.text_ids_by_k(2)
result.describe_text(0)
```

## Demo

See [`datagen_demo.ipynb`](datagen_demo.ipynb) for a short walkthrough
of the generation process.

The demo goes from:

**annotator population → synthetic annotations → ground truth → observed
rating distributions → nDFU → variation across `k`**

It is intentionally small and explanatory rather than a full benchmark.

## From synthetic data to Polarized Trees

The generator provides the controlled environment for evaluating
`polartox.pipeline` (`PolarizedTreesPipeline`):

``` text
synthetic data + known ground truth
              ↓
      Polarized Trees
              ↓
     recovered dimensions
              ↓
        evaluation
```

Because the true active dimensions are known, recovery can be evaluated
using precision, recall, Jaccard similarity, and exact match.

For the full synthetic benchmark and reproducibility workflow, see the
repository's `benchmarks/` directory.
