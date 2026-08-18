# polartox

NLP toolkit for **annotator polarization research**. Provides tools for
synthetic annotation-data generation, Polarized Trees analysis, and
systematic hyperparameter benchmarking.

## Install

```bash
pip install polartox
```

## Repository Structure

```text
polartox/
├── polartox/          installable Python package
├── data_gen/          synthetic-data generation demos and materials
├── polarized_trees/   Polarized Trees demos and research materials
├── benchmarks/        synthetic benchmark and paper reproducibility code
├── tests/             package and benchmark tests
├── CHANGELOG.md
├── LICENSE
├── README.md
└── pyproject.toml
```

### `polartox/`

The installable package containing the implementation.

It provides:

- `polartox.datagen` — synthetic annotator-pool generation;
- `polartox.polarized_trees` — Polarized Trees analysis;
- `polartox.benchmark` — hyperparameter search and model selection.

### `data_gen/`

Materials and demos for generating synthetic annotation datasets with
**known polarization ground truth**.

The synthetic generator creates annotation data for which the active
socio-demographic dimensions are known. This provides a controlled setting
for evaluating whether Polarized Trees can recover the dimensions that
generate observed disagreement.

### `polarized_trees/`

Research and demonstration materials for the Polarized Trees methodology.

Given annotation data, Polarized Trees recursively partitions annotators by
socio-demographic dimensions to identify the dimensions and intersectional
subgroups associated with polarized opinions.

### `benchmarks/`

Code used to reproduce the synthetic benchmark experiments reported in the
paper.

The benchmark workflow:

```text
Synthetic dataset generation
          ↓
Fixed datasets + ground truth
          ↓
Configuration search
          ↓
Recovery-based model selection
          ↓
Selected Polarized Trees pipeline
          ↓
Inference on unseen data
```

The `treesbenchmark.ipynb` notebook serves both as a runnable demonstration
of `PolarizedTreesBenchmark` and as the experimental workflow used to obtain
the configurations and results reported in the paper.

See [`benchmarks/README.md`](benchmarks/README.md) for the complete
benchmark workflow and reproducibility instructions.

## Tools

| Module | Description | Status |
|---|---|---|
| `polartox.datagen` | Synthetic annotator pool with injected, ground-truth polarization | Stable |
| `polartox.polarized_trees` | Polarized Trees detection algorithm | Stable |
| `polartox.benchmark` | Hyperparameter search and model selection for Polarized Trees | Stable |

## `polartox.datagen`

Generates synthetic annotation datasets with known ground truth.

Each text can have zero or more active socio-demographic dimensions that
drive disagreement, allowing the generated data to be used for quantitative
recovery evaluation.

## `polartox.polarized_trees`

Runs the Polarized Trees detection procedure on annotation data.

The pipeline identifies:

- polarized socio-demographic dimensions;
- intersectional subgroups;
- dataset-level polarization summaries **F, C, and P**;
- associated diagnostics.

When ground truth is available, recovery metrics such as Jaccard,
precision, recall, and exact match can also be computed.

## `polartox.benchmark`

`PolarizedTreesBenchmark` provides systematic hyperparameter search and model
selection when ground truth is available.

The benchmark:

1. generates configurations from a search space;
2. evaluates each configuration against the supplied ground truth;
3. computes the requested recovery metrics;
4. ranks configurations according to a selected metric;
5. returns the best configuration and pipeline;
6. provides the complete results, top configurations, and reports.

The default search space corresponds to the configuration space used in the
paper and contains **3,240 valid configurations**.

The valid PRG variant/beta combinations are:

```text
max  → beta = 1.0
var  → beta = 1.0
beta → beta = 0.5, 1.0, 2.0
```

Both `full` and `random` search are supported. Users can also customize the
search space, number of runs, seed, metrics, selection metric, and selection
direction.

## Testing

Run the complete test suite with:

```bash
python -m pytest -q
```

To run the benchmark tests specifically:

```bash
python -m pytest tests/test_benchmark.py -v
```

## nDFU

nDFU scoring is provided by the collaborative
[`ndfu`](https://github.com/ipavlopoulos/ndfu) package (Pavlopoulos & Likas,
2024) rather than reimplemented here. It is installed automatically as a
core dependency.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for release history.
