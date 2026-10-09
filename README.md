# polartox

NLP toolkit for **annotator polarization research**. Provides tools for
synthetic annotation-data generation, Polarized Trees analysis, and
systematic hyperparameter benchmarking.

**[Project website](https://swkratis210204.github.io/polartox/)** &mdash; overview, the five modules, and nDFU explained.

## Install

```bash
pip install polartox
```

## Repository Structure

```text
polartox/
├── polartox/          installable Python package
├── data_gen/          synthetic-data generation demos and materials
├── tree_module/       single-tree (PolarizedTree) demo
├── polarized_trees/   pipeline / corpus-level Polarized Trees demos and research materials
├── peg_comparison/    comparing the PEG formulations: demo and README
├── Dices/             real-data (DICES-350/990) end-to-end inference notebook
├── benchmarks/        synthetic benchmark and paper reproducibility code
├── tests/             package, benchmark and regression tests
├── website/           project website (GitHub Pages)
├── .github/workflows/ CI (tests) and website deployment
├── CHANGELOG.md
├── LICENSE
├── README.md
└── pyproject.toml
```

### `polartox/`

The installable package containing the implementation.

It provides:

- `polartox.datagen` — synthetic annotator-pool generation;
- `polartox.polarized_tree` — single-tree construction (`PolarizedTree`, `detect_polarized_subgroups`);
- `polartox.pipeline` — corpus-level orchestration (`PolarizedTreesPipeline`);
- `polartox.benchmark` — hyperparameter search and model selection;
- `polartox.peg_comparison` — comparison of the PEG formulations on the same corpora (`PEGComparison`, `pairwise_ari`).

### `data_gen/`

Materials and demos for generating synthetic annotation datasets with
**known polarization ground truth**.

The synthetic generator creates annotation data for which the active
socio-demographic dimensions are known. This provides a controlled setting
for evaluating whether Polarized Trees can recover the dimensions that
generate observed disagreement.

### `tree_module/`

Demo for `PolarizedTree`, the single-tree building block: build and inspect
one text's polarized tree directly, without a corpus or a pipeline. See
[`tree_module/README.md`](tree_module/README.md).

### `polarized_trees/`

Research and demonstration materials for the Polarized Trees methodology at
the corpus/pipeline level (`PolarizedTreesPipeline`): filtering a corpus,
building a tree per text, and aggregating the F/C/P summaries across them.

Given annotation data, Polarized Trees recursively partitions annotators by
socio-demographic dimensions to identify the dimensions and intersectional
subgroups associated with polarized opinions.

### `Dices/`

Applies Polarized Trees to the **DICES-350** and **DICES-990** datasets —
real conversational-safety annotations, not synthetic data, so there is no
ground truth and no recovery metrics. See [`Dices/README.md`](Dices/README.md).

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

The [`treesbenchmark.ipynb`](benchmarks/notebooks/treesbenchmark.ipynb) notebook
serves both as a runnable demonstration of `PolarizedTreesBenchmark` and as
the experimental workflow used to select the configurations reported in the
paper; [`resultsexploration.ipynb`](benchmarks/notebooks/resultsexploration.ipynb)
then explores its results, and
[`pegcomparison.ipynb`](benchmarks/notebooks/pegcomparison.ipynb) compares the PEG
formulations on the best of them.

See [`benchmarks/README.md`](benchmarks/README.md) for the complete
benchmark workflow and reproducibility instructions.

## Notebooks

**End-to-end, on real annotation data:**

- [`Dices/DICES_polarized_trees_end_to_end.ipynb`](Dices/DICES_polarized_trees_end_to_end.ipynb)
  — the complete workflow on the DICES-350/990 safety datasets: filtering,
  tree construction, F/C/P, and inference without ground truth. See
  [`Dices/README.md`](Dices/README.md).

**Component demos, one workflow step at a time (synthetic data).** They start
with `pip install polartox`, so they use the *released* package; use
`pip install -e .` from a clone to run them against this repository:

- [`data_gen/datagen_demo.ipynb`](data_gen/datagen_demo.ipynb) — generate
  synthetic annotation data with known polarization ground truth
  (`polartox.datagen`).
- [`tree_module/polarized_tree_demo.ipynb`](tree_module/polarized_tree_demo.ipynb)
  — build and inspect one text's tree directly (`PolarizedTree`), without a
  pipeline or a corpus.
- [`polarized_trees/trees_demo.ipynb`](polarized_trees/trees_demo.ipynb) —
  run the full pipeline over a corpus and compute F/C/P
  (`PolarizedTreesPipeline`).
- [`benchmarks/notebooks/datasetdemo.ipynb`](benchmarks/notebooks/datasetdemo.ipynb)
  — generate the fixed synthetic benchmark corpora.
- [`benchmarks/notebooks/treesbenchmark.ipynb`](benchmarks/notebooks/treesbenchmark.ipynb)
  — hyperparameter search and saved results, including the top 20
  configurations (`PolarizedTreesBenchmark`).
- [`peg_comparison/peg_comparison_demo.ipynb`](peg_comparison/peg_comparison_demo.ipynb)
  — the chain from synthetic corpora to a comparison of the PEG formulations
  (`PEGComparison`), small enough to run in a minute.
- [`benchmarks/notebooks/resultsexploration.ipynb`](benchmarks/notebooks/resultsexploration.ipynb)
  — explores the saved results: the top 20 (per corpus), which formulations and
  hyperparameters matter, `relative_h`, recovery by corpus and by k.
- [`benchmarks/notebooks/pegcomparison.ipynb`](benchmarks/notebooks/pegcomparison.ipynb)
  — takes settings from that top 20 and compares the five PEG formulations:
  recovery, inference without ground truth, and tree shape.

## Tools

| Module | Description | Status |
|---|---|---|
| `polartox.datagen` | Synthetic annotator pool with injected, ground-truth polarization | Stable |
| `polartox.polarized_tree` | Single-tree construction (`PolarizedTree`, `detect_polarized_subgroups`) | Stable |
| `polartox.pipeline` | Corpus-level orchestration (`PolarizedTreesPipeline`) | Stable |
| `polartox.benchmark` | Hyperparameter search and model selection for Polarized Trees | Stable |
| `polartox.peg_comparison` | Comparison of the PEG formulations: recovery, tree shape, ARI | Stable |

## `polartox.datagen`

Generates synthetic annotation datasets with known ground truth.

Each text can have zero or more active socio-demographic dimensions that
drive disagreement, allowing the generated data to be used for quantitative
recovery evaluation.

## `polartox.polarized_tree` and `polartox.pipeline`

Runs the Polarized Trees detection procedure on annotation data.
`polartox.polarized_tree` builds and inspects one text's tree
(`PolarizedTree`); `polartox.pipeline` (`PolarizedTreesPipeline`) filters
a corpus down to polarized texts, builds a tree per text, and aggregates
corpus-level metrics across them.

The pipeline identifies:

- polarized socio-demographic dimensions;
- intersectional subgroups;
- dataset-level polarization summaries **F, C, and P**;
- associated diagnostics.

When ground truth is available, recovery metrics such as Jaccard,
precision, recall, and exact match can also be computed.

To see how much two configurations disagree, for example the same settings with
two different PEG formulations, `pairwise_ari(pipelines, dataset)`
(`polartox.peg_comparison`) returns the mean adjusted Rand index between their
trees: how similarly they split the annotators of each text into groups
(1 = the same groups, about 0 = chance).
`PolarizedTree.leaf_labels(dataset)` gives the group of every annotator.
`PEGComparison` does the whole comparison, see below.

### PEG formulations (`variant`)

At every node the tree splits on the dimension with the highest **PEG**
(Polarization Explanation Gain): how much of the node's polarization (its
nDFU) a split explains. PEG compares the node's nDFU with its subgroups'
nDFU. Three base formulations differ in which subgroup they compare against,
and two combine them:

| `variant` | PEG is the absolute gap between the node's nDFU and ... |
|---|---|
| `max` | the nDFU of the most polarized subgroup (PEGmax) |
| `weighted` | the size-weighted average nDFU of the subgroups (PEGweighted) |
| `min` | the nDFU of the least polarized subgroup (PEGmin) |
| `mean` | the arithmetic mean of PEGmax, PEGweighted and PEGmin |
| `harmonic` | the harmonic mean of PEGmax, PEGweighted and PEGmin (0 if any of them is 0); the default |

`beta` (default 1) only matters for `harmonic`: it weights `weighted` against `max`
and `min` with weights (1, beta^2, 1), so `beta=1` is the plain harmonic
mean. An unknown `variant` raises `ValueError` immediately. The older names
`beta` (the harmonic mean of max and weighted only) and `var` (now `weighted`) no
longer exist.

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

The default search space contains **3,240 valid configurations**: 648
settings of the other hyperparameters times the five PEG formulations
(`max`, `weighted`, `min`, `mean`, `harmonic`). `beta` is not searched; it only
weights `weighted` inside `harmonic` (default 1.0) and can be added to a custom
search space like any other parameter.

Both `full` and `random` search are supported. Users can also customize the
search space, number of runs, seed, metrics, selection metric, and selection
direction.

Long searches can run in parallel (`n_jobs`), save their progress and resume
(`checkpoint_dir`, `checkpoint_every`), and evaluate several corpora at once:
`text_groups` maps each text to a corpus, and the metrics are then computed
per corpus and averaged with equal weight per corpus. Every pipeline is built
before the first evaluation, so an invalid setting fails immediately and not
halfway through a search.

## `polartox.peg_comparison`

`PEGComparison` compares the PEG formulations on the same corpora with
everything else held fixed: for every chosen setting and corpus it runs one
pipeline per formulation. **Ground truth is optional.** Without it (real data) it
reports the trees themselves: retention, leaves, depth, group size, residual nDFU,
split PEG, the dimensions used, the trees that split on several dimensions, how
close the trees of two formulations are (`similarity`: the adjusted Rand index or the
normalized mutual information of the groups they make of the annotators, the
dimensions they split on, their first split) and the F, C and P tables of every
formulation. With ground truth it adds recovery of the true dimensions, also by the
true number of active dimensions (`recovery_by_k`). The settings can come straight
from `PolarizedTreesBenchmark.top_configs_`:

```python
from polartox import PEGComparison

comparison = PEGComparison.from_benchmark(benchmark)    # corpora, dims, scale
comparison.run(benchmark.top_configs_.head(3))          # each row with every formulation
comparison.overview()
```

See [`peg_comparison/README.md`](peg_comparison/README.md).

## Testing

Run the complete test suite with:

```bash
python -m pytest -q
```

To run the benchmark tests specifically:

```bash
python -m pytest tests/test_benchmark.py -v
```

The suite includes a **golden-snapshot regression test**
(`tests/test_regression.py`): the real pipeline and benchmark are run on a
fixed seeded corpus and compared with `tests/golden_snapshot.json`. Those
numbers depend on the versions of `ndfu`, numpy, pandas and scikit-learn
(random streams and float arithmetic), which the golden file records. When
the installed versions differ, the test is skipped with a message saying so;
set `POLARTOX_REQUIRE_GOLDEN=1` to make that a failure instead.

Continuous integration (`.github/workflows/python-package.yml`) runs the
suite on Python 3.9 to 3.12 with the latest dependencies (golden test
skipped), and a second job pinned to the golden file's versions with
`POLARTOX_REQUIRE_GOLDEN=1`, so the comparison always runs somewhere.
Regenerate the golden file only for an intended behaviour change, with
`python tests/_snapshot.py tests/golden_snapshot.json`, and update the pins in
the workflow to match.

## nDFU

nDFU scoring is provided by the collaborative
[`ndfu`](https://github.com/ipavlopoulos/ndfu) package (Pavlopoulos & Likas,
2024) rather than reimplemented here. It is installed automatically as a
core dependency.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for release history.
