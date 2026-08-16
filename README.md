# polartox

NLP toolkit for **annotator polarization research**. Provides tools for synthetic dataset generation and polarization detection in human annotation studies.

## Install

    pip install polartox

## Repository Structure

    data_gen/          synthetic annotation dataset generator
    polarized_trees/   Polarized Trees detection algorithm
    polartox/          installable package (source code)
    benchmarks/        reproducibility code for the paper experiments

### `data_gen/`

Tools for generating synthetic annotation datasets with **injected, known polarization**. Real annotation data cannot provide ground truth for which demographic dimensions drive disagreement — this module does. The generated datasets are the primary validation input for the Polarized Trees algorithm.

→ See `data_gen/README.md` for the full API and usage.

### `polarized_trees/`

The Polarized Trees detection algorithm. Given an annotation dataset, recursively partitions annotators by demographic dimension to find the subgroups most polarized on a given text (paper Steps 1–6): which dimensions drive disagreement, and which intersectional subgroups diverge most strongly.

→ See `polarized_trees/README.md` for the full API, usage, and a comparison of three empirically-validated departures from the paper's literal specification.

## Tools

| Module | Description | Status |
|---|---|---|
| `polartox.datagen` | Synthetic annotator pool with injected, ground-truth polarization | Stable |
| `polartox.polarized_trees` | Polarized Trees detection algorithm | Stable (baseline) |

nDFU scoring is provided by the collaborative [`ndfu`](https://github.com/ipavlopoulos/ndfu) package (Pavlopoulos & Likas, 2024) rather than reimplemented here — installed automatically as a core dependency.

## Reproducing the paper benchmark

The repository also contains the experimental code used to reproduce the synthetic benchmark reported in the paper.

The benchmark code is kept separately from the installable `polartox` package under [`benchmarks/`](benchmarks/). It consists of two notebooks:

1. `benchmarks/notebooks/datasetdemo.ipynb` generates and explores the synthetic datasets with known polarization ground truth.
2. `benchmarks/notebooks/treesbenchmark.ipynb` runs the benchmark over the predefined configuration space, selects the best configuration using the synthetic ground truth, and evaluates it on an unseen synthetic corpus.

The notebooks should be run in this order:

    datasetdemo.ipynb
            ↓
    benchmark_data/
            ↓
    treesbenchmark.ipynb
            ↓
    benchmark_results/

The full benchmark is computationally expensive and may take several hours depending on the available hardware. Existing benchmark results can be inspected without rerunning the complete benchmark.

See [`benchmarks/README.md`](benchmarks/README.md) for the complete reproduction workflow, project structure, and details of the generated datasets and benchmark outputs.

## Changelog

See CHANGELOG.md for release history.
