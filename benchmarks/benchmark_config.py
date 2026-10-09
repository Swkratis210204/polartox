"""
Shared configuration for the Polarized Trees synthetic benchmark.

This module is the single source of truth for what the notebooks
(datasetdemo, treesbenchmark, resultsexploration, pegcomparison) share:

* the synthetic benchmark settings (dimensions, corpora, seeds);
* the data and results folders (DATA_DIR, RESULTS_DIR);
* check_polartox(), which reports which polartox is running;
* load_corpus() / load_benchmark_corpora(), which read the saved datasets;
* load_benchmark_dataset(), the corpora A, B and C as one benchmark input.
"""

import json
from pathlib import Path

import pandas as pd

from polartox.datagen import DEFAULT_DIMENSIONS


# Folders. This file lives in the benchmarks folder, which is the notebooks'
# project root.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "benchmark_data"
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"


# Synthetic annotation environment
DIMS_DICT = DEFAULT_DIMENSIONS
DIMS = list(DIMS_DICT.keys())
SCALE = 5

N_TEXTS = 100
ANNOTATORS_PER_IDENTITY = 10
NOISE = 0.05


# Benchmark corpora used for model selection.
CORPUS_CONFIGS = [
    {
        "name": "A_default",
        "intensity_range": (0.3, 1.0),
        "depth_weights": {
            0: 0.15,
            1: 0.30,
            2: 0.25,
            3: 0.20,
            4: 0.10,
        },
        "seed": 1,
    },
    {
        "name": "B_weak_signal",
        "intensity_range": (0.2, 0.6),
        "depth_weights": {
            0: 0.15,
            1: 0.30,
            2: 0.25,
            3: 0.20,
            4: 0.10,
        },
        "seed": 2,
    },
    {
        "name": "C_deep",
        "intensity_range": (0.3, 1.0),
        "depth_weights": {
            0: 0.05,
            1: 0.10,
            2: 0.20,
            3: 0.30,
            4: 0.35,
        },
        "seed": 3,
    },
]


# Separate unseen corpus reserved for the final inference experiment.
INFERENCE_CONFIG = {
    "name": "inference_unseen",
    "intensity_range": (0.3, 1.0),
    "depth_weights": {
        0: 0.15,
        1: 0.30,
        2: 0.25,
        3: 0.20,
        4: 0.10,
    },
    "seed": 4,
}


BENCHMARK_CORPORA = [cfg["name"] for cfg in CORPUS_CONFIGS]
INFERENCE_CORPUS = INFERENCE_CONFIG["name"]


def check_polartox():
    """Print which polartox and which library versions are in use, and stop if
    this is not the current polartox. Call it at the top of a notebook: its
    results are only meaningful for the current package."""
    import importlib.metadata as md

    import polartox

    try:
        from polartox.polarized_tree import PEG_VARIANTS
    except ImportError as error:
        raise ImportError(
            "Not the current polartox (no PEG_VARIANTS). "
            "From the repository root run: pip install -e ."
        ) from error

    print("polartox", md.version("polartox"), "imported from", polartox.__file__)
    print("PEG formulations:", PEG_VARIANTS)
    for name in ("numpy", "pandas", "ndfu", "scikit-learn"):
        print(f"{name} {md.version(name)}")

    assert set(PEG_VARIANTS) == {"max", "weighted", "min", "mean", "harmonic"}


def load_corpus(name):
    """The annotations and per-text ground truth that datasetdemo.ipynb saved."""
    dataset = pd.read_csv(DATA_DIR / f"{name}_dataset.csv")

    with open(DATA_DIR / f"{name}_ground_truth.json", encoding="utf-8") as f:
        ground_truth = {int(text_id): value for text_id, value in json.load(f).items()}

    return dataset, ground_truth


def load_benchmark_corpora():
    """{name: (dataset, ground_truth)} for the model-selection corpora A, B and C."""
    return {name: load_corpus(name) for name in BENCHMARK_CORPORA}


def load_benchmark_dataset():
    """The corpora A, B and C as one benchmark input: (annotations, ground_truth,
    text_groups). Text ids are renumbered so that they are unique across the
    corpora, and text_groups maps every new id to its corpus."""
    parts, ground_truth, text_groups = [], {}, {}
    offset = 0

    for name, (dataset, truth) in load_benchmark_corpora().items():
        id_map = {old: offset + i for i, old in enumerate(sorted(dataset["text_id"].unique()))}

        parts.append(dataset.assign(text_id=dataset["text_id"].map(id_map)))
        ground_truth.update({id_map[old]: entry for old, entry in truth.items()})
        text_groups.update({new: name for new in id_map.values()})
        offset += len(id_map)

    return pd.concat(parts, ignore_index=True), ground_truth, text_groups
