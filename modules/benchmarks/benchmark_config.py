"""
Shared configuration for the Polarized Trees synthetic benchmark.

This module is the single source of truth for what the notebooks
(datasetdemo, treesbenchmark, resultsexploration, pegcomparison) share:

* the synthetic benchmark settings (dimensions, corpora, seeds);
* the data and results folders (DATA_DIR, RESULTS_DIR);
* check_polartox(), which reports which polartox is running;
* load_corpus() / load_benchmark_corpora(), which read the saved datasets;
* load_benchmark_dataset(), the ten corpora (A to J) as one benchmark input.
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


# The three corpora of the paper (A, B, C).
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


# Seven more regimes (D to J). The search (treesbenchmark.ipynb) runs on all ten corpora
# (A to J), so they are search corpora like A, B and C.
# A, B and C are not changed (same seeds), so their data stay exactly the same.
# Seed 4 belongs to the unseen inference corpus below.
_SHALLOW = {0: 0.30, 1: 0.40, 2: 0.20, 3: 0.07, 4: 0.03}
_DEFAULT = {0: 0.15, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.10}
_UNIFORM = {0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20}
_DEEP = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.30, 4: 0.35}

EXTRA_CORPUS_CONFIGS = [
    {"name": "D_strong_signal", "intensity_range": (0.6, 1.0), "depth_weights": _DEFAULT, "seed": 5},
    {"name": "E_very_weak", "intensity_range": (0.15, 0.5), "depth_weights": _DEFAULT, "seed": 6},
    {"name": "F_shallow", "intensity_range": (0.3, 1.0), "depth_weights": _SHALLOW, "seed": 7},
    {"name": "G_uniform_depth", "intensity_range": (0.3, 1.0), "depth_weights": _UNIFORM, "seed": 8},
    {"name": "H_deep_weak", "intensity_range": (0.2, 0.6), "depth_weights": _DEEP, "seed": 9},
    {"name": "I_deep_strong", "intensity_range": (0.6, 1.0), "depth_weights": _DEEP, "seed": 10},
    {"name": "J_shallow_weak", "intensity_range": (0.2, 0.6), "depth_weights": _SHALLOW, "seed": 11},
]

# All ten synthetic regimes with ground truth (A, B, C and D to J).
ALL_CORPUS_CONFIGS = CORPUS_CONFIGS + EXTRA_CORPUS_CONFIGS


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


BENCHMARK_CORPORA = [cfg["name"] for cfg in ALL_CORPUS_CONFIGS]      # A to J: the search (was A, B, C)
EVALUATION_CORPORA = [cfg["name"] for cfg in ALL_CORPUS_CONFIGS]      # A to J: the ten regimes with ground truth
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
    """{name: (dataset, ground_truth)} for the benchmark (search) corpora, A to J."""
    return {name: load_corpus(name) for name in BENCHMARK_CORPORA}


def load_evaluation_corpora():
    """{name: (dataset, ground_truth)} for all ten regimes A to J (the three of the
    paper plus the seven that datasetdemo.ipynb adds)."""
    missing = [n for n in EVALUATION_CORPORA if not (DATA_DIR / f"{n}_dataset.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Not generated yet: {missing}. Run datasetdemo.ipynb first.")
    return {name: load_corpus(name) for name in EVALUATION_CORPORA}


def load_benchmark_dataset():
    """The ten corpora (A to J) as one benchmark input: (annotations, ground_truth,
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
