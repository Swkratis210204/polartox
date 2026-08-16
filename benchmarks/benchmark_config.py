"""
Shared configuration for the Polarized Trees synthetic benchmark.

This module is the single source of truth for the synthetic benchmark
settings used by both the dataset-generation notebook and the benchmark
notebook.
"""

from polartox.datagen import DEFAULT_DIMENSIONS


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
