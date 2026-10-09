"""
Builds a compact snapshot of the real pipeline and benchmark on a fixed
seeded corpus. tests/test_regression.py compares it with
tests/golden_snapshot.json, so any refactor that changes a reported number
is caught.

Regenerate the golden file ONLY when a behaviour change is intended:

    python tests/_snapshot.py tests/golden_snapshot.json
"""

import json
import sys

import numpy as np
import pandas as pd

from polartox import (
    AnnotatorPool,
    DEFAULT_DIMENSIONS,
    PolarizedTreesBenchmark,
    PolarizedTreesPipeline,
)

N_TEXTS = 12

BASE = dict(dims=list(DEFAULT_DIMENSIONS), scale=5, theta_filter=0.3, h=0.15,
            max_depth=6, theta_stop=0.1)

PIPELINES = {
    "harmonic_rel": dict(variant="harmonic", relative_h=True, min_size_frac=0.03),
    "harmonic_beta2": dict(variant="harmonic", beta=2.0, min_size_frac=0.03),
    "mean": dict(variant="mean", min_size_frac=0.03),
    "max": dict(variant="max", min_size_frac=0.05),
    "weighted": dict(variant="weighted", min_size_frac=0.02),
    "min": dict(variant="min", min_size_frac=0.03),
    "fixed_min_size": dict(min_size=12),
    "schedule": dict(min_size_frac_schedule=(0.03, 0.01)),
}

SEARCH_SPACE = {
    "theta_filter": [0.2, 0.3],
    "min_size_frac": [0.03, 0.05],
    "max_depth": [4, 6],
    "variant": ["max", "mean", "harmonic"],
}


def make_corpus(n_annotators_per_text=None, seed=11):
    """With n_annotators_per_text set, each text gets a random annotator
    subset, so subgroup sizes are unequal (as in real data). The full pool
    is a full factorial: every subgroup is perfectly balanced."""
    pool = AnnotatorPool(DEFAULT_DIMENSIONS, 5, (0.3, 1.0),
                         {0: .15, 1: .30, 2: .25, 3: .20, 4: .10}, 2)
    return pool.generate_dataset(n_texts=N_TEXTS, n_annotators_per_text=n_annotators_per_text,
                                 noise=0.05, seed=seed)


def to_plain(obj):
    """JSON-friendly copy: DataFrames become {columns, rows}, NaN becomes None."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.reset_index()
        return {"columns": [str(c) for c in obj.columns],
                "rows": [to_plain(list(r)) for r in obj.itertuples(index=False)]}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return [to_plain(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if obj != obj else float(obj)
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    return obj


def build_snapshot():
    corpus = make_corpus()
    data, gt = corpus.data, corpus.ground_truth

    snap = {
        "dataset": {
            "rows": len(data),
            "rating_counts": {str(t): g["rating"].value_counts().sort_index().to_dict()
                              for t, g in data.groupby("text_id")},
            "ground_truth": gt,
        },
    }

    for name, kwargs in PIPELINES.items():
        pipe = PolarizedTreesPipeline(**BASE, **kwargs)
        snap[name + "_gt"] = pipe.run_full_evaluation(data, ground_truth=gt, verbose=False)
        snap[name + "_nogt"] = PolarizedTreesPipeline(**BASE, **kwargs).run_full_evaluation(
            data, verbose=False)
        snap[name + "_trees"] = {str(t): [tree.get_root(), tree.get_leaves()]
                                 for t, tree in list(pipe.trees_.items())[:3]}

    # Unbalanced subgroups: size weighting (PEGweighted and every formulation built on it) only matters here.
    unbalanced = make_corpus(n_annotators_per_text=90, seed=5)
    for name in ("harmonic_rel", "weighted", "min", "mean", "schedule"):
        snap[name + "_unbalanced"] = PolarizedTreesPipeline(
            **BASE, **PIPELINES[name]).run_full_evaluation(
            unbalanced.data, ground_truth=unbalanced.ground_truth, verbose=False)

    bench = PolarizedTreesBenchmark(
        PolarizedTreesPipeline(**BASE), data, gt, search_space=SEARCH_SPACE,
        strategy="full", verbose=False,
        text_groups={t: "odd" if t % 2 else "even" for t in range(N_TEXTS)})
    bench.run()
    snap["bench_results"] = bench.get_results()
    snap["bench_best"] = bench.get_best_config()
    snap["bench_text"] = bench.get_text_results()

    return to_plain(snap)


def library_versions():
    from importlib.metadata import version

    return {name: version(name) for name in ("ndfu", "numpy", "pandas", "scikit-learn")}


if __name__ == "__main__":
    snapshot = build_snapshot()
    snapshot["_versions"] = library_versions()
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(snapshot, f, sort_keys=True)
