"""
Adjusted Rand index between trees: adjusted_rand_index, PolarizedTree.leaf_labels
and pairwise_ari (how much different PEG formulations agree on the groups).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ndfu")

from test_trees import make_toy_dataset

from polartox import (
    PolarizedTree,
    PolarizedTreesPipeline,
    adjusted_rand_index,
    pairwise_ari,
)

DIMS = ["gender", "politics", "age"]


# ---------------------------------------------------------------------
# adjusted_rand_index
# ---------------------------------------------------------------------

def test_worked_example():
    # {0,1 | 2,3} against {0,1 | 2 | 3}. Ordered pairs: both 2, only a 2, only b 0,
    # neither 8, so ARI = 2 * (2*8 - 2*0) / ((2+2)*(2+8) + (2+0)*(0+8)) = 32/56 = 4/7.
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4 / 7)


def test_identical_partitions_are_1_whatever_the_label_names():
    assert adjusted_rand_index([0, 0, 1, 1, 2], ["x", "x", "y", "y", "z"]) == 1.0


def test_symmetric_and_label_independent():
    a = [0, 0, 1, 1, 2, 2, 2]
    b = [1, 0, 0, 1, 1, 2, 2]
    assert adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_index(b, a))
    assert adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_index([5 - x for x in a], b))


def test_single_group_cases():
    assert adjusted_rand_index([0, 0, 0, 0], [7, 7, 7, 7]) == 1.0          # both trivial: identical
    assert adjusted_rand_index([0, 0, 0, 0], [0, 0, 1, 1]) == 0.0          # trivial against a split


def test_no_agreement_beyond_chance_is_about_zero():
    rng = np.random.default_rng(0)
    values = [adjusted_rand_index(rng.integers(0, 4, 400), rng.integers(0, 4, 400)) for _ in range(30)]
    assert abs(np.mean(values)) < 0.02


def test_tiny_inputs():
    assert adjusted_rand_index([], []) == 1.0
    assert adjusted_rand_index([3], [9]) == 1.0


@pytest.mark.parametrize("a, b", [([0, 1], [0, 1, 2]), ([[0, 1]], [[0, 1]])])
def test_invalid_shapes_raise(a, b):
    with pytest.raises(ValueError):
        adjusted_rand_index(a, b)


def test_matches_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(2, 60))
        a = rng.integers(0, int(rng.integers(1, 6)), n)
        b = rng.integers(0, int(rng.integers(1, 6)), n)
        assert adjusted_rand_index(a, b) == pytest.approx(metrics.adjusted_rand_score(a, b), abs=1e-12)


# ---------------------------------------------------------------------
# PolarizedTree.leaf_labels
# ---------------------------------------------------------------------

def build_tree(data, text_id=0, **overrides):
    settings = dict(dims=DIMS, min_size=10, h=0.05, max_depth=4, scale=5, theta_stop=0.1, text_id=text_id)
    settings.update(overrides)
    return PolarizedTree.build(data, **settings)


def test_leaf_labels_partition_the_annotators_into_the_leaves():
    dataset, _ = make_toy_dataset()
    text = dataset[dataset["text_id"] == 0]
    tree = build_tree(text)
    labels = tree.leaf_labels(dataset)

    assert labels.shape == (len(text),)
    assert (labels >= 0).all()
    assert sorted(np.unique(labels)) == list(range(tree.n_leaves))

    for k, leaf in enumerate(tree.get_leaves()):
        assert (labels == k).sum() == leaf["n"]
        assert sorted(text["rating"].to_numpy()[labels == k]) == sorted(tree.node_ratings(dataset, leaf["path"]))


def test_leaf_labels_of_a_tree_that_never_split_are_all_the_same():
    flat = pd.DataFrame({"text_id": 0, "gender": ["m", "f"] * 20, "rating": [3] * 40})
    tree = build_tree(flat, dims=["gender"])
    assert tree.n_leaves == 1
    assert (tree.leaf_labels(flat) == 0).all()


def test_rows_without_a_leaf_get_minus_one():
    dataset, _ = make_toy_dataset()
    text = dataset[dataset["text_id"] == 0].copy()
    text.loc[text.index[:7], "gender"] = None
    tree = build_tree(text, dims=["gender"])             # the only split is on gender, so rows with no gender drop out
    assert not tree.get_root()["is_leaf"]
    labels = tree.leaf_labels(text)
    assert (labels == -1).sum() == 7
    assert (labels[7:] >= 0).all()


def test_labels_use_only_the_rows_of_the_trees_own_text():
    dataset, _ = make_toy_dataset()
    tree = build_tree(dataset[dataset["text_id"] == 2], text_id=2)
    assert len(tree.leaf_labels(dataset)) == (dataset["text_id"] == 2).sum()


# ---------------------------------------------------------------------
# pairwise_ari
# ---------------------------------------------------------------------

def fitted(dataset, variant):
    pipe = PolarizedTreesPipeline(dims=DIMS, scale=5, theta_filter=0.1, h=0.05, max_depth=4,
                                  min_size_frac=0.05, theta_stop=0.1, variant=variant)
    pipe.run_full_evaluation(dataset, verbose=False)
    return pipe


def test_pairwise_ari_shape_symmetry_and_range():
    dataset, _ = make_toy_dataset(n_texts=5)
    pipes = {v: fitted(dataset, v) for v in ("max", "weighted", "harmonic")}
    matrix, per_text = pairwise_ari(pipes, dataset)

    assert list(matrix.index) == list(matrix.columns) == ["max", "weighted", "harmonic"]
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert ((matrix >= -1) & (matrix <= 1)).all().all()
    assert len(per_text) == len(pipes["max"].trees_) * 3          # texts x pairs
    assert set(per_text.columns) == {"text_id", "a", "b", "ari"}


def test_identical_pipelines_agree_completely():
    dataset, _ = make_toy_dataset(n_texts=4)
    matrix, per_text = pairwise_ari({"one": fitted(dataset, "weighted"), "two": fitted(dataset, "weighted")}, dataset)
    assert matrix.loc["one", "two"] == 1.0
    assert (per_text["ari"] == 1.0).all()


def test_pairwise_ari_matches_a_direct_computation():
    dataset, _ = make_toy_dataset(n_texts=3)
    pipes = {v: fitted(dataset, v) for v in ("max", "min")}
    matrix, per_text = pairwise_ari(pipes, dataset)

    expected = [
        adjusted_rand_index(pipes["max"].trees_[t].leaf_labels(dataset), pipes["min"].trees_[t].leaf_labels(dataset))
        for t in sorted(pipes["max"].trees_)
    ]
    assert per_text["ari"].tolist() == pytest.approx(expected)
    assert matrix.loc["max", "min"] == pytest.approx(np.mean(expected))


def test_only_texts_that_every_pipeline_analysed_are_compared():
    dataset, _ = make_toy_dataset(n_texts=4)
    loose, strict = fitted(dataset, "weighted"), fitted(dataset, "weighted")
    strict.trees_ = {t: tree for t, tree in strict.trees_.items() if t != 0}       # one text dropped
    _, per_text = pairwise_ari({"loose": loose, "strict": strict}, dataset)
    assert 0 not in set(per_text["text_id"])
    assert set(per_text["text_id"]) == set(strict.trees_)
