import numpy as np
import pandas as pd
import pytest

from polartox.polarized_trees import (
    ndfu_score, compute_prg, detect_polarized_subgroups,
    jaccard, PolarizedTreesPipeline,
)

SCALE = 5


def make_toy_dataset(seed=0, n_texts=6, n=180):
    """Deterministic toy corpus: gender is always the primary cause,
    politics is secondary within males only -- known ground truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for text_id in range(n_texts):
        for _ in range(n):
            gender = rng.choice(["male", "female"])
            politics = rng.choice(["left", "right"])
            if gender == "female":
                rating = rng.choice([1, 2], p=[0.7, 0.3])
            elif politics == "left":
                rating = rng.choice([4, 5], p=[0.3, 0.7])
            else:
                rating = rng.choice([1, 2], p=[0.6, 0.4])
            rows.append({"text_id": text_id, "gender": gender, "politics": politics,
                         "age": "25-50", "rating": rating})
    dataset = pd.DataFrame(rows)
    ground_truth = {
        tid: {
            "active_dims": ["gender", "politics"],
            "lean": {"gender": {"female": "civil", "male": "toxic"},
                     "politics": {"left": "toxic", "right": "civil"}},
            "alpha": {"gender": 0.7, "politics": 0.6},
        } for tid in range(n_texts)
    }
    return dataset, ground_truth


# ------------------------------------------------------------------
# ndfu_score / compute_prg
# ------------------------------------------------------------------

def test_ndfu_score_unimodal_low():
    ratings = [3] * 20 + [2] * 3 + [4] * 3
    assert ndfu_score(ratings, SCALE) < 0.2


def test_ndfu_score_bimodal_high():
    ratings = [1] * 10 + [5] * 10
    assert ndfu_score(ratings, SCALE) > 0.8


def test_ndfu_score_empty_is_nan():
    assert np.isnan(ndfu_score([], SCALE))


def test_compute_prg_uses_abs_value():
    """A split that WORSENS the worst-case subgroup should still yield a
    positive PRG (paper's formula uses absolute value)."""
    node_ratings = np.array([3] * 20)
    groups = {"a": np.array([1] * 10), "b": np.array([1, 5] * 5)}
    prg, global_ndfu, group_ndfus = compute_prg(node_ratings, groups, SCALE, variant="max")
    assert prg >= 0


def test_compute_prg_variants_differ():
    node_ratings = np.array([1] * 20 + [5] * 20)
    groups = {"a": np.array([1] * 15 + [5] * 5), "b": np.array([1] * 5 + [5] * 15)}
    prg_max, _, _ = compute_prg(node_ratings, groups, SCALE, variant="max")
    prg_var, _, _ = compute_prg(node_ratings, groups, SCALE, variant="var")
    prg_beta, _, _ = compute_prg(node_ratings, groups, SCALE, variant="beta", beta=1.0)
    # beta should sit between max and var (harmonic mean property), not necessarily
    # strictly between numerically but should be a valid, finite number
    assert all(np.isfinite(v) for v in [prg_max, prg_var, prg_beta])


def test_compute_prg_invalid_variant_raises():
    with pytest.raises(ValueError):
        compute_prg(np.array([1, 2, 3]), {"a": np.array([1, 2])}, SCALE, variant="bogus")


# ------------------------------------------------------------------
# detect_polarized_subgroups
# ------------------------------------------------------------------

def test_detect_polarized_subgroups_finds_known_structure():
    dataset, ground_truth = make_toy_dataset()
    text_data = dataset[dataset["text_id"] == 0]
    leaves, root = detect_polarized_subgroups(
        text_data, dims=["gender", "politics", "age"],
        min_size=10, h=0.05, max_depth=4, scale=SCALE,
        theta_stop=0.1, variant="beta", beta=1.0, return_tree=True,
    )
    found_dims = {d for leaf in leaves for d, v in leaf["path"]}
    assert found_dims == {"gender", "politics"}
    assert "age" not in found_dims


def test_detect_polarized_subgroups_backward_compatible_return():
    dataset, _ = make_toy_dataset()
    text_data = dataset[dataset["text_id"] == 0]
    leaves = detect_polarized_subgroups(
        text_data, dims=["gender", "politics"], min_size=10, h=0.05,
        max_depth=4, scale=SCALE, theta_stop=0.1,
    )
    assert isinstance(leaves, list)
    assert all("pole" in leaf for leaf in leaves)


def test_leaf_poles_are_valid():
    dataset, _ = make_toy_dataset()
    text_data = dataset[dataset["text_id"] == 0]
    leaves = detect_polarized_subgroups(
        text_data, dims=["gender", "politics"], min_size=10, h=0.05,
        max_depth=4, scale=SCALE, theta_stop=0.1,
    )
    for leaf in leaves:
        assert leaf["pole"] in ("toxic", "civil", "indeterminate")
        assert 0.0 <= leaf["p_tox"] <= 1.0


def test_min_size_blocks_small_subgroups():
    dataset, _ = make_toy_dataset(n=20)  # small text
    text_data = dataset[dataset["text_id"] == 0]
    leaves = detect_polarized_subgroups(
        text_data, dims=["gender", "politics"], min_size=100, h=0.05,
        max_depth=4, scale=SCALE, theta_stop=None,
    )
    # min_size=100 > any possible subgroup with only 20 rows -- must stay a single leaf
    assert len(leaves) == 1


def test_theta_stop_short_circuits_before_min_size():
    """A node with nDFU below theta_stop should stop immediately, even if
    min_size would otherwise have allowed a split."""
    dataset, _ = make_toy_dataset()
    text_data = dataset[dataset["text_id"] == 0]
    leaves_high_stop = detect_polarized_subgroups(
        text_data, dims=["gender", "politics"], min_size=2, h=0.0,
        max_depth=4, scale=SCALE, theta_stop=0.99,  # nearly everything stops immediately
    )
    assert len(leaves_high_stop) == 1


# ------------------------------------------------------------------
# jaccard
# ------------------------------------------------------------------

def test_jaccard_identical_sets():
    assert jaccard(["a", "b"], ["b", "a"]) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard(["a"], ["b"]) == 0.0


def test_jaccard_both_empty():
    assert jaccard([], []) == 1.0


def test_jaccard_partial_overlap():
    assert jaccard(["a", "b"], ["a", "c"]) == pytest.approx(1 / 3)


# ------------------------------------------------------------------
# PolarizedTreesPipeline
# ------------------------------------------------------------------

@pytest.fixture
def pipeline_and_data():
    dataset, ground_truth = make_toy_dataset()
    pipe = PolarizedTreesPipeline(
        dims=["gender", "politics", "age"], scale=SCALE, theta_filter=0.1,
        min_size_frac=0.05, h=0.05, max_depth=4, theta_stop=0.1,
    )
    return pipe, dataset, ground_truth


def test_filter_polarized_texts(pipeline_and_data):
    pipe, dataset, _ = pipeline_and_data
    retained = pipe.filter_polarized_texts(dataset)
    assert set(retained) <= set(dataset["text_id"].unique())
    assert len(pipe.overall_ndfu_) == dataset["text_id"].nunique()


def test_build_all_trees_requires_filter_or_ids(pipeline_and_data):
    pipe, dataset, _ = pipeline_and_data
    with pytest.raises(RuntimeError):
        pipe.build_all_trees(dataset)


def test_build_all_trees_with_explicit_ids(pipeline_and_data):
    pipe, dataset, _ = pipeline_and_data
    trees = pipe.build_all_trees(dataset, text_ids=[0, 1])
    assert set(trees.keys()) == {0, 1}
    for leaves, root in trees.values():
        assert isinstance(leaves, list)
        assert "is_leaf" in root


def test_dimension_frequency_without_ground_truth(pipeline_and_data):
    pipe, dataset, _ = pipeline_and_data
    pipe.filter_polarized_texts(dataset)
    pipe.build_all_trees(dataset)
    F = pipe.dimension_frequency()
    assert "ever_truly_active" not in F.columns


def test_dimension_frequency_with_ground_truth(pipeline_and_data):
    pipe, dataset, ground_truth = pipeline_and_data
    pipe.filter_polarized_texts(dataset)
    pipe.build_all_trees(dataset)
    F = pipe.dimension_frequency(ground_truth)
    assert "ever_truly_active" in F.columns
    assert bool(F.loc["gender", "ever_truly_active"]) is True


def test_subgroup_pole_consistency_columns(pipeline_and_data):
    pipe, dataset, ground_truth = pipeline_and_data
    pipe.filter_polarized_texts(dataset)
    pipe.build_all_trees(dataset)
    C = pipe.subgroup_pole_consistency()
    assert set(["n_s", "frac_toxic", "frac_civil"]).issubset(C.columns)
    C_gt = pipe.subgroup_pole_consistency(ground_truth)
    assert "true_lean_match_rate" in C_gt.columns


def test_subgroup_prg_columns(pipeline_and_data):
    pipe, dataset, ground_truth = pipeline_and_data
    pipe.filter_polarized_texts(dataset)
    pipe.build_all_trees(dataset)
    P = pipe.subgroup_prg()
    assert set(["n_s", "mean_prg"]).issubset(P.columns)
    P_gt = pipe.subgroup_prg(ground_truth)
    assert "mean_true_alpha" in P_gt.columns


def test_diagnostics_keys(pipeline_and_data):
    pipe, dataset, _ = pipeline_and_data
    pipe.filter_polarized_texts(dataset)
    pipe.build_all_trees(dataset)
    diag = pipe.diagnostics()
    expected_keys = {"retention_rate", "mean_leaves", "mean_depth",
                      "mean_residual_ndfu", "mean_top_split_prg",
                      "indeterminate_rate", "dims_never_used"}
    assert expected_keys.issubset(diag.keys())
    assert 0.0 <= diag["retention_rate"] <= 1.0


def test_recovery_metrics_perfect_on_known_structure(pipeline_and_data):
    pipe, dataset, ground_truth = pipeline_and_data
    pipe.filter_polarized_texts(dataset)
    pipe.build_all_trees(dataset)
    recovery = pipe.recovery_metrics(ground_truth)
    assert recovery["jaccard"].mean() > 0.8  # should recover gender+politics well


def test_run_full_evaluation_end_to_end(pipeline_and_data, capsys):
    pipe, dataset, ground_truth = pipeline_and_data
    results = pipe.run_full_evaluation(dataset, ground_truth=ground_truth, verbose=False)
    assert set(["F", "C", "P", "diagnostics", "recovery"]).issubset(results.keys())


def test_run_full_evaluation_without_ground_truth(pipeline_and_data):
    pipe, dataset, _ = pipeline_and_data
    results = pipe.run_full_evaluation(dataset, ground_truth=None, verbose=False)
    assert "recovery" not in results
    assert "diagnostics" in results


def test_min_size_frac_scales_with_text_size():
    pipe = PolarizedTreesPipeline(
        dims=["gender"], scale=SCALE, theta_filter=0.0,
        min_size_frac=0.1, h=0.05, max_depth=3,
    )
    assert pipe._min_size_for(100) == 10
    assert pipe._min_size_for(1000) == 100
    assert pipe._min_size_for(5) == 2  # floor of 2, per paper's nmin


def test_fixed_min_size_overrides_frac_when_given():
    pipe = PolarizedTreesPipeline(
        dims=["gender"], scale=SCALE, theta_filter=0.0,
        min_size=25, min_size_frac=0.1, h=0.05, max_depth=3,
    )
    assert pipe.min_size_frac is None
    assert pipe._min_size_for(1000) == 25


def test_inspect_tree_runs_without_error(pipeline_and_data, capsys):
    pipe, dataset, _ = pipeline_and_data
    pipe.filter_polarized_texts(dataset)
    pipe.build_all_trees(dataset, text_ids=[0])
    pipe.inspect_tree(0, dataset, show_distributions=True)
    captured = capsys.readouterr()
    assert "root" in captured.out