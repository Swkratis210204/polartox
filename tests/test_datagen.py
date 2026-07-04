import numpy as np
import pandas as pd
import pytest

from polartox.datagen import AnnotatorPool, DEFAULT_DIMENSIONS, DEFAULT_DEPTH_WEIGHTS, DEFAULT_INTENSITY_RANGE

SMALL_DIMS = {
    "politics": ["left", "center", "right"],
    "age": ["<25", ">25"],
}

SCALE = 5
INTENSITY_RANGE = (0.3, 1.0)
DEPTH_WEIGHTS_SMALL = {0: 0.2, 1: 0.4, 2: 0.4}


def make_pool(**overrides):
    kwargs = dict(
        dimensions=SMALL_DIMS,
        scale=SCALE,
        intensity_range=INTENSITY_RANGE,
        depth_weights=DEPTH_WEIGHTS_SMALL,
        annotators_per_identity=5,
    )
    kwargs.update(overrides)
    return AnnotatorPool(**kwargs)


def test_pool_size():
    pool = make_pool(annotators_per_identity=5)
    assert pool.pool_size == 6 * 5
    assert pool.n_identities == 6


def test_generate_dataset_shape():
    pool = make_pool()
    result = pool.generate_dataset(n_texts=5, n_annotators_per_text=10, seed=0)
    dataset, ground_truth = result
    assert len(dataset) == 5 * 10
    assert len(ground_truth) == 5


def test_dataset_columns():
    pool = make_pool()
    dataset, _ = pool.generate_dataset(n_texts=2, n_annotators_per_text=5, seed=0)
    assert "text_id" in dataset.columns
    assert "annotator_id" in dataset.columns
    assert "rating" in dataset.columns
    for dim in SMALL_DIMS:
        assert dim in dataset.columns


def test_ratings_within_scale():
    pool = make_pool()
    dataset, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=10, seed=1)
    assert dataset["rating"].between(1, SCALE).all()


def test_ratings_within_scale_custom():
    pool = make_pool(scale=7, intensity_range=(0.3, 1.0))
    dataset, _ = pool.generate_dataset(n_texts=5, n_annotators_per_text=10, seed=2)
    assert dataset["rating"].between(1, 7).all()


def test_n_annotators_exceeding_pool_uses_full_pool():
    pool = make_pool(annotators_per_identity=5)
    dataset, _ = pool.generate_dataset(n_texts=1, n_annotators_per_text=pool.pool_size + 1000, seed=0)
    assert len(dataset) == pool.pool_size


def test_none_annotators_per_text_uses_full_pool():
    pool = make_pool(annotators_per_identity=5)
    dataset, _ = pool.generate_dataset(n_texts=3, n_annotators_per_text=None, seed=0)
    assert len(dataset) == 3 * pool.pool_size


def test_ground_truth_keys_match_text_ids():
    pool = make_pool()
    n = 5
    _, ground_truth = pool.generate_dataset(n_texts=n, n_annotators_per_text=5, seed=0)
    assert set(ground_truth.keys()) == set(range(n))


def test_ground_truth_structure_is_valid():
    pool = make_pool()
    _, ground_truth = pool.generate_dataset(n_texts=30, n_annotators_per_text=5, seed=3)
    for text_id, cfg in ground_truth.items():
        assert "active_dims" in cfg
        if cfg["active_dims"]:
            assert "lean" in cfg and "alpha" in cfg
            assert set(cfg["lean"].keys()) == set(cfg["active_dims"])
            assert set(cfg["alpha"].keys()) == set(cfg["active_dims"])
            for dim in cfg["active_dims"]:
                assert set(cfg["lean"][dim].keys()) == set(SMALL_DIMS[dim])
                assert set(cfg["lean"][dim].values()) <= {"toxic", "civil"}
                alpha_min, alpha_max = INTENSITY_RANGE
                assert alpha_min <= cfg["alpha"][dim] <= alpha_max
        else:
            assert "peak" in cfg and "spread" in cfg
            assert 1 <= cfg["peak"] <= SCALE


def test_noise_zero_ratings_in_range():
    pool = make_pool()
    dataset, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=10, noise=0.0, seed=0)
    assert dataset["rating"].between(1, SCALE).all()


def test_text_ids_are_correct():
    pool = make_pool()
    n = 7
    dataset, _ = pool.generate_dataset(n_texts=n, n_annotators_per_text=5, seed=0)
    assert set(dataset["text_id"].unique()) == set(range(n))


def test_default_dimensions():
    pool = AnnotatorPool(
        dimensions=DEFAULT_DIMENSIONS,
        scale=5,
        intensity_range=DEFAULT_INTENSITY_RANGE,
        depth_weights=DEFAULT_DEPTH_WEIGHTS,
        annotators_per_identity=10,
    )
    assert pool.n_identities == 162
    assert pool.pool_size == 1620


def test_mandatory_args_enforced():
    with pytest.raises(TypeError):
        AnnotatorPool(dimensions=DEFAULT_DIMENSIONS)


def test_intensity_range_out_of_bounds_raises():
    with pytest.raises(AssertionError):
        make_pool(intensity_range=(-0.1, 1.0))
    with pytest.raises(AssertionError):
        make_pool(intensity_range=(0.0, 1.5))
    with pytest.raises(AssertionError):
        make_pool(intensity_range=(0.8, 0.2))


def test_depth_weights_respected_approximately():
    pool = AnnotatorPool(
        dimensions=DEFAULT_DIMENSIONS,
        scale=5,
        intensity_range=DEFAULT_INTENSITY_RANGE,
        depth_weights=DEFAULT_DEPTH_WEIGHTS,
        annotators_per_identity=10,
    )
    n = 500
    _, ground_truth = pool.generate_dataset(n_texts=n, n_annotators_per_text=50, seed=0)
    k_values = [len(cfg["active_dims"]) for cfg in ground_truth.values()]
    k1_share = k_values.count(1) / n
    assert 0.10 < k1_share < 0.35


def test_k0_is_near_zero_polarization():
    pool = AnnotatorPool(
        dimensions=DEFAULT_DIMENSIONS,
        scale=5,
        intensity_range=DEFAULT_INTENSITY_RANGE,
        depth_weights={0: 1.0},
        annotators_per_identity=10,
    )
    dataset, ground_truth = pool.generate_dataset(n_texts=5, n_annotators_per_text=None, noise=0.0, seed=0)
    for text_id in ground_truth:
        assert ground_truth[text_id]["active_dims"] == []
    for text_id, group in dataset.groupby("text_id"):
        assert group["rating"].std() < 2.0


def test_seed_reproducibility():
    pool = make_pool()
    d1, gt1 = pool.generate_dataset(n_texts=10, n_annotators_per_text=20, seed=7)
    d2, gt2 = pool.generate_dataset(n_texts=10, n_annotators_per_text=20, seed=7)
    pd.testing.assert_frame_equal(d1, d2)
    assert gt1.keys() == gt2.keys()


def test_different_seeds_produce_different_data():
    pool = make_pool()
    d1, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=20, seed=1)
    d2, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=20, seed=2)
    assert not d1["rating"].equals(d2["rating"])


def test_generated_dataset_unpacks_as_tuple():
    pool = make_pool()
    result = pool.generate_dataset(n_texts=3, n_annotators_per_text=5, seed=0)
    dataset, ground_truth = result
    assert isinstance(dataset, pd.DataFrame)
    assert isinstance(ground_truth, dict)


def test_generated_dataset_head_tail_sample():
    pool = make_pool()
    result = pool.generate_dataset(n_texts=5, n_annotators_per_text=10, seed=0)
    assert len(result.head(3)) == 3
    assert len(result.tail(2)) == 2
    assert len(result.sample(4, random_state=0)) == 4


def test_generated_dataset_text_ids_by_k():
    pool = AnnotatorPool(
        dimensions=DEFAULT_DIMENSIONS,
        scale=5,
        intensity_range=DEFAULT_INTENSITY_RANGE,
        depth_weights=DEFAULT_DEPTH_WEIGHTS,
        annotators_per_identity=10,
    )
    result = pool.generate_dataset(n_texts=200, n_annotators_per_text=50, seed=0)
    for k in range(5):
        ids = result.text_ids_by_k(k)
        for tid in ids:
            assert len(result.ground_truth[tid]["active_dims"]) == k


def test_generated_dataset_describe_text_runs_without_error(capsys):
    pool = make_pool()
    result = pool.generate_dataset(n_texts=5, n_annotators_per_text=10, seed=0)
    for text_id in result.ground_truth:
        result.describe_text(text_id)
    captured = capsys.readouterr()
    assert "Text" in captured.out


def test_generated_dataset_len_matches_dataframe():
    pool = make_pool()
    result = pool.generate_dataset(n_texts=4, n_annotators_per_text=10, seed=0)
    assert len(result) == len(result.data) == 40