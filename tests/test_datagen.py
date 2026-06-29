import pytest
from toxpol.datagen import AnnotatorPool, DEFAULT_DIMENSIONS

SMALL_DIMS = {
    "politics": ["left", "center", "right"],
    "age": ["<25", ">25"],
}


def test_pool_size():
    pool = AnnotatorPool(SMALL_DIMS, annotators_per_identity=5)
    assert pool.pool_size == 6 * 5
    assert pool.n_identities == 6


def test_exclude_reduces_dimensions():
    pool = AnnotatorPool(SMALL_DIMS, exclude=["age"])
    assert "age" not in pool.active_dims
    assert "politics" in pool.active_dims
    assert pool.n_identities == 3


def test_generate_dataset_shape():
    pool = AnnotatorPool(SMALL_DIMS, annotators_per_identity=10)
    dataset, _ = pool.generate_dataset(n_texts=5, n_annotators_per_text=10)
    assert len(dataset) == 5 * 10


def test_dataset_columns():
    pool = AnnotatorPool(SMALL_DIMS)
    dataset, _ = pool.generate_dataset(n_texts=2, n_annotators_per_text=5)
    assert "text_id" in dataset.columns
    assert "rating" in dataset.columns
    for dim in SMALL_DIMS:
        assert dim in dataset.columns


def test_ratings_within_scale():
    pool = AnnotatorPool(SMALL_DIMS, scale=5)
    dataset, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=10)
    assert dataset["rating"].between(1, 5).all()


def test_ratings_within_scale_custom():
    pool = AnnotatorPool(SMALL_DIMS, scale=7, toxic_range=(6, 7), civil_range=(1, 2))
    dataset, _ = pool.generate_dataset(n_texts=5, n_annotators_per_text=10)
    assert dataset["rating"].between(1, 7).all()


def test_n_annotators_exceeds_pool_raises():
    pool = AnnotatorPool(SMALL_DIMS, annotators_per_identity=5)
    with pytest.raises(ValueError, match="exceeds pool size"):
        pool.generate_dataset(n_texts=1, n_annotators_per_text=pool.pool_size + 1)


def test_invalid_ratio_sum_raises():
    pool = AnnotatorPool(SMALL_DIMS)
    with pytest.raises(AssertionError, match="sum to 1.0"):
        pool.generate_dataset(
            n_texts=2, n_annotators_per_text=5,
            high_ratio=0.5, moderate_ratio=0.3, low_ratio=0.3,
        )


def test_bias_configs_keys_match_text_ids():
    pool = AnnotatorPool(SMALL_DIMS)
    n = 5
    dataset, bias_configs = pool.generate_dataset(n_texts=n, n_annotators_per_text=5)
    assert set(bias_configs.keys()) == set(range(n))


def test_bias_config_tiers_are_valid():
    pool = AnnotatorPool(SMALL_DIMS)
    _, bias_configs = pool.generate_dataset(n_texts=10, n_annotators_per_text=5)
    for text_id, cfg in bias_configs.items():
        assert cfg["tier"] in ("high", "moderate", "low")
        if cfg["tier"] in ("high", "moderate") or cfg.get("subcase") == "weighted":
            assert cfg["config"] is not None
            assert "threshold" in cfg
            assert set(cfg["config"].keys()) == set(pool.active_dims.keys())
        else:
            assert cfg["config"] is None
            assert "peak" in cfg
            assert "spread" in cfg


def test_noise_zero_ratings_in_range():
    pool = AnnotatorPool(SMALL_DIMS, scale=5, toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3))
    dataset, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=10, noise=0.0)
    assert dataset["rating"].between(1, 5).all()


def test_text_ids_are_correct():
    pool = AnnotatorPool(SMALL_DIMS)
    n = 7
    dataset, _ = pool.generate_dataset(n_texts=n, n_annotators_per_text=5)
    assert set(dataset["text_id"].unique()) == set(range(n))


def test_default_dimensions():
    pool = AnnotatorPool(DEFAULT_DIMENSIONS)
    assert pool.n_identities == 162
    assert pool.pool_size == 1620


def test_tier_ratios_respected_approximately():
    pool = AnnotatorPool(DEFAULT_DIMENSIONS)
    n = 200
    _, bias_configs = pool.generate_dataset(
        n_texts=n, n_annotators_per_text=50,
        high_ratio=0.6, moderate_ratio=0.2, low_ratio=0.2,
    )
    tiers = [cfg["tier"] for cfg in bias_configs.values()]
    high_share = tiers.count("high") / n
    assert 0.45 < high_share < 0.75  # generous tolerance for a stochastic draw


def test_severity_ordering_high_gt_moderate_gt_low():
    pool = AnnotatorPool(DEFAULT_DIMENSIONS)
    dataset, bias_configs = pool.generate_dataset(
        n_texts=150, n_annotators_per_text=100,
        high_ratio=0.6, moderate_ratio=0.2, low_ratio=0.2,
    )
    results = pool.analyze(dataset, bias_configs)
    tier_scores = {"high": [], "moderate": [], "low": []}
    for text_id, cfg in bias_configs.items():
        tier_scores[cfg["tier"]].append(results[text_id]["overall"])
    import numpy as np
    means = {t: np.mean(s) for t, s in tier_scores.items() if s}
    assert means["high"] > means["moderate"] > means["low"]