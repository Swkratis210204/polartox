import pytest

from polartox.datagen import AnnotatorPool, DEFAULT_DIMENSIONS, DEFAULT_INTENSITY_RANGE


def test_ndfu_available_and_importable():
    pytest.importorskip("ndfu")
    from ndfu import dfu, pdf
    assert callable(dfu)
    assert callable(pdf)


def test_k0_negative_control_scores_low_with_real_ndfu():
    pytest.importorskip("ndfu")
    from ndfu import dfu, pdf

    pool = AnnotatorPool(
        dimensions=DEFAULT_DIMENSIONS,
        scale=5,
        intensity_range=DEFAULT_INTENSITY_RANGE,
        depth_weights={0: 1.0},
        annotators_per_identity=10,
    )
    dataset, ground_truth = pool.generate_dataset(n_texts=5, n_annotators_per_text=None, noise=0.0, seed=0)

    for text_id, group in dataset.groupby("text_id"):
        hist = pdf(group["rating"].tolist(), range(1, pool.scale + 1))
        score = dfu(hist)
        assert score < 0.2


def test_high_intensity_single_dim_scores_high_with_real_ndfu():
    pytest.importorskip("ndfu")
    from ndfu import dfu, pdf

    pool = AnnotatorPool(
        dimensions=DEFAULT_DIMENSIONS,
        scale=5,
        intensity_range=(0.9, 1.0),
        depth_weights={1: 1.0},
        annotators_per_identity=10,
    )
    dataset, ground_truth = pool.generate_dataset(n_texts=5, n_annotators_per_text=None, noise=0.0, seed=0)

    for text_id, group in dataset.groupby("text_id"):
        hist = pdf(group["rating"].tolist(), range(1, pool.scale + 1))
        score = dfu(hist)
        assert score > 0.5