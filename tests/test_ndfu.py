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
        # Odd-cardinality dims (e.g. 3 values) split 2:1, capping dfu near
        # 0.5; only even-cardinality dims (e.g. "orientation") reach ~1.0.
        # 0.4 stays well above the k=0 control's < 0.2 while tolerating both.
        assert score > 0.4


# ---------------------------------------------------------------------------
# The two nDFU code paths must agree: ndfu_score (corpus filter, compute_peg)
# and _ndfu_from_counts over _histograms (tree-building loop).
# ---------------------------------------------------------------------------

import numpy as np

from polartox.polarized_tree import _histograms, _ndfu_from_counts, ndfu_score


def _via_counts(ratings, scale):
    ratings = np.asarray(ratings, dtype=float)
    n = len(ratings)
    counts = _histograms(np.zeros(n, dtype=np.int64), ratings, 1, scale)[0]
    return _ndfu_from_counts(tuple(counts.tolist()), n)


def test_ndfu_paths_agree_on_random_ratings():
    pytest.importorskip("ndfu")
    rng = np.random.default_rng(0)
    for _ in range(300):
        ratings = rng.integers(1, 6, size=int(rng.integers(1, 60)))
        assert _via_counts(ratings, 5) == pytest.approx(ndfu_score(ratings, 5), abs=1e-12)


@pytest.mark.parametrize("ratings", [
    [3],                        # single rating
    [2] * 10,                   # all identical
    [1] * 5 + [5] * 5,          # perfectly bimodal
    [0, 1, 2, 6, 7],            # some ratings outside 1..scale
    [1.5, 2, 3],                # a non-integer rating
])
def test_ndfu_paths_agree_on_edge_cases(ratings):
    pytest.importorskip("ndfu")
    assert _via_counts(ratings, 5) == pytest.approx(ndfu_score(ratings, 5), abs=1e-12)


def test_ndfu_paths_agree_on_empty_input():
    pytest.importorskip("ndfu")
    assert np.isnan(ndfu_score([], 5))
    assert np.isnan(_via_counts([], 5))


def test_ndfu_paths_both_raise_when_every_rating_is_out_of_range():
    pytest.importorskip("ndfu")
    with pytest.raises(ValueError):
        ndfu_score([0, 9], 5)
    with pytest.raises(ValueError):
        _via_counts([0, 9], 5)
