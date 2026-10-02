"""
Regression guards for refactoring.

* test_matches_golden_snapshot: the real pipeline and benchmark on a fixed
  seeded corpus must reproduce tests/golden_snapshot.json, which was
  generated from the code before the refactors. The golden numbers depend
  on the versions of ndfu, numpy, pandas and scikit-learn (random streams
  and float arithmetic), so the comparison only runs when the installed
  versions equal the ones recorded in the golden file; otherwise it is
  skipped with a visible reason. Setting POLARTOX_REQUIRE_GOLDEN=1 turns
  that skip into a failure -- the pinned CI job sets it, so the comparison
  can never silently stop running there.
* The remaining tests cover paths the other test files do not reach: the
  depth schedule, the benchmark rebuilding a real pipeline, and parallel
  vs serial benchmark runs with the real pipeline.
"""

import json
import os
import pickle
from pathlib import Path

import pytest

pytest.importorskip("ndfu")

from _snapshot import BASE, build_snapshot, library_versions, make_corpus
from test_trees import make_toy_dataset

from polartox import (
    AnnotatorPool,
    DEFAULT_DIMENSIONS,
    PolarizedTreesBenchmark,
    PolarizedTreesPipeline,
    detect_polarized_subgroups,
)
from polartox.datagen import CONTROL_SPREAD_RANGE
from polartox.pipeline import _DepthSchedule
from polartox.polarized_tree import default_theta_pole

GOLDEN = json.loads((Path(__file__).parent / "golden_snapshot.json").read_text(encoding="utf-8"))


def assert_same(actual, expected, path="snapshot"):
    """Recursive equality; floats compared with a small tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), path
    else:
        assert actual == expected, path


GOLDEN_VERSIONS = GOLDEN.pop("_versions")


@pytest.fixture(scope="module")
def snapshot():
    now = library_versions()
    if now != GOLDEN_VERSIONS:
        reason = (f"golden snapshot was made with {GOLDEN_VERSIONS}, but {now} is installed; "
                  "install the recorded versions to run it (see .github/workflows)")
        if os.environ.get("POLARTOX_REQUIRE_GOLDEN"):
            pytest.fail(reason)
        pytest.skip(reason)
    return build_snapshot()


@pytest.mark.parametrize("section", sorted(GOLDEN))
def test_matches_golden_snapshot(snapshot, section):
    assert_same(snapshot[section], GOLDEN[section], section)


# ---------------------------------------------------------------------
# min_size schedule
# ---------------------------------------------------------------------

def test_depth_schedule_values_and_pickle():
    schedule = _DepthSchedule(0.03, 0.01)
    assert [schedule(d) for d in (1, 2, 5)] == pytest.approx([0.03, 0.04, 0.07])
    assert pickle.loads(pickle.dumps(schedule))(3) == schedule(3)


def test_pipeline_with_schedule_is_picklable_and_keeps_settings():
    pipe = PolarizedTreesPipeline(**BASE, min_size_frac_schedule=(0.03, 0.01))
    clone = pickle.loads(pickle.dumps(pipe))
    assert clone.min_size_frac_schedule == (0.03, 0.01)
    assert clone.min_size_frac is None
    assert clone.min_size(4) == pipe.min_size(4)


def test_callable_min_size_matches_equivalent_fixed_size():
    data, _ = make_toy_dataset(n_texts=1, n=180)
    kwargs = dict(dims=["gender", "politics"], h=0.05, max_depth=4, scale=5,
                  theta_stop=0.1, return_tree=True)
    # 10% of 180 annotators = 18 at every depth
    by_callable = detect_polarized_subgroups(data, min_size=lambda depth: 0.1, **kwargs)
    by_fixed = detect_polarized_subgroups(data, min_size=18, **kwargs)
    assert by_callable == by_fixed


# ---------------------------------------------------------------------
# Benchmark with the real pipeline
# ---------------------------------------------------------------------

def test_benchmark_rebuild_overrides_only_the_searched_parameters():
    data, gt = make_toy_dataset(n_texts=2)
    base = PolarizedTreesPipeline(**{**BASE, "dims": ["gender", "politics", "age"]},
                                  variant="max", relative_h=True,
                                  min_size_frac_schedule=(0.03, 0.01))
    bench = PolarizedTreesBenchmark(base, data, gt, search_space={"h": [0.1]},
                                    strategy="full", verbose=False)
    rebuilt = bench._build_pipeline({"h": 0.2})
    assert rebuilt.h == 0.2
    assert (rebuilt.variant, rebuilt.relative_h) == ("max", True)
    assert rebuilt.dims == ["gender", "politics", "age"]
    assert rebuilt.min_size_frac_schedule == (0.03, 0.01)
    assert rebuilt.min_size(4) == base.min_size(4)


def test_benchmark_parallel_matches_serial_with_real_pipeline_and_schedule():
    corpus = make_corpus()
    texts = corpus.data[corpus.data["text_id"] < 4]
    gt = {t: corpus.ground_truth[t] for t in range(4)}
    base = PolarizedTreesPipeline(**BASE, min_size_frac_schedule=(0.03, 0.01))
    space = {"theta_filter": [0.2, 0.3], "h": [0.1, 0.15]}

    def run(n_jobs):
        return PolarizedTreesBenchmark(base, texts, gt, search_space=space, strategy="full",
                                       verbose=False, n_jobs=n_jobs).run().get_results()

    serial, parallel = run(1), run(2)
    assert serial.equals(parallel)


# ---------------------------------------------------------------------
# Defaults that must stay in sync
# ---------------------------------------------------------------------

@pytest.mark.parametrize("scale, expected", [(5, 3), (7, 4), (10, 6)])
def test_default_theta_pole(scale, expected):
    assert default_theta_pole(scale) == expected


def test_pipeline_and_tree_resolve_the_same_default_theta_pole():
    data, _ = make_toy_dataset(n_texts=1)
    pipe = PolarizedTreesPipeline(dims=["gender", "politics"], scale=5, theta_filter=0.2,
                                  h=0.05, max_depth=3)
    pipe.filter_polarized_texts(data)
    pipe.build_all_trees(data)
    assert pipe.theta_pole == default_theta_pole(5)
    assert all(tree.theta_pole == default_theta_pole(5) for tree in pipe.trees_.values())


# ---------------------------------------------------------------------
# k=0 negative control
# ---------------------------------------------------------------------

def test_negative_control_spread_stays_in_the_fixed_range():
    pool = AnnotatorPool(DEFAULT_DIMENSIONS, 5, (0.3, 1.0), {0: 1.0}, 2)
    _, gt = pool.generate_dataset(n_texts=50, seed=3)
    low, high = CONTROL_SPREAD_RANGE
    assert all(low <= cfg["spread"] <= high and not cfg["active_dims"] for cfg in gt.values())


# ---------------------------------------------------------------------
# PEG with unequal subgroup sizes
# ---------------------------------------------------------------------
# The synthetic corpora are a full factorial, so their subgroups are always
# balanced and cannot reveal a broken size weighting. These groups are not.

def test_peg_variants_with_unequal_group_sizes():
    from polartox.polarized_tree import compute_peg, ndfu_score

    big = [1] * 25 + [5] * 25          # 50 ratings, polarized (high nDFU)
    small = [3] * 10                   # 10 ratings, unimodal (nDFU 0)
    groups = {"a": big, "b": small}
    node = big + small

    g, na, nb = ndfu_score(node, 5), ndfu_score(big, 5), ndfu_score(small, 5)
    weighted_avg = (50 * na + 10 * nb) / 60  # size-weighted
    assert abs(weighted_avg - (na + nb) / 2) > 1e-3   # weights must matter here

    peg_max = abs(g - max(na, nb))
    peg_avg = abs(g - weighted_avg)
    peg_min = abs(g - min(na, nb))

    assert compute_peg(node, groups, 5, "max")[0] == pytest.approx(peg_max)
    assert compute_peg(node, groups, 5, "avg")[0] == pytest.approx(peg_avg)
    assert compute_peg(node, groups, 5, "min")[0] == pytest.approx(peg_min)
    for beta in (0.5, 1.0, 2.0):
        expected = (1 + beta**2) * peg_max * peg_avg / (beta**2 * peg_max + peg_avg)
        assert compute_peg(node, groups, 5, "beta", beta)[0] == pytest.approx(expected)
