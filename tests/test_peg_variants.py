"""
The five PEG formulations: max, weighted, min, mean and harmonic.

The expected values below are worked out by hand from the definitions, not
taken from the code, so they check the formulas themselves (a golden file
only shows that results did not change).

Worked example. Node nDFU G = 0.8 split into two equal-sized subgroups with
nDFU 0.2 and 0.5:
    PEGmax = |0.8 - 0.5|            = 0.30
    PEGmin = |0.8 - 0.2|            = 0.60
    PEGweighted = |0.8 - (0.2 + 0.5)/2|  = 0.45
    mean     = (0.30 + 0.45 + 0.60) / 3             = 0.45
    harmonic = 3 / (1/0.30 + 1/0.45 + 1/0.60)       = 27/65
"""

import numpy as np
import pytest

pytest.importorskip("ndfu")

from polartox import PolarizedTreesPipeline, detect_polarized_subgroups
from polartox.polarized_tree import (
    PEG_VARIANTS,
    _combine_peg,
    check_variant,
    compute_peg,
)

G = 0.8
GROUPS = {"a": 0.2, "b": 0.5}
SIZES = {"a": 10, "b": 10}


def peg(variant, beta=1.0, node=G, groups=GROUPS, sizes=SIZES):
    return _combine_peg(node, groups, sizes, sum(sizes.values()), variant, beta)


# ---------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------

def test_the_five_variants_are_the_documented_ones():
    assert PEG_VARIANTS == ("max", "weighted", "min", "mean", "harmonic")


@pytest.mark.parametrize("variant, expected", [
    ("max", 0.30),
    ("min", 0.60),
    ("weighted", 0.45),
    ("mean", 0.45),
    ("harmonic", 27 / 65),
])
def test_worked_example(variant, expected):
    assert peg(variant) == pytest.approx(expected)


def test_harmonic_beta_weights_avg_against_max_and_min():
    # weights (max, weighted, min) = (1, beta^2, 1); beta = 2 -> (1, 4, 1):
    # 6 / (1/0.30 + 4/0.45 + 1/0.60) = 54/125
    assert peg("harmonic", beta=2.0) == pytest.approx(54 / 125)
    # beta = 0.5 -> (1, 0.25, 1): 2.25 / (1/0.30 + 0.25/0.45 + 1/0.60) = 81/200
    assert peg("harmonic", beta=0.5) == pytest.approx(81 / 200)


@pytest.mark.parametrize("variant", ["max", "weighted", "min", "mean"])
def test_beta_is_ignored_by_the_other_variants(variant):
    assert peg(variant, beta=0.5) == peg(variant, beta=2.0) == peg(variant)


def test_size_weighting_only_affects_avg_based_formulas():
    # subgroup "a" (nDFU 0.2) now holds 3 of 4 annotators
    sizes = {"a": 30, "b": 10}
    # PEGweighted = |0.8 - (0.75 * 0.2 + 0.25 * 0.5)| = 0.525
    assert peg("weighted", sizes=sizes) == pytest.approx(0.525)
    assert peg("max", sizes=sizes) == pytest.approx(0.30)
    assert peg("min", sizes=sizes) == pytest.approx(0.60)
    assert peg("mean", sizes=sizes) == pytest.approx((0.30 + 0.525 + 0.60) / 3)
    assert peg("harmonic", sizes=sizes) == pytest.approx(3 / (1 / 0.30 + 1 / 0.525 + 1 / 0.60))


def test_a_subgroup_more_polarized_than_the_node_still_gives_a_positive_gain():
    # the paper's formulas use absolute values: a worsened subgroup counts too
    groups, sizes = {"a": 0.9, "b": 0.3}, {"a": 10, "b": 10}
    assert peg("max", groups=groups, sizes=sizes, node=0.6) == pytest.approx(0.3)
    assert peg("min", groups=groups, sizes=sizes, node=0.6) == pytest.approx(0.3)
    assert peg("weighted", groups=groups, sizes=sizes, node=0.6) == pytest.approx(0.0)
    assert peg("mean", groups=groups, sizes=sizes, node=0.6) == pytest.approx(0.2)


# ---------------------------------------------------------------------
# Zero handling (unchanged rule: no gain from any view -> harmonic is 0)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("groups", [
    {"a": 0.5, "b": 0.1},     # PEGmax = 0   (node 0.5 equals the most polarized subgroup)
    {"a": 0.9, "b": 0.1},     # PEGweighted = 0   (node 0.5 equals the size-weighted average)
    {"a": 0.5, "b": 0.9},     # PEGmin = 0   (node 0.5 equals the least polarized subgroup)
    {"a": 0.5, "b": 0.5},     # all three are 0
])
def test_harmonic_is_zero_when_any_base_is_zero(groups):
    assert peg("harmonic", node=0.5, groups=groups) == 0.0


def test_mean_is_not_zero_when_only_one_base_is_zero():
    # node 0.5, subgroups 0.5 and 0.1: PEGmax = 0, PEGweighted = 0.2, PEGmin = 0.4
    assert peg("mean", node=0.5, groups={"a": 0.5, "b": 0.1}) == pytest.approx(0.2)


def test_harmonic_with_beta_zero_still_returns_zero_for_a_zero_base():
    assert peg("harmonic", beta=0.0, node=0.5, groups={"a": 0.5, "b": 0.1}) == 0.0


# ---------------------------------------------------------------------
# Properties that must hold for any subgroup nDFUs
# ---------------------------------------------------------------------

def test_mean_and_harmonic_stay_between_the_bases_and_harmonic_never_exceeds_mean():
    rng = np.random.default_rng(0)
    for _ in range(500):
        k = int(rng.integers(2, 5))
        groups = {i: float(rng.uniform(0, 1)) for i in range(k)}
        sizes = {i: int(rng.integers(1, 50)) for i in range(k)}
        node = float(rng.uniform(0, 1))
        bases = [peg(v, node=node, groups=groups, sizes=sizes) for v in ("max", "weighted", "min")]
        mean = peg("mean", node=node, groups=groups, sizes=sizes)
        harmonic = peg("harmonic", node=node, groups=groups, sizes=sizes)
        assert min(bases) - 1e-12 <= harmonic <= mean + 1e-12 <= max(bases) + 2e-12


def test_the_variants_agree_when_all_three_bases_are_equal():
    # a single usable subgroup split: max = weighted = min
    for variant in PEG_VARIANTS:
        assert peg(variant, groups={"a": 0.2}, sizes={"a": 20}) == pytest.approx(0.6)


def test_compute_peg_end_to_end_matches_the_formulas():
    node = np.array([1] * 20 + [5] * 20)
    groups = {"a": np.array([1] * 15 + [5] * 5), "b": np.array([1] * 5 + [5] * 15)}
    bases = [compute_peg(node, groups, 5, v)[0] for v in ("max", "weighted", "min")]
    assert compute_peg(node, groups, 5, "mean")[0] == pytest.approx(sum(bases) / 3)
    assert compute_peg(node, groups, 5, "harmonic")[0] == pytest.approx(
        3 / sum(1 / b for b in bases))


# ---------------------------------------------------------------------
# Variant names are validated early, and removed names say what replaced them
# ---------------------------------------------------------------------

@pytest.mark.parametrize("variant", PEG_VARIANTS)
def test_every_documented_variant_is_accepted(variant):
    check_variant(variant)


def test_unknown_variant_lists_the_valid_ones():
    with pytest.raises(ValueError, match="'max', 'weighted', 'min', 'mean', 'harmonic'.*'bogus'"):
        check_variant("bogus")


def test_old_beta_name_is_rejected_with_a_pointer_to_harmonic():
    with pytest.raises(ValueError, match="'beta' no longer exists.*'harmonic'"):
        check_variant("beta")


def test_old_var_name_is_rejected_with_a_pointer_to_avg():
    with pytest.raises(ValueError, match="'var' no longer exists.*'weighted'"):
        check_variant("var")


def test_unhashable_variant_is_a_value_error_not_a_crash():
    with pytest.raises(ValueError, match="variant must be one of"):
        check_variant(["max"])


def test_compute_peg_rejects_unknown_variants():
    with pytest.raises(ValueError, match="variant must be one of"):
        compute_peg(np.array([1, 5]), {"a": np.array([1]), "b": np.array([5])}, 5, "beta")


def test_pipeline_rejects_a_bad_variant_at_construction():
    with pytest.raises(ValueError, match="variant must be one of"):
        PolarizedTreesPipeline(dims=["gender"], scale=5, theta_filter=0.3, h=0.1,
                               max_depth=3, variant="beta")


def test_tree_builder_rejects_a_bad_variant_even_when_no_node_needs_a_split():
    import pandas as pd

    flat = pd.DataFrame({"gender": ["m", "f"] * 10, "rating": [3] * 20})   # nDFU 0: stops at the root
    with pytest.raises(ValueError, match="variant must be one of"):
        detect_polarized_subgroups(flat, dims=["gender"], min_size=2, h=0.1, max_depth=3,
                                   scale=5, variant="bogus")


def test_default_variant_is_harmonic():
    assert PolarizedTreesPipeline(dims=["gender"], scale=5, theta_filter=0.3, h=0.1,
                                  max_depth=3).variant == "harmonic"
