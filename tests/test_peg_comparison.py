"""
polartox.peg_comparison: PEGComparison reproduces what a hand-written loop over
pipelines gives (the logic pegcomparison.ipynb used to carry), needs ground
truth, takes settings from a benchmark's top_configs_, and saves its tables.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ndfu")

from test_trees import make_toy_dataset

from polartox import PEGComparison, PolarizedTreesBenchmark, PolarizedTreesPipeline, pairwise_ari
from polartox.polarized_tree import PEG_VARIANTS

DIMS = ["gender", "politics", "age"]
VARIANTS = ["max", "avg", "harmonic"]
SETTING = dict(theta_filter=0.1, min_size_frac=0.05, max_depth=4, h=0.05, relative_h=False, theta_stop=0.1)


@pytest.fixture(scope="module")
def corpora():
    return {"one": make_toy_dataset(seed=0, n_texts=4), "two": make_toy_dataset(seed=1, n_texts=3)}


@pytest.fixture(scope="module")
def comparison(corpora):
    return PEGComparison(corpora, dims=DIMS, scale=5, variants=VARIANTS).run(SETTING)


def reference_runs(corpora, setting=SETTING):
    """The loop the notebook used to hold: one pipeline per (formulation, corpus)."""
    return {
        (variant, name): (pipe := PolarizedTreesPipeline(dims=DIMS, scale=5, variant=variant, **setting),
                          pipe.run_full_evaluation(data, ground_truth=truth, verbose=False))
        for variant in VARIANTS
        for name, (data, truth) in corpora.items()
    }


# ---------------------------------------------------------------------
# Same numbers as the loop it replaces
# ---------------------------------------------------------------------

def test_overview_matches_a_hand_written_loop(corpora, comparison):
    runs = reference_runs(corpora)
    overview = comparison.overview()

    for variant in VARIANTS:
        per_corpus = [runs[variant, name] for name in corpora]
        for metric in ("jaccard", "precision", "recall", "exact_match"):
            expected = np.mean([results["recovery"][metric].astype(float).mean() for _, results in per_corpus])
            assert overview.loc[(0, variant), metric] == pytest.approx(expected)
        for key in ("retention_rate", "mean_leaves", "mean_depth", "mean_residual_ndfu"):
            expected = np.mean([results["diagnostics"][key] for _, results in per_corpus])
            assert overview.loc[(0, variant), key] == pytest.approx(expected)
        expected = np.mean([np.mean([np.mean([leaf["n"] for leaf in tree.get_leaves()])
                                     for tree in pipe.trees_.values()]) for pipe, _ in per_corpus])
        assert overview.loc[(0, variant), "mean_leaf_size"] == pytest.approx(expected)


def test_recovery_by_k_matches_the_pooled_texts(corpora, comparison):
    runs = reference_runs(corpora)
    table = comparison.recovery_by_k()
    pooled = pd.concat(runs["avg", name][1]["recovery"] for name in corpora)

    for k, group in pooled.groupby("k_true"):
        assert table.loc[(0, "avg", k), "texts"] == len(group)
        assert table.loc[(0, "avg", k), "jaccard"] == pytest.approx(group["jaccard"].mean())
        assert table.loc[(0, "avg", k), "exact_match"] == pytest.approx(group["exact_match"].astype(float).mean())


def test_tables_list_formulations_in_the_given_order_and_k_ascending(comparison):
    table = comparison.recovery_by_k()
    assert list(dict.fromkeys(table.index.get_level_values("variant"))) == VARIANTS
    ks = table.index.get_level_values("k_true")
    assert list(ks) == sorted(ks)
    pairs = list(dict.fromkeys(comparison.ari_by_k().index.get_level_values("pair")))
    assert pairs == ["max - avg", "max - harmonic", "avg - harmonic", "all pairs"]


def test_ari_matches_pairwise_ari(corpora, comparison):
    runs = reference_runs(corpora)
    for name, (data, _) in corpora.items():
        matrix, per_text = pairwise_ari({v: runs[v, name][0] for v in VARIANTS}, data)
        got_matrix, got_per_text = comparison.ari(corpus=name)
        pd.testing.assert_frame_equal(got_matrix, matrix)
        pd.testing.assert_frame_equal(got_per_text, per_text)

    mean = sum(pairwise_ari({v: runs[v, n][0] for v in VARIANTS}, corpora[n][0])[0] for n in corpora) / len(corpora)
    pd.testing.assert_frame_equal(comparison.ari_matrix(), mean)


def test_ari_by_k_and_disagreement(corpora, comparison):
    by_k = comparison.ari_by_k()
    assert set(by_k.index.get_level_values("pair")) == {"max - avg", "max - harmonic", "avg - harmonic", "all pairs"}
    # the toy corpus has one structure: k = 2 for every text
    assert set(by_k.index.get_level_values("k_true")) == {2}

    runs = reference_runs(corpora)
    values = [
        row["ari"] for name, (data, _) in corpora.items()
        for row in pairwise_ari({v: runs[v, name][0] for v in VARIANTS}, data)[1].to_dict("records")
    ]
    assert by_k.loc[(0, "all pairs", 2)] == pytest.approx(np.mean(values))

    ranking = comparison.disagreement()
    assert list(ranking.columns) == ["corpus", "text_id", "k_true", "mean_ari"]
    assert ranking["mean_ari"].is_monotonic_increasing
    assert len(ranking) == sum(data["text_id"].nunique() for data, _ in corpora.values())


def test_text_summary_and_show_text(comparison, capsys):
    summary = comparison.text_summary("one", 0)
    assert list(summary.index) == VARIANTS
    assert {"first split", "dimensions used", "leaves", "depth", "Jaccard"} == set(summary.columns)
    assert ((summary["Jaccard"] >= 0) & (summary["Jaccard"] <= 1)).all()

    comparison.show_text("one", 0)
    printed = capsys.readouterr().out
    assert "true active dimensions: gender (alpha 0.70), politics (alpha 0.60)" in printed
    assert all(f"--- {variant} ---" in printed for variant in VARIANTS)


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def test_ground_truth_is_required(corpora):
    data, _ = corpora["one"]
    with pytest.raises(TypeError, match="no ground truth"):
        PEGComparison({"one": data}, dims=DIMS, scale=5)
    with pytest.raises(TypeError):
        PEGComparison({"one": (data, None)}, dims=DIMS, scale=5)
    with pytest.raises(ValueError, match="missing entries"):
        PEGComparison({"one": (data, {0: {"active_dims": []}})}, dims=DIMS, scale=5)


def test_a_single_corpus_and_string_ids_are_accepted(corpora):
    data, truth = corpora["one"]
    stringly = {str(t): entry for t, entry in truth.items()}      # as read back from JSON
    single = PEGComparison((data, stringly), dims=DIMS, scale=5, variants=["max", "avg"])
    assert list(single.corpora) == ["corpus"]
    assert set(single.corpora["corpus"][1]) == set(truth)


def test_generated_dataset_flows_through_the_whole_chain():
    from polartox import (AnnotatorPool, DEFAULT_DEPTH_WEIGHTS, DEFAULT_DIMENSIONS,
                          DEFAULT_INTENSITY_RANGE)

    pool = AnnotatorPool(DEFAULT_DIMENSIONS, scale=5, intensity_range=DEFAULT_INTENSITY_RANGE,
                         depth_weights=DEFAULT_DEPTH_WEIGHTS, annotators_per_identity=3)
    generated = pool.generate_dataset(n_texts=6, seed=0)      # a GeneratedDataset, not unpacked

    comparison = PEGComparison(generated, dims=list(DEFAULT_DIMENSIONS), scale=5, variants=["max", "harmonic"])
    comparison.run(dict(theta_filter=0.2, min_size_frac=0.05, max_depth=4, h=0.1, relative_h=True, theta_stop=0.1))
    assert list(comparison.overview().index) == [(0, "max"), (0, "harmonic")]


def test_invalid_arguments(corpora):
    with pytest.raises(ValueError):
        PEGComparison(corpora, dims=DIMS, scale=5, variants=["max"])
    with pytest.raises(ValueError):
        PEGComparison(corpora, dims=DIMS, scale=5, variants=["max", "max"])
    with pytest.raises(ValueError):
        PEGComparison(corpora, dims=DIMS, scale=5, variants=["max", "beta"])
    with pytest.raises(ValueError, match="missing columns"):
        PEGComparison(corpora, dims=DIMS + ["nope"], scale=5)
    with pytest.raises(RuntimeError, match="run"):
        PEGComparison(corpora, dims=DIMS, scale=5).overview()


def test_settings_forms(corpora):
    one = PEGComparison(corpora, dims=DIMS, scale=5, variants=["max", "avg"])
    assert list(one.run(SETTING).settings_) == [0]
    assert list(one.run([SETTING, {**SETTING, "h": 0.1}]).settings_) == [0, 1]
    with pytest.raises(ValueError, match="not pipeline arguments"):
        one.run({**SETTING, "thetafilter": 0.3})
    with pytest.raises(TypeError):
        one.run(0.3)

    # a DataFrame: columns that are not pipeline arguments are skipped, a variant is ignored, empty cells too
    table = pd.DataFrame([{**SETTING, "variant": "min", "rank": 1, "jaccard": 0.9, "min_size": np.nan}], index=[7])
    assert list(one.run(table).settings_) == [7]
    assert one.settings_[7] == SETTING
    assert {variant for (_, variant, _) in one.runs_} == {"max", "avg"}


def test_a_bad_setting_fails_before_anything_runs(corpora):
    bad = PEGComparison(corpora, dims=DIMS, scale=5, variants=["max", "avg"])
    with pytest.raises(TypeError):
        bad.run({"theta_filter": 0.1})                  # h and max_depth missing
    assert bad.runs_ == {}


# ---------------------------------------------------------------------
# Connection to the benchmark
# ---------------------------------------------------------------------

def make_benchmark(corpora, text_groups=True):
    frames, truth, groups = [], {}, {}
    for offset, (name, (data, t)) in zip((0, 1000), corpora.items()):
        frames.append(data.assign(text_id=data["text_id"] + offset))
        truth.update({k + offset: v for k, v in t.items()})
        groups.update({k + offset: name for k in t})
    pipeline = PolarizedTreesPipeline(dims=DIMS, scale=5, **{k: v for k, v in SETTING.items()})
    configs = [{**SETTING, "variant": v} for v in ("avg", "max")]
    return PolarizedTreesBenchmark(
        pipeline, pd.concat(frames, ignore_index=True), truth, search_space=configs, strategy="full",
        text_groups=groups if text_groups else None, verbose=False, top_k=2,
    ).run()


def test_from_benchmark_and_its_top_configs(corpora):
    benchmark = make_benchmark(corpora)
    comparison = PEGComparison.from_benchmark(benchmark, variants=VARIANTS)

    assert list(comparison.corpora) == ["one", "two"]
    assert comparison.dims == DIMS and comparison.scale == 5
    assert {len(data) for data, _ in comparison.corpora.values()} == {len(corpora[n][0]) for n in corpora}

    comparison.run(benchmark.top_configs_)
    assert len(comparison.settings_) == 2
    # each chosen row, with its own formulation, gives back the numbers the benchmark saved
    assert comparison.check_against_benchmark(benchmark.top_configs_) > 0


def test_check_against_benchmark_detects_a_difference(corpora):
    benchmark = make_benchmark(corpora, text_groups=False)
    comparison = PEGComparison.from_benchmark(benchmark, variants=VARIANTS).run(benchmark.top_configs_)
    assert list(comparison.corpora) == ["all"]
    assert comparison.check_against_benchmark(benchmark.top_configs_) > 0

    changed = benchmark.top_configs_.copy()
    changed["jaccard"] = changed["jaccard"] + 0.1
    with pytest.raises(ValueError, match="differs from the benchmark"):
        comparison.check_against_benchmark(changed)


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def test_save_writes_every_table(comparison, tmp_path):
    folder = comparison.save(tmp_path / "out")
    names = {path.name for path in folder.iterdir()}
    assert {"selected_settings.csv", "per_run.csv", "overview.csv", "recovery_by_k.csv", "ari_by_k.csv"} <= names
    assert {"ari_setting0_one.csv", "ari_per_text_setting0_two.csv"} <= names

    saved = pd.read_csv(folder / "overview.csv", index_col=[0, 1])
    assert np.allclose(saved["jaccard"], comparison.overview()["jaccard"])
