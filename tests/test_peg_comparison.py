"""
polartox.peg_comparison: PEGComparison reproduces what a hand-written loop over
pipelines gives (the logic pegcomparison.ipynb used to carry), takes settings from
a benchmark's top_configs_, saves its tables, and analyses the trees with or
without ground truth.
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

def test_ground_truth_is_optional_but_checked_when_given(corpora):
    data, truth = corpora["one"]
    without = PEGComparison({"one": data, "two": (corpora["two"][0], None)}, dims=DIMS, scale=5)
    assert without.truth_corpora == []

    mixed = PEGComparison({"one": (data, truth), "two": corpora["two"][0]}, dims=DIMS, scale=5)
    assert mixed.truth_corpora == ["one"]

    with pytest.raises(TypeError):
        PEGComparison({"one": (data, [1, 2])}, dims=DIMS, scale=5)
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


def test_a_setting_that_keeps_no_text_says_which(corpora):
    comparison = PEGComparison(corpora, dims=DIMS, scale=5, variants=["max", "avg"])
    with pytest.raises(ValueError, match="keeps no text"):
        comparison.run({**SETTING, "theta_filter": 0.99})


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


# ---------------------------------------------------------------------
# Without ground truth: the trees themselves
# ---------------------------------------------------------------------

RECOVERY = ["jaccard", "precision", "recall", "exact_match"]


@pytest.fixture(scope="module")
def blind(corpora):
    """The same corpora and settings, without the ground truth; two settings."""
    data = {name: corpus[0] for name, corpus in corpora.items()}
    comparison = PEGComparison(data, dims=DIMS, scale=5, variants=VARIANTS)
    return comparison.run([SETTING, {**SETTING, "theta_filter": 0.14, "h": 0.1}])


def test_blind_runs_have_no_recovery_but_everything_about_the_trees(blind):
    frame = blind.per_run()
    assert len(frame) == 2 * len(VARIANTS) * 2          # settings x formulations x corpora
    assert not set(RECOVERY) & set(frame.columns)
    wanted = {"n_texts", "n_retained", "retention_rate", "mean_leaves", "mean_depth", "mean_tree_depth",
              "mean_leaf_size", "mean_residual_ndfu", "weighted_residual_ndfu", "explained_share",
              "mean_split_peg", "mean_top_split_peg", "indeterminate_rate", "no_split_rate",
              "multi_dim_share", "mean_dims_used", "dims_never_used"}
    assert wanted <= set(frame.columns)

    overview = blind.overview()
    assert list(overview.index.names) == ["setting", "variant"]
    assert not set(RECOVERY) & set(overview.columns)
    assert ((overview["multi_dim_share"] >= 0) & (overview["multi_dim_share"] <= 1)).all()
    assert (overview["n_texts"] == 4 + 3).all()


def test_tree_statistics_reproduce_the_pipeline_diagnostics(corpora):
    from polartox import tree_statistics

    data, truth = corpora["one"]
    pipe = PolarizedTreesPipeline(dims=DIMS, scale=5, variant="avg", **SETTING)
    diagnostics = pipe.run_full_evaluation(data, ground_truth=truth, verbose=False)["diagnostics"]
    stats = tree_statistics(pipe)
    for key in ("retention_rate", "mean_leaves", "mean_depth", "mean_residual_ndfu", "mean_top_split_peg",
                "indeterminate_rate"):
        assert stats[key] == pytest.approx(diagnostics[key]), key
    assert stats["dims_never_used"] == diagnostics["dims_never_used"]


def test_tree_statistics_by_hand():
    from polartox import tree_statistics

    data, _ = make_toy_dataset(n_texts=3)
    pipe = PolarizedTreesPipeline(dims=DIMS, scale=5, variant="max", **SETTING)
    pipe.run_full_evaluation(data, verbose=False)
    stats = tree_statistics(pipe)
    trees = list(pipe.trees_.values())

    assert stats["mean_tree_depth"] == pytest.approx(np.mean([t.depth for t in trees]))
    assert stats["multi_dim_share"] == pytest.approx(
        np.mean([len({d for _, d, _, _, _ in t.internal_nodes()}) >= 2 for t in trees]))
    assert stats["no_split_rate"] == pytest.approx(np.mean([t.n_leaves == 1 for t in trees]))
    assert stats["mean_split_peg"] == pytest.approx(
        np.mean([peg for t in trees for _, _, peg, _, _ in t.internal_nodes()]))
    assert 0 <= stats["weighted_residual_ndfu"] <= 1       # the nDFU left, weighted by the leaves' annotators
    assert stats["n_retained"] == len(trees) and stats["n_texts"] == 3


def test_per_text_and_retention_curve(blind):
    table = blind.per_text()
    assert {"setting", "variant", "corpus", "text_id", "n_annotators", "ndfu", "retained", "n_leaves",
            "tree_depth", "first_split", "dims_used", "root_ndfu", "explained_share"} <= set(table.columns)
    assert table["retained"].sum() == table["n_leaves"].notna().sum()
    assert not {"k_true", "jaccard"} & set(table.columns)

    curve = blind.retention_curve([0.0, 0.5, 1.5])
    assert list(curve.columns) == ["one", "two"]
    assert (curve.loc[0.0] == 1.0).all() and (curve.loc[1.5] == 0.0).all()
    assert (curve.diff().dropna() <= 0).all().all()            # a higher theta keeps fewer texts


def test_retention_curve_matches_the_pipelines_own_retention(blind):
    curve = blind.retention_curve([0.1, 0.14])
    frame = blind.per_run()
    for setting, theta in ((0, 0.1), (1, 0.14)):
        for corpus in ("one", "two"):
            row = frame[(frame["setting"] == setting) & (frame["corpus"] == corpus)].iloc[0]
            assert curve.loc[theta, corpus] == pytest.approx(row["retention_rate"])


def test_depth_distribution_and_dimension_usage(blind):
    depth = blind.depth_distribution()
    assert np.allclose(depth.sum(), 1.0) and list(depth.columns) == VARIANTS
    cumulative = blind.depth_distribution(cumulative=True)
    assert np.allclose(cumulative.iloc[-1], 1.0) and (cumulative.diff().dropna() >= 0).all().all()

    usage = blind.dimension_usage()
    assert list(usage.index.get_level_values("variant").unique()) == VARIANTS
    assert set(usage.index.get_level_values("dim")) == set(DIMS)
    assert ((usage["first_split"] <= usage["used"]) & (usage["used"] <= 1)).all()
    for variant in VARIANTS:
        assert usage.loc[variant, "first_split"].sum() <= 1.0 + 1e-9
    assert usage.loc[(slice(None), "age"), "used"].eq(0).all()      # the toy corpus has no age effect


# ---------------------------------------------------------------------
# How close are the trees of two runs
# ---------------------------------------------------------------------

def test_normalized_mutual_information():
    from polartox import normalized_mutual_information

    assert normalized_mutual_information([0, 0, 1, 1], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    assert normalized_mutual_information([0, 0, 0, 0], [5, 5, 5, 5]) == 1.0
    assert normalized_mutual_information([0, 0, 0, 0], [0, 0, 1, 1]) == 0.0
    assert normalized_mutual_information([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0, abs=1e-12)
    a, b = [0, 0, 1, 1, 2, 2, 2], [1, 0, 0, 1, 1, 2, 2]
    assert normalized_mutual_information(a, b) == pytest.approx(normalized_mutual_information(b, a))
    with pytest.raises(ValueError):
        normalized_mutual_information([0, 1], [0, 1, 2])


def test_normalized_mutual_information_matches_scikit_learn():
    from polartox import normalized_mutual_information

    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(2, 80))
        a = rng.integers(0, int(rng.integers(1, 6)), n)
        b = rng.integers(0, int(rng.integers(1, 6)), n)
        assert normalized_mutual_information(a, b) == pytest.approx(metrics.normalized_mutual_info_score(a, b), abs=1e-9)


def test_similarity_within_a_setting_matches_ari_matrix_and_the_hand_computation(blind, corpora):
    from polartox import adjusted_rand_index

    for measure in ("ari", "nmi", "dims", "first_split"):
        matrix = blind.similarity(measure, 0)
        assert list(matrix.index) == list(matrix.columns) == VARIANTS
        assert np.allclose(np.diag(matrix), 1.0) and np.allclose(matrix, matrix.T)
        assert ((matrix >= -1e-12) & (matrix <= 1 + 1e-12)).all().all()

    pd.testing.assert_frame_equal(blind.similarity("ari", 0), blind.ari_matrix(0))
    pd.testing.assert_frame_equal(blind.similarity("dims", 0), blind.dims_agreement(0))
    pd.testing.assert_frame_equal(blind.similarity("first_split", 0), blind.first_split_agreement(0))

    # the mean over the texts, by hand, for one pair in one corpus
    data = corpora["one"][0]
    pipes = {v: blind.runs_[0, v, "one"][0] for v in ("max", "avg")}
    expected = np.mean([
        adjusted_rand_index(pipes["max"].trees_[t].leaf_labels(data[data["text_id"] == t]),
                            pipes["avg"].trees_[t].leaf_labels(data[data["text_id"] == t]))
        for t in pipes["max"].trees_
    ])
    one_corpus = PEGComparison({"one": data}, dims=DIMS, scale=5, variants=["max", "avg"]).run(SETTING)
    assert one_corpus.similarity("ari", 0).loc["max", "avg"] == pytest.approx(expected)


def test_similarity_across_all_settings(blind):
    matrix = blind.similarity("ari")
    runs = [(s, v) for s in (0, 1) for v in VARIANTS]
    assert matrix.index.tolist() == matrix.columns.tolist() == runs
    assert np.allclose(np.diag(matrix), 1.0) and np.allclose(matrix, matrix.T)
    # the block of one setting is that setting's own matrix
    assert np.allclose(matrix.loc[0, 0].to_numpy(), blind.similarity("ari", 0).to_numpy())
    assert np.allclose(matrix.loc[1, 1].to_numpy(), blind.similarity("ari", 1).to_numpy())

    with pytest.raises(ValueError, match="measure"):
        blind.similarity("correlation")
    with pytest.raises(KeyError):
        blind.similarity("ari", 7)


def test_identical_settings_are_identical_runs(corpora):
    data = {name: corpus[0] for name, corpus in corpora.items()}
    twin = PEGComparison(data, dims=DIMS, scale=5, variants=["max", "avg"]).run([SETTING, dict(SETTING)])
    matrix = twin.similarity("ari")
    assert matrix.loc[(0, "max"), (1, "max")] == pytest.approx(1.0)
    assert matrix.loc[(0, "avg"), (1, "avg")] == pytest.approx(1.0)


# ---------------------------------------------------------------------
# F, C, P, and what needs ground truth
# ---------------------------------------------------------------------

def test_fcp_and_subgroup_overlap(blind):
    tables = blind.fcp(0, "one")
    assert list(tables) == VARIANTS and set(tables["max"]) == {"F", "C", "P"}
    pipe, results = blind.runs_[0, "avg", "one"]
    assert blind.fcp(0, "one", "avg")["C"].equals(results["C"])
    assert "ever_truly_active" not in tables["max"]["F"].columns          # no ground truth: no validation columns

    for table in ("C", "P"):
        matrix = blind.subgroup_overlap(0, "one", table=table, top=5)
        assert list(matrix.index) == VARIANTS and np.allclose(np.diag(matrix), 1.0)
        assert ((matrix >= 0) & (matrix <= 1)).all().all()
    with pytest.raises(ValueError):
        blind.subgroup_overlap(0, "one", table="F")
    with pytest.raises(ValueError, match="name a corpus"):
        blind.fcp(0)


def test_fcp_comparison_puts_the_formulations_side_by_side(blind):
    f = blind.fcp_comparison(0, "one", "F")
    assert list(f.index) == DIMS
    assert f.columns.get_level_values(0).unique().tolist() == VARIANTS
    tables = blind.fcp(0, "one")
    for variant in VARIANTS:
        own = tables[variant]["F"]
        for depth in [c for c in own.columns if isinstance(c, int)]:
            for dim in own.index:
                assert f.loc[dim, (variant, depth)] == own.loc[dim, depth]

    for table, columns in (("C", ["n_s", "frac_toxic"]), ("P", ["n_s", "mean_peg"])):
        side = blind.fcp_comparison(0, "one", table, top=3)
        assert side.columns.get_level_values(0).unique().tolist() == VARIANTS
        assert side.columns.get_level_values(1).unique().tolist() == columns
        for variant in VARIANTS:                                    # every formulation's own top 3 is in the table
            own = tables[variant][table]
            for subgroup in own.index[:3]:
                position = list(side.index).index(subgroup)               # a tuple label: look it up by position
                assert side[(variant, "n_s")].iloc[position] == own["n_s"].iloc[list(own.index).index(subgroup)]
        assert len(side) <= 3 * len(VARIANTS)
    with pytest.raises(ValueError):
        blind.fcp_comparison(0, "one", "X")


def test_what_needs_ground_truth_says_so(blind):
    for call in (blind.recovery_by_k, blind.ari_by_k, lambda: blind.check_against_benchmark(pd.DataFrame())):
        with pytest.raises(RuntimeError, match="ground truth"):
            call()


def test_blind_disagreement_text_summary_show_text_and_save(blind, tmp_path, capsys):
    ranking = blind.disagreement(0)
    assert ranking["mean_ari"].is_monotonic_increasing and ranking["k_true"].isna().all()

    summary = blind.text_summary("one", 0, 0)
    assert "Jaccard" not in summary.columns
    blind.show_text("one", 0, 0)
    assert "nDFU" in capsys.readouterr().out

    folder = blind.save(tmp_path / "blind")
    names = {path.name for path in folder.iterdir()}
    assert {"overview.csv", "per_run.csv", "per_text.csv", "retention_curve.csv",
            "similarity_ari_setting0.csv", "similarity_nmi_setting1.csv", "similarity_ari_all_settings.csv",
            "dimension_usage_setting0.csv", "depth_distribution_setting1.csv"} <= names
    assert "recovery_by_k.csv" not in names
    assert (folder / "fcp" / "F_setting0_one_max.csv").exists()


def test_with_ground_truth_the_same_analysis_plus_recovery(comparison):
    frame = comparison.per_run()
    assert set(RECOVERY) <= set(frame.columns) and "multi_dim_share" in frame.columns
    assert comparison.similarity("nmi").shape == (len(VARIANTS), len(VARIANTS))
    assert comparison.retention_curve().shape[1] == 2


def test_mixed_corpora_average_recovery_over_those_that_have_it(corpora):
    mixed = PEGComparison({"one": corpora["one"], "two": corpora["two"][0]}, dims=DIMS, scale=5,
                          variants=["max", "avg"]).run(SETTING)
    frame = mixed.per_run()
    assert frame[frame["corpus"] == "one"]["jaccard"].notna().all()
    assert frame[frame["corpus"] == "two"]["jaccard"].isna().all()
    only_one, both = mixed.overview(["one"]), mixed.overview()
    assert both["jaccard"].tolist() == pytest.approx(only_one["jaccard"].tolist())   # the corpus without truth adds nothing
    assert (both["n_texts"] == 7).all() and (only_one["n_texts"] == 4).all()
    assert mixed.recovery_by_k()["texts"].sum() > 0
