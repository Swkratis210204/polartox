"""
Tests for branches the other test files do not reach: stopping rules of the
tree, empty F/C/P summaries, verbose output, benchmark input validation and
error handling, checkpoint edge cases, and the datagen printing helpers.
"""

import pandas as pd
import pytest

pytest.importorskip("ndfu")

from test_benchmark import FakePipeline, annotations, ground_truth, pipeline  # noqa: F401
from test_trees import make_toy_dataset

from polartox import (
    AnnotatorPool,
    DEFAULT_DIMENSIONS,
    PolarizedTreesBenchmark,
    PolarizedTreesPipeline,
    detect_polarized_subgroups,
)
from polartox.datagen import _demo

SPACE = {"theta_filter": [0.2, 0.3], "max_depth": [4, 6]}


def make_benchmark(pipeline, annotations, ground_truth, **kwargs):
    kwargs.setdefault("search_space", SPACE)
    kwargs.setdefault("strategy", "full")
    kwargs.setdefault("verbose", False)
    return PolarizedTreesBenchmark(pipeline, annotations, ground_truth, **kwargs)


# ---------------------------------------------------------------------
# Tree stopping rules and verbose output
# ---------------------------------------------------------------------

TREE_KWARGS = dict(h=0.05, scale=5, theta_stop=None, return_tree=True)


def test_max_depth_turns_deeper_nodes_into_leaves():
    data, _ = make_toy_dataset(n_texts=1)
    leaves, root = detect_polarized_subgroups(
        data, dims=["gender", "politics", "age"], min_size=10, max_depth=1, **TREE_KWARGS)
    assert not root["is_leaf"]                       # the root (depth 1) still splits
    assert all(len(leaf["path"]) == 1 for leaf in leaves)
    assert all(leaf["stop_reason"] == "max_depth/dimension exhaustion" for leaf in leaves)


def test_a_dimension_is_used_once_per_branch_then_leaves_remain():
    data, _ = make_toy_dataset(n_texts=1)
    leaves, root = detect_polarized_subgroups(
        data, dims=["gender"], min_size=10, max_depth=10, **TREE_KWARGS)
    assert root["split_dim"] == "gender"
    assert all(len(leaf["path"]) == 1 for leaf in leaves)
    assert all(leaf["stop_reason"] == "max_depth/dimension exhaustion" for leaf in leaves)


def test_verbose_build_prints_each_node(capsys):
    data, _ = make_toy_dataset(n_texts=1)
    detect_polarized_subgroups(data, dims=["gender"], min_size=10, max_depth=3, h=0.05,
                               scale=5, verbose=True)
    out = capsys.readouterr().out
    assert "[root] nDFU=" in out and "[gender=" in out


# ---------------------------------------------------------------------
# Pipeline: nothing to explain, and verbose reporting
# ---------------------------------------------------------------------

def balanced_polarized_text():
    """Half 1s, half 5s inside every gender: highly polarized overall, but no
    dimension explains it, so the tree stays a single indeterminate leaf."""
    rows = [{"text_id": 0, "gender": "male" if (i // 2) % 2 == 0 else "female",
             "rating": 1 if i % 2 == 0 else 5} for i in range(40)]
    return pd.DataFrame(rows)


def unexplained_pipeline():
    return PolarizedTreesPipeline(dims=["gender"], scale=5, theta_filter=0.3, h=0.05, max_depth=3)


def test_summaries_are_empty_when_no_dimension_explains_the_polarization():
    pipe = unexplained_pipeline()
    results = pipe.run_full_evaluation(balanced_polarized_text(), verbose=False)
    assert len(pipe.trees_) == 1 and pipe.trees_[0].n_leaves == 1
    assert results["F"].empty
    assert results["C"].empty          # the only leaf is indeterminate (p_tox = 0.5)
    assert results["P"].empty
    assert results["diagnostics"]["indeterminate_rate"] == 1.0


def test_verbose_evaluation_prints_diagnostics_and_recovery(capsys):
    gt = {0: {"active_dims": [], "peak": 3, "spread": 1.0}}
    unexplained_pipeline().run_full_evaluation(balanced_polarized_text(), ground_truth=gt)
    out = capsys.readouterr().out
    assert "=== diagnostics ===" in out and "=== recovery ===" in out
    assert "exact match rate" in out


def test_verbose_evaluation_without_ground_truth_skips_recovery(capsys):
    unexplained_pipeline().run_full_evaluation(balanced_polarized_text())
    out = capsys.readouterr().out
    assert "=== diagnostics ===" in out and "=== recovery ===" not in out


# ---------------------------------------------------------------------
# Benchmark: constructor validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, error, match", [
    (dict(strategy="random", n_runs=1.5), TypeError, "n_runs"),
    (dict(metrics=[]), ValueError, "At least one metric"),
    (dict(top_k="3"), TypeError, "top_k"),
    (dict(top_k=0), ValueError, "top_k"),
    (dict(search_space="abc"), TypeError, "dictionary or a list"),
    (dict(search_space={}), ValueError, "cannot be empty"),
    (dict(search_space=[]), ValueError, "cannot be empty"),
    (dict(search_space={"h": []}), ValueError, "'h'"),
    (dict(search_space=[{}]), ValueError, "Configuration 0 cannot be empty"),
    (dict(search_space=[{"h": 0.1}, 5]), TypeError, "Configuration 1 must be a dictionary"),
])
def test_invalid_constructor_arguments(pipeline, annotations, ground_truth, kwargs, error, match):
    with pytest.raises(error, match=match):
        make_benchmark(pipeline, annotations, ground_truth, **kwargs)


def test_search_values_must_be_a_list_or_tuple(pipeline, annotations, ground_truth):
    # The constructor already turns values into lists, so this check is only
    # reachable if the search space is replaced afterwards.
    bench = make_benchmark(pipeline, annotations, ground_truth)
    bench.search_space = {"h": 0.1}
    with pytest.raises(TypeError, match="'h'"):
        bench._validate_search_parameters()


def test_unknown_pipeline_parameter_in_search_space_is_rejected(
        pipeline, annotations, ground_truth):
    with pytest.raises(ValueError, match=r"Unknown pipeline parameters.*\['nope'\]"):
        make_benchmark(pipeline, annotations, ground_truth, search_space={"nope": [1]})


def test_unknown_variant_fails_before_any_configuration_is_evaluated(
        annotations, ground_truth, monkeypatch):
    real = PolarizedTreesPipeline(dims=["gender"], scale=5, theta_filter=0.3, h=0.1, max_depth=3)
    bench = PolarizedTreesBenchmark(
        real, annotations, ground_truth, strategy="full", verbose=False,
        search_space={"variant": ["max", "bogus"], "h": [0.1, 0.2]})
    monkeypatch.setattr(bench, "_run_one_configuration",
                        lambda config: pytest.fail("a configuration was evaluated"))
    with pytest.raises(ValueError, match="variant must be one of.*'bogus'"):
        bench.run()


def test_old_beta_variant_in_a_search_is_rejected_by_the_real_pipeline(
        annotations, ground_truth):
    real = PolarizedTreesPipeline(dims=["gender"], scale=5, theta_filter=0.3, h=0.1, max_depth=3)
    bench = PolarizedTreesBenchmark(real, annotations, ground_truth, strategy="full",
                                    verbose=False, search_space={"variant": ["beta"]})
    with pytest.raises(ValueError, match="'beta' no longer exists"):
        bench.run()


# ---------------------------------------------------------------------
# Benchmark: text ids and pipeline rebuilding
# ---------------------------------------------------------------------

def test_ground_truth_with_string_keys_matches_integer_text_ids(pipeline, annotations):
    gt = {"1": {"active_dims": ["gender"]}, "2": {"active_dims": ["gender"]}}
    bench = make_benchmark(pipeline, annotations, gt)
    assert set(bench._normalized_ground_truth()) == {1, 2}
    assert len(bench.run().get_results()) == 4


def test_non_numeric_text_ids_are_kept_as_they_are(pipeline, annotations):
    annotations = annotations.assign(text_id=annotations["text_id"].map({1: "a", 2: 1}))
    gt = {"a": {"active_dims": []}, "1": {"active_dims": []}}
    bench = make_benchmark(pipeline, annotations, gt)
    assert set(bench._normalized_ground_truth()) == {"a", 1}


class PipelineThatDoesNotStoreH:
    def __init__(self, dims, scale, h=0.1):
        self.dims, self.scale, self.received_h = dims, scale, h


def test_rebuild_falls_back_to_the_constructor_default(annotations, ground_truth):
    base = PipelineThatDoesNotStoreH(["gender"], 5)
    bench = PolarizedTreesBenchmark(base, annotations, ground_truth, search_space={"h": [0.1]},
                                    strategy="full", verbose=False)
    assert bench._build_pipeline({}).received_h == 0.1
    assert bench._build_pipeline({"h": 0.3}).received_h == 0.3


class PipelineWithRequiredParameterNotStored:
    def __init__(self, dims, scale, h):
        self.dims, self.scale = dims, scale


def test_rebuild_fails_for_a_required_parameter_it_cannot_find(annotations, ground_truth):
    base = PipelineWithRequiredParameterNotStored(["gender"], 5, h=0.1)
    bench = PolarizedTreesBenchmark(base, annotations, ground_truth, search_space={"h": [0.1]},
                                    strategy="full", verbose=False)
    with pytest.raises(ValueError, match="Cannot determine value for pipeline parameter 'h'"):
        bench._build_pipeline({})
    # ...but not when the configuration being built supplies that value
    assert bench._build_pipeline({"h": 0.2}) is not None


# ---------------------------------------------------------------------
# Benchmark: running
# ---------------------------------------------------------------------

def test_explicit_configuration_list_can_be_run(pipeline, annotations, ground_truth):
    configs = [{"theta_filter": 0.2, "max_depth": 4}, {"theta_filter": 0.3, "max_depth": 6}]
    bench = make_benchmark(pipeline, annotations, ground_truth, search_space=configs).run()
    assert {"theta_filter", "max_depth"} <= set(bench.get_results().columns)
    assert bench.get_best_config() == {"theta_filter": 0.3, "max_depth": 6}


def test_n_jobs_minus_one_means_all_cpus(pipeline, annotations, ground_truth):
    bench = make_benchmark(pipeline, annotations, ground_truth, n_jobs=-1,
                           search_space=[{"theta_filter": 0.3}]).run()
    assert len(bench.get_results()) == 1


def test_run_without_configurations_fails(pipeline, annotations, ground_truth, monkeypatch):
    bench = make_benchmark(pipeline, annotations, ground_truth)
    monkeypatch.setattr(bench, "_generate_configurations", lambda: [])
    with pytest.raises(RuntimeError, match="No configurations"):
        bench.run()


class PipelineWithTextIds(FakePipeline):
    def run_full_evaluation(self, annotations, ground_truth=None, verbose=False):
        recovery = super().run_full_evaluation(annotations, ground_truth, verbose)["recovery"]
        recovery = pd.concat([recovery, recovery], ignore_index=True)
        recovery.insert(0, "text_id", [1, 2])
        return {"recovery": recovery}


def test_text_groups_must_cover_every_evaluated_text(annotations, ground_truth):
    base = PipelineWithTextIds(dims=["gender"], scale=5)
    bench = make_benchmark(base, annotations, ground_truth, text_groups={1: "a", 2: "b"})
    bench.text_groups = {1: "a"}        # text 2 loses its corpus after validation
    with pytest.raises(ValueError, match="text_groups does not cover"):
        bench.run()


# ---------------------------------------------------------------------
# Benchmark: checkpoints and progress output
# ---------------------------------------------------------------------

def test_corrupt_checkpoint_is_ignored(pipeline, annotations, ground_truth, tmp_path):
    (tmp_path / "benchmark_checkpoint.pkl").write_bytes(b"not a pickle")
    bench = make_benchmark(pipeline, annotations, ground_truth, checkpoint_dir=tmp_path,
                           checkpoint_every=2).run()
    assert len(bench.get_results()) == 4


def test_progress_checkpoint_and_resume_messages(pipeline, annotations, ground_truth,
                                                 tmp_path, capsys):
    kwargs = dict(checkpoint_dir=tmp_path, checkpoint_every=2, verbose=True)
    make_benchmark(pipeline, annotations, ground_truth, **kwargs).run()
    first = capsys.readouterr().out
    assert "[1/4] jaccard=" in first and "Checkpoint saved at 2/4." in first

    make_benchmark(pipeline, annotations, ground_truth, **kwargs).run()
    assert "Resuming from checkpoint at 2/4." in capsys.readouterr().out


def test_checkpoint_from_another_search_is_reported(pipeline, annotations, ground_truth,
                                                    tmp_path, capsys):
    make_benchmark(pipeline, annotations, ground_truth, checkpoint_dir=tmp_path,
                   checkpoint_every=2).run()
    make_benchmark(pipeline, annotations, ground_truth, checkpoint_dir=tmp_path,
                   checkpoint_every=2, verbose=True,
                   search_space={"theta_filter": [0.2, 0.3], "max_depth": [5, 7]}).run()
    assert "different search; ignoring it" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Benchmark: results access
# ---------------------------------------------------------------------

@pytest.mark.parametrize("method", [
    "get_results", "get_best_config", "get_best_score", "get_best_pipeline", "get_top_configs",
    "summary", "get_report", "save_results", "save_report", "get_group_results",
    "save_group_results", "get_text_results", "save_text_results",
])
def test_every_accessor_requires_a_run(pipeline, annotations, ground_truth, tmp_path,
                                       monkeypatch, method):
    monkeypatch.chdir(tmp_path)     # savers must not write into the repository
    bench = make_benchmark(pipeline, annotations, ground_truth)
    with pytest.raises(RuntimeError):
        getattr(bench, method)()


def test_top_configs_k_must_be_an_integer(pipeline, annotations, ground_truth):
    bench = make_benchmark(pipeline, annotations, ground_truth).run()
    with pytest.raises(TypeError, match="k must be an integer"):
        bench.get_top_configs("2")


def test_summary_prints_the_best_configuration_and_returns_top_results(
        pipeline, annotations, ground_truth, capsys):
    bench = make_benchmark(pipeline, annotations, ground_truth).run()
    top = bench.summary(k=2)
    out = capsys.readouterr().out
    assert "Configurations evaluated: 4" in out
    assert "Best score:" in out and "theta_filter: 0.3" in out and "max_depth: 6" in out
    assert len(top) == 2 and top.iloc[0]["theta_filter"] == 0.3


def test_summary_defaults_to_top_k_rows(pipeline, annotations, ground_truth):
    bench = make_benchmark(pipeline, annotations, ground_truth, top_k=3).run()
    assert len(bench.summary()) == 3


# ---------------------------------------------------------------------
# datagen printing helpers
# ---------------------------------------------------------------------

def small_pool():
    return AnnotatorPool(DEFAULT_DIMENSIONS, 5, (0.3, 1.0), {0: 0.5, 1: 0.5}, 1)


def test_generated_dataset_repr_reports_size():
    result = small_pool().generate_dataset(n_texts=3, seed=0)
    assert repr(result).startswith(f"GeneratedDataset(3 texts, {len(result)} rows)")


def test_pool_summary_prints_its_settings(capsys):
    small_pool().summary()
    out = capsys.readouterr().out
    assert "Pool size         : 162" in out and "Rating scale      : 1-5" in out


def test_demo_runs_end_to_end(capsys):
    _demo(n_texts=3)
    out = capsys.readouterr().out
    assert "Pool size" in out and "Dataset shape: (4860, 8)" in out and "Text 0" in out
