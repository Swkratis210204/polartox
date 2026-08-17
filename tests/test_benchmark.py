import json

import pandas as pd
import pytest

from polartox.benchmark import (
    PolarizedTreesBenchmark,
    DEFAULT_SEARCH_SPACE,
)


# ---------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------
#
# This deliberately avoids running the real Polarized Trees algorithm.
# We only want to test the benchmark machinery here.
# ---------------------------------------------------------------------

class FakePipeline:

    def __init__(
        self,
        dims,
        scale,
        theta_filter=0.2,
        min_size_frac=0.03,
        max_depth=4,
        variant="beta",
        beta=1.0,
        h=0.1,
        relative_h=True,
        theta_stop=0.1,
    ):
        self.dims = dims
        self.scale = scale
        self.theta_filter = theta_filter
        self.min_size_frac = min_size_frac
        self.max_depth = max_depth
        self.variant = variant
        self.beta = beta
        self.h = h
        self.relative_h = relative_h
        self.theta_stop = theta_stop

    def run_full_evaluation(
        self,
        annotations,
        ground_truth=None,
        verbose=False,
    ):
        # Produce deterministic scores from the configuration.
        #
        # The "best" configuration is deliberately known:
        # theta_filter=0.3 and max_depth=6.
        score = 0.0

        if self.theta_filter == 0.3:
            score += 0.5

        if self.max_depth == 6:
            score += 0.4

        if self.relative_h:
            score += 0.05

        if self.h == 0.15:
            score += 0.05

        recovery = pd.DataFrame(
            {
                "jaccard": [score],
                "precision": [score],
                "recall": [score],
                "exact_match": [score],
            }
        )

        return {
            "recovery": recovery,
        }


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def annotations():
    return pd.DataFrame(
        {
            "text_id": [1, 1, 1, 2, 2, 2],
            "rating": [1, 5, 1, 5, 1, 5],
            "gender": [
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
            ],
        }
    )


@pytest.fixture
def ground_truth():
    return {
        1: {"active_dims": ["gender"]},
        2: {"active_dims": ["gender"]},
    }


@pytest.fixture
def pipeline():
    return FakePipeline(
        dims=["gender"],
        scale=5,
    )


@pytest.fixture
def small_search_space():
    return {
        "theta_filter": [0.2, 0.3],
        "max_depth": [4, 6],
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def test_valid_initialization(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
    )

    assert benchmark.strategy == "full"
    assert benchmark.metrics == [
        "jaccard",
        "precision",
        "recall",
        "exact_match",
    ]
    assert benchmark.selection_metric == "jaccard"


def test_annotations_must_be_dataframe(
    pipeline,
    ground_truth,
    small_search_space,
):
    with pytest.raises(TypeError):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations="not a dataframe",
            ground_truth=ground_truth,
            search_space=small_search_space,
        )


def test_missing_required_annotation_column(
    pipeline,
    ground_truth,
    small_search_space,
):
    annotations = pd.DataFrame(
        {
            "text_id": [1, 1],
            "gender": ["male", "female"],
        }
    )

    with pytest.raises(ValueError, match="rating"):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
        )


def test_empty_annotations(
    pipeline,
    ground_truth,
    small_search_space,
):
    annotations = pd.DataFrame(
        columns=["text_id", "rating"]
    )

    with pytest.raises(ValueError, match="empty"):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
        )


def test_ground_truth_must_be_dict(
    pipeline,
    annotations,
    small_search_space,
):
    with pytest.raises(TypeError):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=[],
            search_space=small_search_space,
        )


def test_missing_ground_truth_text_id(
    pipeline,
    annotations,
    small_search_space,
):
    ground_truth = {
        1: {"active_dims": ["gender"]},
    }

    with pytest.raises(ValueError, match="missing entries"):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
        )


def test_invalid_strategy(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    with pytest.raises(ValueError, match="strategy"):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
            strategy="invalid",
        )


def test_random_requires_positive_integer_runs(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    with pytest.raises(ValueError):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
            strategy="random",
            n_runs=0,
        )


def test_invalid_metric(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    with pytest.raises(ValueError, match="Unknown metrics"):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
            metrics=["jaccard", "not_a_metric"],
        )


def test_selection_metric_must_be_requested(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    with pytest.raises(ValueError, match="selection_metric"):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
            metrics=["precision"],
            selection_metric="jaccard",
        )


def test_invalid_selection_direction(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    with pytest.raises(ValueError, match="selection_direction"):
        PolarizedTreesBenchmark(
            pipeline=pipeline,
            annotations=annotations,
            ground_truth=ground_truth,
            search_space=small_search_space,
            selection_direction="invalid",
        )


# ---------------------------------------------------------------------
# Configuration generation
# ---------------------------------------------------------------------

def test_full_strategy_generates_all_combinations(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
    )

    configurations = benchmark.configurations()

    assert len(configurations) == 4

    assert {
        "theta_filter": 0.2,
        "max_depth": 4,
    } in configurations

    assert {
        "theta_filter": 0.3,
        "max_depth": 6,
    } in configurations


def test_random_strategy_is_reproducible(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark1 = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="random",
        n_runs=2,
        seed=42,
    )

    benchmark2 = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="random",
        n_runs=2,
        seed=42,
    )

    assert benchmark1.configurations() == benchmark2.configurations()


def test_random_strategy_does_not_exceed_search_space(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="random",
        n_runs=100,
        seed=42,
    )

    assert len(benchmark.configurations()) == 4


def test_explicit_configuration_list(
    pipeline,
    annotations,
    ground_truth,
):
    configurations = [
        {
            "theta_filter": 0.2,
            "max_depth": 4,
        },
        {
            "theta_filter": 0.3,
            "max_depth": 6,
        },
    ]

    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=configurations,
        strategy="full",
    )

    assert benchmark.configurations() == configurations


# ---------------------------------------------------------------------
# Default paper search space
# ---------------------------------------------------------------------

def test_default_search_space_has_paper_configuration_count(
    pipeline,
    annotations,
    ground_truth,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        strategy="full",
    )

    configurations = benchmark.configurations()

    # The paper search space contains 3,240 valid configurations.
    assert len(configurations) == 3240


def test_default_search_space_matches_paper_dimensions(
    pipeline,
    annotations,
    ground_truth,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        strategy="full",
    )

    configurations = benchmark.configurations()

    assert set(
        config["theta_filter"] for config in configurations
    ) == {0.2, 0.3, 0.4}

    assert set(
        config["min_size_frac"] for config in configurations
    ) == {0.02, 0.03, 0.05}

    assert set(
        config["max_depth"] for config in configurations
    ) == {4, 6, 8}

    assert set(
        config["variant"] for config in configurations
    ) == {"max", "var", "beta"}

    assert set(
        config["h"] for config in configurations
    ) == {0.05, 0.10, 0.15, 0.20}

    assert set(
        config["relative_h"] for config in configurations
    ) == {False, True}

    assert set(
        config["theta_stop"] for config in configurations
    ) == {0.05, 0.10, 0.15}


def test_default_search_space_has_correct_conditional_beta_values(
    pipeline,
    annotations,
    ground_truth,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        strategy="full",
    )

    configurations = benchmark.configurations()

    beta_by_variant = {
        variant: {
            config["beta"]
            for config in configurations
            if config["variant"] == variant
        }
        for variant in {"max", "var", "beta"}
    }

    assert beta_by_variant["max"] == {1.0}
    assert beta_by_variant["var"] == {1.0}
    assert beta_by_variant["beta"] == {0.5, 1.0, 2.0}


def test_default_search_space_contains_only_valid_variant_beta_pairs(
    pipeline,
    annotations,
    ground_truth,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        strategy="full",
    )

    configurations = benchmark.configurations()

    for config in configurations:
        if config["variant"] in {"max", "var"}:
            assert config["beta"] == 1.0
        elif config["variant"] == "beta":
            assert config["beta"] in {0.5, 1.0, 2.0}
        else:
            pytest.fail(
                f"Unexpected variant: {config['variant']}"
            )


def test_default_search_space_is_used_when_not_provided(
    pipeline,
    annotations,
    ground_truth,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        strategy="full",
    )

    assert benchmark.search_space == DEFAULT_SEARCH_SPACE


# ---------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------

def test_run_evaluates_all_configurations(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    returned = benchmark.run()

    assert returned is benchmark

    results = benchmark.get_results()

    assert len(results) == 4
    assert "rank" in results.columns
    assert "configuration_id" in results.columns
    assert "theta_filter" in results.columns
    assert "max_depth" in results.columns
    assert "jaccard" in results.columns
    assert "precision" in results.columns
    assert "recall" in results.columns
    assert "exact_match" in results.columns


def test_best_configuration_is_selected(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    benchmark.run()

    assert benchmark.get_best_config() == {
        "theta_filter": 0.3,
        "max_depth": 6,
    }

    assert benchmark.get_best_score() == pytest.approx(0.95)


def test_results_are_ranked(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    benchmark.run()

    results = benchmark.get_results()

    assert results["rank"].tolist() == [1, 2, 3, 4]

    scores = results["jaccard"].tolist()

    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------
# Selection direction
# ---------------------------------------------------------------------

def test_min_selection_direction(
    pipeline,
    annotations,
    ground_truth,
):
    search_space = {
        "theta_filter": [0.2, 0.3],
    }

    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=search_space,
        strategy="full",
        selection_metric="jaccard",
        selection_direction="min",
        verbose=False,
    )

    benchmark.run()

    assert benchmark.get_best_config() == {
        "theta_filter": 0.2,
    }


# ---------------------------------------------------------------------
# Top configurations
# ---------------------------------------------------------------------

def test_top_configs(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        top_k=2,
        verbose=False,
    )

    benchmark.run()

    top = benchmark.get_top_configs()

    assert len(top) == 2
    assert top.iloc[0]["rank"] == 1
    assert top.iloc[1]["rank"] == 2


def test_get_top_configs_custom_k(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    benchmark.run()

    assert len(benchmark.get_top_configs(3)) == 3


def test_invalid_top_configs_k(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    benchmark.run()

    with pytest.raises(ValueError):
        benchmark.get_top_configs(0)


# ---------------------------------------------------------------------
# Best pipeline
# ---------------------------------------------------------------------

def test_get_best_pipeline(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    benchmark.run()

    best_pipeline = benchmark.get_best_pipeline()

    assert isinstance(best_pipeline, FakePipeline)

    assert best_pipeline.dims == pipeline.dims
    assert best_pipeline.scale == pipeline.scale

    assert best_pipeline.theta_filter == 0.3
    assert best_pipeline.max_depth == 6


# ---------------------------------------------------------------------
# Access before run
# ---------------------------------------------------------------------

def test_results_require_run(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
    )

    with pytest.raises(RuntimeError):
        benchmark.get_results()

    with pytest.raises(RuntimeError):
        benchmark.get_best_config()

    with pytest.raises(RuntimeError):
        benchmark.get_best_score()

    with pytest.raises(RuntimeError):
        benchmark.get_best_pipeline()


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def test_report_contains_complete_summary(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        top_k=2,
        seed=123,
        verbose=False,
    )

    benchmark.run()

    report = benchmark.get_report()

    assert "benchmark" in report
    assert "search_space" in report
    assert "best_configuration" in report
    assert "top_configurations" in report

    assert report["benchmark"]["strategy"] == "full"
    assert report["benchmark"]["seed"] == 123
    assert report["benchmark"]["total_configurations"] == 4

    assert report["best_configuration"]["config"] == {
        "theta_filter": 0.3,
        "max_depth": 6,
    }

    assert len(report["top_configurations"]) == 2


def test_save_results(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
    tmp_path,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    benchmark.run()

    path = tmp_path / "results.csv"

    returned_path = benchmark.save_results(path)

    assert returned_path == path
    assert path.exists()

    saved = pd.read_csv(path)

    assert len(saved) == 4
    assert "jaccard" in saved.columns


def test_save_report(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
    tmp_path,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="full",
        verbose=False,
    )

    benchmark.run()

    path = tmp_path / "report.json"

    returned_path = benchmark.save_report(path)

    assert returned_path == path
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["best_configuration"]["score"] == pytest.approx(
        benchmark.get_best_score()
    )


# ---------------------------------------------------------------------
# Full end-to-end benchmark workflow
# ---------------------------------------------------------------------

def test_end_to_end_workflow(
    pipeline,
    annotations,
    ground_truth,
    small_search_space,
    tmp_path,
):
    benchmark = PolarizedTreesBenchmark(
        pipeline=pipeline,
        annotations=annotations,
        ground_truth=ground_truth,
        search_space=small_search_space,
        strategy="random",
        n_runs=3,
        seed=42,
        metrics=[
            "jaccard",
            "precision",
            "recall",
        ],
        selection_metric="recall",
        top_k=2,
        verbose=False,
    )

    benchmark.run()

    assert len(benchmark.get_results()) == 3

    best_config = benchmark.get_best_config()
    best_score = benchmark.get_best_score()
    best_pipeline = benchmark.get_best_pipeline()

    assert isinstance(best_config, dict)
    assert isinstance(best_score, float)
    assert isinstance(best_pipeline, FakePipeline)

    report = benchmark.get_report()

    assert report["benchmark"]["selection_metric"] == "recall"

    results_path = benchmark.save_results(
        tmp_path / "benchmark_results.csv"
    )

    report_path = benchmark.save_report(
        tmp_path / "benchmark_report.json"
    )

    assert results_path.exists()
    assert report_path.exists()