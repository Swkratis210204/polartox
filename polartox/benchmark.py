import inspect
import itertools
import json
import random
import time
from pathlib import Path

import pandas as pd


DEFAULT_SEARCH_SPACE = {
    "theta_filter": [0.2, 0.3, 0.4],
    "min_size_frac": [0.02, 0.03, 0.05],
    "max_depth": [4, 6, 8],
    "variant": ["max", "var", "beta"],
    "h": [0.05, 0.10, 0.15, 0.20],
    "relative_h": [False, True],
    "theta_stop": [0.05, 0.10, 0.15],
}

DEFAULT_METRICS = [
    "jaccard",
    "precision",
    "recall",
    "exact_match",
]

DEFAULT_SELECTION_METRIC = "jaccard"


class PolarizedTreesBenchmark:

    def __init__(
        self,
        pipeline,
        annotations,
        ground_truth,
        search_space=None,
        strategy="random",
        n_runs=800,
        seed=0,
        metrics=None,
        selection_metric=None,
        selection_direction="max",
        top_k=10,
        verbose=True,
    ):
        self.pipeline = pipeline
        self.annotations = annotations
        self.ground_truth = ground_truth

        if search_space is None:
            self.search_space = {
                key: list(values)
                for key, values in DEFAULT_SEARCH_SPACE.items()
            }
        elif isinstance(search_space, dict):
            self.search_space = {
                key: list(values)
                for key, values in search_space.items()
            }
        elif isinstance(search_space, (list, tuple)):
            self.search_space = [
                dict(config)
                for config in search_space
            ]
        else:
            raise TypeError(
                "search_space must be a dictionary or a list "
                "of configuration dictionaries."
            )

        self.strategy = strategy
        self.n_runs = n_runs
        self.seed = seed

        self.metrics = (
            DEFAULT_METRICS.copy()
            if metrics is None
            else list(metrics)
        )

        self.selection_metric = (
            DEFAULT_SELECTION_METRIC
            if selection_metric is None
            else selection_metric
        )

        self.selection_direction = selection_direction
        self.top_k = top_k
        self.verbose = verbose

        self.results_ = None
        self.best_config_ = None
        self.best_score_ = None
        self.best_pipeline_ = None
        self.top_configs_ = None
        self.runtime_ = None

        self._validate_inputs()

    def _validate_inputs(self):
        if not isinstance(self.annotations, pd.DataFrame):
            raise TypeError(
                "annotations must be a pandas DataFrame."
            )

        required_columns = {"text_id", "rating"}
        missing = required_columns - set(self.annotations.columns)

        if missing:
            raise ValueError(
                "annotations is missing required columns: "
                f"{sorted(missing)}"
            )

        if self.annotations.empty:
            raise ValueError("annotations cannot be empty.")

        if not isinstance(self.ground_truth, dict):
            raise TypeError("ground_truth must be a dictionary.")

        annotation_ids = set(self.annotations["text_id"].unique())
        ground_truth_ids = set(self.ground_truth.keys())

        if annotation_ids != ground_truth_ids:
            normalized_ids = set()
            for text_id in ground_truth_ids:
                try:
                    normalized_ids.add(int(text_id))
                except (TypeError, ValueError):
                    normalized_ids.add(text_id)
            missing_ground_truth = annotation_ids - normalized_ids
        else:
            missing_ground_truth = annotation_ids - ground_truth_ids

        if missing_ground_truth:
            raise ValueError(
                "ground_truth is missing entries for text_ids: "
                f"{sorted(missing_ground_truth)[:10]}"
            )

        if self.strategy not in {"full", "random"}:
            raise ValueError(
                "strategy must be either 'full' or 'random'."
            )

        if self.strategy == "random":
            if not isinstance(self.n_runs, int):
                raise TypeError("n_runs must be an integer.")
            if self.n_runs < 1:
                raise ValueError("n_runs must be at least 1.")

        valid_metrics = {
            "jaccard",
            "precision",
            "recall",
            "exact_match",
        }

        invalid_metrics = set(self.metrics) - valid_metrics

        if invalid_metrics:
            raise ValueError(
                f"Unknown metrics: {sorted(invalid_metrics)}"
            )

        if not self.metrics:
            raise ValueError("At least one metric must be requested.")

        if self.selection_metric not in self.metrics:
            raise ValueError(
                "selection_metric must be included in metrics."
            )

        if self.selection_direction not in {"max", "min"}:
            raise ValueError(
                "selection_direction must be 'max' or 'min'."
            )

        if not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer.")

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self._validate_search_parameters()

    def _validate_search_parameters(self):
        signature = inspect.signature(type(self.pipeline).__init__)

        valid_parameters = {
            name
            for name in signature.parameters
            if name != "self"
        }

        if isinstance(self.search_space, dict):
            invalid_parameters = (
                set(self.search_space) - valid_parameters
            )
        else:
            invalid_parameters = set()

            for config in self.search_space:
                invalid_parameters.update(
                    set(config) - valid_parameters
                )

        if invalid_parameters:
            raise ValueError(
                "Unknown pipeline parameters in search_space: "
                f"{sorted(invalid_parameters)}"
            )

        if isinstance(self.search_space, dict):
            if not self.search_space:
                raise ValueError("search_space cannot be empty.")

            for parameter, values in self.search_space.items():
                if not isinstance(values, (list, tuple)):
                    raise TypeError(
                        f"Search values for '{parameter}' "
                        "must be a list or tuple."
                    )
                if len(values) == 0:
                    raise ValueError(
                        f"Search space for '{parameter}' cannot be empty."
                    )
        else:
            if not self.search_space:
                raise ValueError("search_space cannot be empty.")

            for i, config in enumerate(self.search_space):
                if not isinstance(config, dict):
                    raise TypeError(
                        f"Configuration {i} must be a dictionary."
                    )
                if not config:
                    raise ValueError(
                        f"Configuration {i} cannot be empty."
                    )

    def _generate_configurations(self):
        # Explicit configuration list:
        # use the configurations exactly as supplied.
        if isinstance(self.search_space, (list, tuple)):
            configurations = [
                dict(config)
                for config in self.search_space
            ]

        # Dictionary search space.
        else:
            # The paper search space has a conditional relationship
            # between `variant` and `beta`.
            #
            #   max  -> beta = 1.0
            #   var  -> beta = 1.0
            #   beta -> beta = 0.5, 1.0, 2.0
            #
            # `beta` is intentionally NOT part of DEFAULT_SEARCH_SPACE.
            # It is generated here conditionally from `variant`.

            if (
                "variant" in self.search_space
                and "beta" not in self.search_space
            ):
                other_keys = [
                    key
                    for key in self.search_space
                    if key != "variant"
                ]

                base_combinations = itertools.product(
                    *(
                        self.search_space[key]
                        for key in other_keys
                    )
                )

                configurations = []

                for values in base_combinations:
                    base_config = dict(
                        zip(other_keys, values)
                    )

                    for variant in self.search_space["variant"]:

                        if variant in {"max", "var"}:
                            beta_values = [1.0]

                        elif variant == "beta":
                            beta_values = [0.5, 1.0, 2.0]

                        else:
                            raise ValueError(
                                f"Unknown variant '{variant}'."
                            )

                        for beta in beta_values:
                            configurations.append(
                                {
                                    **base_config,
                                    "variant": variant,
                                    "beta": beta,
                                }
                            )

            # Ordinary dictionary search space.
            else:
                keys = list(self.search_space)

                combinations = itertools.product(
                    *(
                        self.search_space[key]
                        for key in keys
                    )
                )

                configurations = [
                    dict(zip(keys, values))
                    for values in combinations
                ]

        # Apply search strategy.
        if self.strategy == "full":
            selected = configurations

        else:
            rng = random.Random(self.seed)

            n = min(
                self.n_runs,
                len(configurations),
            )

            selected = rng.sample(
                configurations,
                n,
            )

        return selected

    def configurations(self):
        """Return the configurations that will be evaluated."""
        return self._generate_configurations()

    def _build_pipeline(self, config):
        signature = inspect.signature(type(self.pipeline).__init__)
        params = {}

        for name, parameter in signature.parameters.items():
            if name == "self":
                continue

            if hasattr(self.pipeline, name):
                params[name] = getattr(self.pipeline, name)
            elif parameter.default is not inspect.Parameter.empty:
                params[name] = parameter.default
            else:
                raise ValueError(
                    f"Cannot determine value for pipeline parameter '{name}'."
                )

        params.update(config)
        return type(self.pipeline)(**params)

    def _normalized_ground_truth(self):
        annotation_ids = set(self.annotations["text_id"].unique())

        if set(self.ground_truth) == annotation_ids:
            return self.ground_truth

        normalized = {}

        for text_id, value in self.ground_truth.items():
            try:
                text_id = int(text_id)
            except (TypeError, ValueError):
                pass
            normalized[text_id] = value

        missing = annotation_ids - set(normalized)

        if missing:
            raise ValueError(
                "ground_truth is missing entries for text_ids: "
                f"{sorted(missing)[:10]}"
            )

        return normalized

    def _python_value(self, value):
        if hasattr(value, "item"):
            return value.item()

        if isinstance(value, dict):
            return {
                key: self._python_value(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                self._python_value(val)
                for val in value
            ]

        return value

    def _run_one_configuration(self, config):
        candidate = self._build_pipeline(config)

        output = candidate.run_full_evaluation(
            self.annotations,
            ground_truth=self._normalized_ground_truth(),
            verbose=False,
        )

        recovery = output["recovery"]
        row = dict(config)

        for metric in self.metrics:
            row[metric] = float(recovery[metric].mean())

        return row

    def run(self):
        configurations = self._generate_configurations()

        if not configurations:
            raise RuntimeError("No configurations were generated.")

        start = time.time()
        rows = []
        total = len(configurations)

        for i, config in enumerate(configurations, start=1):
            row = self._run_one_configuration(config)
            row["configuration_id"] = i
            rows.append(row)

            if self.verbose:
                score = row[self.selection_metric]
                print(
                    f"[{i}/{total}] "
                    f"{self.selection_metric}={score:.4f}"
                )

        self.runtime_ = time.time() - start
        results = pd.DataFrame(rows)

        parameter_columns = list(
            self.search_space.keys()
        ) if isinstance(self.search_space, dict) else sorted(
            {
                key
                for config in configurations
                for key in config
            }
        )

        columns = [
            "configuration_id",
            *parameter_columns,
            *self.metrics,
        ]

        results = results[columns]

        results = results.sort_values(
            by=self.selection_metric,
            ascending=self.selection_direction == "min",
        ).reset_index(drop=True)

        results["rank"] = range(1, len(results) + 1)

        results = results[
            [
                "rank",
                "configuration_id",
                *parameter_columns,
                *self.metrics,
            ]
        ]

        self.results_ = results
        best_row = results.iloc[0]

        self.best_score_ = float(
            best_row[self.selection_metric]
        )

        self.best_config_ = {
            parameter: self._python_value(best_row[parameter])
            for parameter in parameter_columns
            if pd.notna(best_row[parameter])
        }

        # Store the selected pipeline so the user does not need
        # to reconstruct it manually from best_config.
        self.best_pipeline_ = self._build_pipeline(
            self.best_config_
        )

        self.top_configs_ = results.head(self.top_k).copy()

        return self

    def get_results(self):
        """Return results for all evaluated configurations."""
        if self.results_ is None:
            raise RuntimeError("Run the benchmark first.")
        return self.results_.copy()

    def get_best_config(self):
        """Return the selected best configuration."""
        if self.best_config_ is None:
            raise RuntimeError("Run the benchmark first.")
        return self.best_config_.copy()

    def get_best_score(self):
        """Return the score of the selected configuration."""
        if self.best_score_ is None:
            raise RuntimeError("Run the benchmark first.")
        return self.best_score_

    def get_best_pipeline(self):
        """Return the pipeline configured with the selected hyperparameters."""
        if self.best_pipeline_ is None:
            raise RuntimeError("Run the benchmark first.")
        return self.best_pipeline_

    def get_top_configs(self, k=None):
        """Return the top-ranked configurations."""
        if self.results_ is None:
            raise RuntimeError("Run the benchmark first.")

        if k is None:
            k = self.top_k

        if not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        return self.results_.head(k).copy()

    def summary(self, k=None):
        """Print a concise benchmark summary and return top results."""
        if self.results_ is None:
            raise RuntimeError("Run the benchmark first.")

        if k is None:
            k = self.top_k

        print(
            f"Configurations evaluated: {len(self.results_)}"
        )
        print(
            f"Selection metric: {self.selection_metric}"
        )
        print(
            f"Best score: {self.best_score_:.4f}"
        )
        print("\nBest configuration:")

        for parameter, value in self.best_config_.items():
            print(f"  {parameter}: {value}")

        print(
            f"\nRuntime: {self.runtime_:.2f} seconds"
        )
        print("\nTop configurations:")

        return self.get_top_configs(k)

    def get_report(self):
        """Return the complete benchmark report as a dictionary."""
        if self.results_ is None:
            raise RuntimeError("Run the benchmark first.")

        parameter_space = (
            self.search_space
            if isinstance(self.search_space, dict)
            else self.search_space
        )

        report = {
            "benchmark": {
                "strategy": self.strategy,
                "n_runs": self.n_runs,
                "seed": self.seed,
                "metrics": self.metrics,
                "selection_metric": self.selection_metric,
                "selection_direction": self.selection_direction,
                "top_k": self.top_k,
                "total_configurations": len(self.results_),
                "runtime_seconds": self.runtime_,
            },
            "search_space": parameter_space,
            "best_configuration": {
                "config": self.best_config_,
                "score": self.best_score_,
            },
            "top_configurations": (
                self.get_top_configs().to_dict(
                    orient="records"
                )
            ),
        }

        return self._python_value(report)

    def save_results(self, path="benchmark_results.csv"):
        """Save all configuration results to CSV."""
        if self.results_ is None:
            raise RuntimeError("Run the benchmark first.")

        path = Path(path)
        self.results_.to_csv(path, index=False)
        return path

    def save_report(self, path="benchmark_report.json"):
        """Save the complete benchmark report to JSON."""
        report = self.get_report()
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return path