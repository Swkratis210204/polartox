import inspect
import itertools
import json
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
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

# Metrics that get distribution statistics next to their mean.
# exact_match is 0/1, so its mean is already the whole distribution.
STAT_METRICS = ["jaccard", "precision", "recall"]

STATS = {
    "median": lambda s: s.median(),
    "std": lambda s: s.std(),
    "q1": lambda s: s.quantile(0.25),
    "q3": lambda s: s.quantile(0.75),
    "min": lambda s: s.min(),
    "max": lambda s: s.max(),
}

GROUP_COLUMN = "corpus"

# Set once per worker process by _init_worker (see n_jobs in run()).
_WORKER_BENCHMARK = None


def _init_worker(benchmark):
    global _WORKER_BENCHMARK
    _WORKER_BENCHMARK = benchmark


def _run_in_worker(config):
    return _WORKER_BENCHMARK._run_one_configuration(config)


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
        text_groups=None,
        keep_text_results=True,
        checkpoint_dir=None,
        checkpoint_every=50,
        n_jobs=1,
    ):
        self.pipeline = pipeline
        self.annotations = annotations
        self.ground_truth = ground_truth

        # Optional {text_id: corpus} mapping. When given, every metric
        # and statistic is computed per corpus and then averaged over
        # corpora. Without it all texts form a single group.
        self.text_groups = (
            None if text_groups is None else dict(text_groups)
        )
        self.keep_text_results = keep_text_results

        # When checkpoint_dir is set, run() saves its progress every
        # checkpoint_every configurations and resumes from that file.
        self.checkpoint_dir = (
            None if checkpoint_dir is None else Path(checkpoint_dir)
        )
        self.checkpoint_every = checkpoint_every

        # Worker processes used to evaluate configurations. Each worker
        # holds its own copy of the annotations (memory scales with it);
        # -1 means all CPUs.
        self.n_jobs = n_jobs

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
        self.group_results_ = None
        self.text_results_ = None
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

        if self.text_groups is not None:
            missing_groups = annotation_ids - {
                self._as_id(text_id) for text_id in self.text_groups
            }

            if missing_groups:
                raise ValueError(
                    "text_groups is missing entries for text_ids: "
                    f"{sorted(missing_groups)[:10]}"
                )

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

    @staticmethod
    def _as_id(text_id):
        try:
            return int(text_id)
        except (TypeError, ValueError):
            return text_id

    def _value_columns(self):
        """Metric columns in output order: mean, then its statistics."""
        columns = []

        for metric in self.metrics:
            columns.append(metric)

            if metric in STAT_METRICS:
                columns.extend(f"{metric}_{stat}" for stat in STATS)

        return columns

    def _parameter_columns(self, configurations):
        """Search-space keys, plus generated keys such as `beta`."""
        if isinstance(self.search_space, dict):
            columns = list(self.search_space)
        else:
            columns = []

        extras = []

        for config in configurations:
            for key in config:
                if key not in columns and key not in extras:
                    extras.append(key)

        if not isinstance(self.search_space, dict):
            return sorted(extras)

        position = (
            columns.index("variant") + 1
            if "variant" in columns
            else len(columns)
        )

        return columns[:position] + extras + columns[position:]

    def _run_one_configuration(self, config):
        candidate = self._build_pipeline(config)

        output = candidate.run_full_evaluation(
            self.annotations,
            ground_truth=self._normalized_ground_truth(),
            verbose=False,
        )

        recovery = output["recovery"]

        if self.text_groups is None:
            labels = pd.Series(
                "all",
                index=recovery.index,
            )
        else:
            labels = recovery["text_id"].map(
                {
                    self._as_id(text_id): group
                    for text_id, group in self.text_groups.items()
                }
            )

            if labels.isna().any():
                raise ValueError(
                    "text_groups does not cover every evaluated text."
                )

        # One row per corpus: mean and statistics of the per-text values.
        group_rows = []

        for name, part in recovery.groupby(labels, sort=False):
            group_row = {
                GROUP_COLUMN: name,
                "n_texts_evaluated": len(part),
            }

            for metric in self.metrics:
                values = part[metric].astype(float)
                group_row[metric] = float(values.mean())

                if metric in STAT_METRICS:
                    for stat, function in STATS.items():
                        group_row[f"{metric}_{stat}"] = float(
                            function(values)
                        )

            group_rows.append(group_row)

        # One row per configuration: every column is the average of
        # the per-corpus values (average of the averages).
        group_frame = pd.DataFrame(group_rows)
        row = dict(config)

        for column in self._value_columns():
            row[column] = float(group_frame[column].mean())

        text_frame = None

        if self.keep_text_results:
            wanted = [
                column
                for column in [
                    "text_id",
                    "k_true",
                    "true_dims",
                    "found_dims",
                    *self.metrics,
                ]
                if column in recovery.columns
            ]

            text_frame = recovery[wanted].copy()
            text_frame.insert(0, GROUP_COLUMN, labels.to_numpy())

            for column in ("true_dims", "found_dims"):
                if column in text_frame.columns:
                    text_frame[column] = text_frame[column].map(
                        lambda dims: ",".join(map(str, dims))
                    )

        return row, group_rows, text_frame

    def _checkpoint_path(self):
        return self.checkpoint_dir / "benchmark_checkpoint.pkl"

    def _save_checkpoint(self, configurations, rows, group_rows, text_frames):
        """Atomically save progress, plus a readable partial summary."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path()
        temp = path.with_suffix(".tmp")

        with open(temp, "wb") as f:
            pickle.dump(
                {
                    "configurations": configurations,
                    "rows": rows,
                    "group_rows": group_rows,
                    "text_frames": text_frames,
                },
                f,
            )

        os.replace(temp, path)

        pd.DataFrame(rows).to_csv(
            self.checkpoint_dir / "benchmark_partial_summary.csv",
            index=False,
        )

    def _load_checkpoint(self, configurations):
        """Return saved progress, or None if absent or for another search."""
        if self.checkpoint_dir is None:
            return None

        path = self._checkpoint_path()

        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except Exception:
            return None

        if state.get("configurations") != configurations:
            if self.verbose:
                print(
                    "Checkpoint belongs to a different search; ignoring it."
                )
            return None

        return state

    def _evaluate(self, configurations):
        """Yield _run_one_configuration output for each configuration, in order."""
        n_jobs = self.n_jobs

        if n_jobs is None or n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer or -1.")

        workers = min(n_jobs, len(configurations))

        if workers <= 1:
            for config in configurations:
                yield self._run_one_configuration(config)
            return

        # Workers already run in parallel: keep BLAS from adding its own
        # threads on top (inherited by the spawned processes).
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")

        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(self,),
        )

        try:
            yield from executor.map(_run_in_worker, configurations)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def run(self):
        configurations = self._generate_configurations()

        if not configurations:
            raise RuntimeError("No configurations were generated.")

        start = time.time()
        rows = []
        group_rows = []
        text_frames = []
        total = len(configurations)

        state = self._load_checkpoint(configurations)

        if state is not None:
            rows = state["rows"]
            group_rows = state["group_rows"]
            text_frames = state["text_frames"]

            if self.verbose:
                print(f"Resuming from checkpoint at {len(rows)}/{total}.")

        done = len(rows)
        pending = configurations[done:]

        # Results arrive in configuration order, whether they were
        # computed here or in worker processes.
        outputs = self._evaluate(pending)

        for i, (config, output) in enumerate(
            zip(pending, outputs), start=done + 1
        ):
            row, config_group_rows, text_frame = output
            row["configuration_id"] = i
            rows.append(row)

            for group_row in config_group_rows:
                group_rows.append(
                    {
                        "configuration_id": i,
                        **config,
                        **group_row,
                    }
                )

            if text_frame is not None:
                text_frame.insert(0, "configuration_id", i)
                text_frames.append(text_frame)

            if self.verbose:
                score = row[self.selection_metric]
                print(
                    f"[{i}/{total}] "
                    f"{self.selection_metric}={score:.4f}"
                )

            if (
                self.checkpoint_dir is not None
                and self.checkpoint_every
                and i % self.checkpoint_every == 0
                and i < total
            ):
                self._save_checkpoint(
                    configurations, rows, group_rows, text_frames
                )

                if self.verbose:
                    print(f"Checkpoint saved at {i}/{total}.")

        self.runtime_ = time.time() - start
        results = pd.DataFrame(rows)

        parameter_columns = self._parameter_columns(configurations)
        value_columns = self._value_columns()

        results = results[
            [
                "configuration_id",
                *parameter_columns,
                *value_columns,
            ]
        ]

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
                *value_columns,
            ]
        ]

        rank_of = dict(
            zip(results["configuration_id"], results["rank"])
        )

        group_results = pd.DataFrame(group_rows)
        group_results.insert(
            0,
            "rank",
            group_results["configuration_id"].map(rank_of),
        )
        group_results = group_results[
            [
                "rank",
                "configuration_id",
                *parameter_columns,
                GROUP_COLUMN,
                "n_texts_evaluated",
                *value_columns,
            ]
        ].sort_values(
            "rank",
            kind="stable",
        ).reset_index(drop=True)

        self.group_results_ = group_results

        if text_frames:
            text_results = pd.concat(
                text_frames,
                ignore_index=True,
            )
            text_results.insert(
                0,
                "rank",
                text_results["configuration_id"].map(rank_of),
            )
            self.text_results_ = text_results.sort_values(
                "rank",
                kind="stable",
            ).reset_index(drop=True)
        else:
            self.text_results_ = None

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

    def get_group_results(self):
        """Return per-corpus results (one row per configuration and corpus)."""
        if self.group_results_ is None:
            raise RuntimeError("Run the benchmark first.")
        return self.group_results_.copy()

    def get_text_results(self):
        """Return raw per-text recovery values for every configuration."""
        if self.text_results_ is None:
            raise RuntimeError(
                "No per-text results. Run the benchmark with "
                "keep_text_results=True."
            )
        return self.text_results_.copy()

    def save_group_results(self, path="benchmark_runs.csv"):
        """Save per-corpus results to CSV."""
        path = Path(path)
        self.get_group_results().to_csv(path, index=False)
        return path

    def save_text_results(self, path="benchmark_text_results.csv"):
        """Save raw per-text recovery values to CSV."""
        path = Path(path)
        self.get_text_results().to_csv(path, index=False)
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