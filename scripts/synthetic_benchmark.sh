#!/usr/bin/env bash
# The hyperparameter search (800 random configurations, 10 corpora): tens of minutes. It checkpoints every 50
# configurations in modules/benchmarks/benchmark_results/checkpoint/, so an interrupted run resumes.
# Writes benchmark_results.csv, benchmark_configuration_summary.csv, top_configurations.csv, the report.
source "$(dirname "$0")/_common.sh"
run_nb modules/benchmarks/notebooks/treesbenchmark.ipynb
