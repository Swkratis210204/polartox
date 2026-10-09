#!/usr/bin/env bash
# The hyperparameter search (800 random configurations, 3 corpora): hours. It checkpoints every 50
# configurations in benchmarks/benchmark_results/checkpoint/, so an interrupted run resumes.
# Writes benchmark_results.csv, benchmark_configuration_summary.csv, top_configurations.csv, the report.
source "$(dirname "$0")/_common.sh"
run_nb benchmarks/notebooks/treesbenchmark.ipynb
