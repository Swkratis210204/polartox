#!/usr/bin/env bash
# Regenerates the synthetic benchmark corpora A to J and the unseen one (modules/benchmarks/benchmark_data/).
# Seeded, so it reproduces the same files; the benchmark search must be run again afterwards.
source "$(dirname "$0")/_common.sh"
run_nb modules/benchmarks/notebooks/datasetdemo.ipynb
