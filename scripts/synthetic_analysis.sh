#!/usr/bin/env bash
# Everything computed from the saved benchmark results: the exploration tables and the comparison of
# the five PEG formulations (it also checks that the chosen rows reproduce the saved benchmark numbers).
source "$(dirname "$0")/_common.sh"
run_nb benchmarks/notebooks/resultsexploration.ipynb
run_nb benchmarks/notebooks/pegcomparison.ipynb
