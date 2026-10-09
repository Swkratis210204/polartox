#!/usr/bin/env bash
# The DICES end-to-end notebook: trees on DICES-350 and DICES-990, the tables, the LaTeX tables and
# the figures in Dices/dices_polarized_trees_results/.
source "$(dirname "$0")/_common.sh"
run_nb Dices/DICES_polarized_trees_end_to_end.ipynb
