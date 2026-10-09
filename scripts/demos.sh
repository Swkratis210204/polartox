#!/usr/bin/env bash
# The small demo notebooks: data generation, one tree, the tree module, the PEG comparison. Minutes.
source "$(dirname "$0")/_common.sh"
run_nb data_gen/datagen_demo.ipynb
run_nb polarized_trees/trees_demo.ipynb
run_nb tree_module/polarized_tree_demo.ipynb
run_nb peg_comparison/peg_comparison_demo.ipynb
