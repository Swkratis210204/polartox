# DICES — Real-Data Inference

This folder applies Polarized Trees to the **DICES-350** and **DICES-990**
datasets: real conversational-safety annotations, rated by a diverse pool
of annotators, with four socio-demographic (SCD) dimensions per annotator
(`gender`, `age`, `education`, `race`) and an overall bias rating
(`Q3_bias_overall`, encoded `No / Unsure / Yes` → `1 / 2 / 3`).

Unlike the rest of the repository's demos, **this is real annotation data,
not synthetic**: there is no known ground truth, so recovery metrics
(jaccard/precision/recall/exact match) are not used here. The output is
F/C/P and diagnostics only — the same ground-truth-free outputs
`PolarizedTreesPipeline` produces on any real dataset.

## Contents

- [`DICES_polarized_trees_end_to_end.ipynb`](DICES_polarized_trees_end_to_end.ipynb)
  — the full workflow: load and preprocess DICES-350/990, keep the four
  SCD dimensions plus `Q3_bias_overall`, validate/EDA the annotation-level
  data, compute nDFU per item, run `PolarizedTreesPipeline` in inference
  mode using the five configurations selected during synthetic model
  selection (see [`../benchmarks/README.md`](../benchmarks/README.md)),
  and report F/C/P and diagnostics for each configuration.
- `DICES-350_diagnostics_all_configs.csv` / `DICES-990_diagnostics_all_configs.csv`
  — the saved diagnostics table (retention rate, mean leaves/depth,
  residual nDFU, top-split PRG, indeterminate rate) for each of the five
  selected configurations, on each dataset.
- `dices990_polarized_tree.png` — an example recovered tree from DICES-990.

## Why this notebook exists

The rest of the repository (`data_gen/`, `polarized_tree/`, `polarized_trees/`,
`benchmarks/`) validates and tunes Polarized Trees on synthetic data with
known ground truth. This notebook is the real-data counterpart: it takes
the configuration selected on synthetic corpora and applies it, unmodified,
to actual annotated conversations — showing what the method actually
recovers when there is no known answer to check against.

Across both datasets, the trees select `race`, `gender`, and `age` as the
most frequent splitting dimensions, and the strongest subgroup-level PRG
values tend to come from combinations of multiple SCD dimensions rather
than any single one — see the notebook's own results sections for the
detailed per-configuration breakdown on DICES-350 and DICES-990.

## Method documentation

For what F/C/P and the diagnostics actually mean, and how the method
works, see [`../polarized_trees/README.md`](../polarized_trees/README.md).
For the single-tree API used to build and inspect any one item's tree
directly, see [`../polarized_tree/README.md`](../polarized_tree/README.md).
