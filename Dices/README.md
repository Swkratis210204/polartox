# DICES — Real-Data Inference

This folder applies Polarized Trees to the **DICES-350** and **DICES-990**
datasets: real conversational-safety annotations, rated by a diverse pool
of annotators, with four socio-demographic (SCD) dimensions per annotator
(`gender`, `age`, `education`, `race`) and an overall bias rating
(`Q3_bias_overall`, encoded `No / Unsure / Yes` → `1 / 2 / 3`).

Unlike the rest of the repository's demos, **this is real annotation data,
not synthetic**: there is no known ground truth, so recovery metrics
(jaccard/precision/recall/exact match) are not used here. The analysis has two
parts, both ground-truth-free: a **comparison of the PEG formulations** on the trees
they build (`polartox.peg_comparison`), and the **F/C/P tables** of every
formulation.

## Contents

- [`DICES_polarized_trees_end_to_end.ipynb`](DICES_polarized_trees_end_to_end.ipynb)
  — the full workflow: load and preprocess DICES-350/990, keep the four
  SCD dimensions plus `Q3_bias_overall`, validate/EDA the annotation-level
  data, compute nDFU per item, then run the configuration selected on the synthetic
  benchmark (see [`../benchmarks/README.md`](../benchmarks/README.md)) once per PEG
  formulation (`max`, `weighted`, `min`, `mean`, `harmonic`), **everything else fixed**,
  through `PEGComparison`. It then:
  - compares the formulations overall: retention, leaves, depth (also by depth,
    accumulated), annotators per leaf, residual nDFU, split PEG, indeterminate
    leaves, the trees that never split, the share of polarized texts whose tree
    splits on more than one dimension, and which dimensions are used;
  - measures how close the trees of any two formulations are: ARI and NMI of the
    groups they make of the annotators, the dimensions they split on, and their
    first split;
  - compares F, C and P between the formulations, side by side, and whether the
    formulations find the same subgroups; each formulation's own tables follow;
  - shows the texts on which the formulations disagree most, and a text that every
    formulation splits more than once (depth 2 or more in all five), with the tree
    each builds;
  - saves everything, tables and figures.
- `dices_polarized_trees_results/` — the saved results: `csv_tables/`, `latex_tables/`
  and `figures/`. For each dataset (`DICES-350`, `DICES-990`): the overview of the
  comparison (`<dataset>_overview`), F, C and P side by side
  (`fcp_comparison_F`, `_C`, `_P`), and the comparison tables in
  `csv_tables/<dataset>/comparison/` (per text, retention curve, dimension usage,
  similarity matrices, and the F, C and P tables of every formulation in `fcp/`).
- `dices990_polarized_tree.png` — an example recovered tree from DICES-990.

> **Note.** The notebook needs `polartox >= 0.8.1` (`pip install -e .` from the
> repository root). Files in `dices_polarized_trees_results/` that are not listed above
> (`config_<n>_F/C/P`, `<dataset>_diagnostics_*`, and the `config_<n>_<formulation>_*`
> files of an earlier version of the notebook) are from earlier runs and are not
> produced any more.

## Why this notebook exists

The rest of the repository (`data_gen/`, `tree_module/`, `polarized_trees/`,
`benchmarks/`) validates and tunes Polarized Trees on synthetic data with
known ground truth. This notebook is the real-data counterpart: it takes
the configuration selected on synthetic corpora and applies it, unmodified,
to actual annotated conversations — showing what the method actually
recovers when there is no known answer to check against.

See the notebook's own results sections for the comparison of the formulations and
the F, C and P tables on DICES-350 and DICES-990.

## Method documentation

For what F/C/P and the diagnostics actually mean, and how the method
works, see [`../polarized_trees/README.md`](../polarized_trees/README.md).
For the single-tree API used to build and inspect any one item's tree
directly, see [`../tree_module/README.md`](../tree_module/README.md).
