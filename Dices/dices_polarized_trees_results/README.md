# dices_polarized_trees_results — which cell writes which file

Everything here is written by `../DICES_polarized_trees_end_to_end.ipynb` (`scripts/dices.sh` runs it). The configuration is
labelled `1` in the tables and file names (`CONFIG = 1`, the selected configuration of the synthetic benchmark). Cells are
named by their section heading, not by number. Re-running overwrites; a file here that is **not** listed below is a leftover
of an earlier version of the notebook and can be deleted.

| Files | Cell that writes them |
|---|---|
| `csv_tables/<dataset>/comparison/…` (`overview`, `per_run`, `per_text`, `retention_curve`, `selected_settings`, `dimension_usage_setting1`, `depth_distribution_setting1`, `similarity_{ari,nmi,dims,first_split}_setting1`, `ari_setting1_<dataset>`, `ari_per_text_setting1_<dataset>`, `fcp/{F,C,P}_setting1_<dataset>_<formulation>`) | **13.1 SAVE RESULTS AS CSV**: `comparison.save(...)` |
| `csv_tables/<dataset>/fcp_comparison_{F,C,P}.csv` | **13.1 SAVE RESULTS AS CSV**: `fcp_comparison(...)` |
| `latex_tables/<dataset>/config_1_<formulation>_{F,C,P}.tex`, `<dataset>_overview.tex`, `<dataset>_fcp_comparison_{F,C,P}.tex` | **13.2 SAVE RESULTS AS LATEX** |
| `figures/retention_curve.png` | **10.** Retention: how many texts does `theta_filter` keep? |
| `figures/depth_cumulative.png` | **10.** Depth: the trees by depth |
| `figures/overall_dimensions.png`, `.pdf` | **10.** One dimension or several? |
| `figures/similarity_groups.png`, `figures/similarity_splits.png` | **10.** How close are the trees of two formulations? |
| `figures/subgroup_overlap.png` | **11.** Do the formulations find the same subgroups? |
| `sensitivity_gender_only/…` | **14.** Sensitivity: keep "Other" and "Multiracial" as categories |

`<dataset>` is `DICES-350` or `DICES-990`.

## Static files (no code in the notebook makes them)

- `figures/dices_ndfu_distribution.png`, `.pdf`: the nDFU histogram of each dataset with the `theta_filter` line. The notebook only
  plots the nDFU distribution on screen (section 6 and 7); the code that saved this figure is not in the notebook or its git history.
- `../dices990_polarized_tree.png`: an example tree of DICES-990.

The synthetic counterpart of these tables and figures is in `benchmarks/benchmark_results/peg_comparison/`
(`benchmarks/benchmark_results/README.md`).
