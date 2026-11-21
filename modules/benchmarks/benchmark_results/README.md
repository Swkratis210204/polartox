# benchmark_results — which notebook writes which file

Every file in this folder is written by one of the notebooks in `../notebooks/`, in this order:

    datasetdemo.ipynb  →  ../benchmark_data/
    treesbenchmark.ipynb  →  this folder (top level) and checkpoint/
    resultsexploration.ipynb  →  exploration/
    pegcomparison.ipynb  →  peg_comparison/

Re-running a notebook overwrites its files. The scripts in `../../../scripts/` run them (`synthetic_benchmark.sh`,
`synthetic_analysis.sh`). A file in these folders that is **not** listed below is a leftover of an earlier version
of a notebook: nothing reads it, and it can be deleted.

Cells are named by their section heading, not by number.

## The search — `treesbenchmark.ipynb` (tens of minutes on 6 workers; the only notebook that is not re-run by the quick path)

| File | What it is | Call in the save cell |
|---|---|---|
| `benchmark_configuration_summary.csv` | one row per configuration (the ten corpora, A to J, averaged) | `benchmark.save_results` |
| `benchmark_results.csv` | one row per configuration and corpus | `benchmark.save_group_results` |
| `benchmark_text_results.csv` | one row per configuration, corpus and text (about 56 MB, **not tracked by git**: run `treesbenchmark.ipynb` to create it; `resultsexploration.ipynb` reads it) | `benchmark.save_text_results` |
| `benchmark_report.json` | settings, search space, best and top configurations | `benchmark.save_report` |
| `top_configurations.csv` | the top 20 configurations; its row `0` is the setting used everywhere else | `benchmark.get_top_configs().to_csv` |
| `checkpoint/` | the search state, saved every 50 configurations so that an interrupted run resumes (not tracked by git; delete it before a new search on different corpora, or the run resumes from the old one) | `checkpoint_dir=` of the run |

## `exploration/` — `resultsexploration.ipynb`, last code cell ("Saved to")

`top20_formulations.csv`, `hyperparameter_effects.csv`, `best_per_corpus.csv`, `recovery_by_k.csv`,
`top20_by_corpus_{exact_match,jaccard,precision,recall}.csv` and the four `relative_h_*_base_relative_h_True.csv`
files (`grid`, `grid_by_corpus`, `best_h`, `tree_size`).

## `peg_comparison/` — `pegcomparison.ipynb`

The setting label `setting0` is row 0 of `top_configurations.csv` (`SELECTED = [0]`). A to J are the corpora with
ground truth, `inference_unseen` the unseen corpus (no ground truth is used).

| Where | Section of the notebook that makes them |
|---|---|
| `abc/`: `selected_settings.csv`, `per_run.csv`, `overview.csv`, `per_text.csv`, `retention_curve.csv`, `dimension_usage_setting0.csv`, `depth_distribution_setting0.csv`, `similarity_{ari,nmi,dims,first_split}_setting0.csv`, `ari_setting0_<corpus>.csv`, `ari_per_text_setting0_<corpus>.csv`, `recovery_by_k.csv`, `ari_by_k.csv`, `fcp/{F,C,P}_setting0_<corpus>_<formulation>.csv` (the tables of the ten corpora A to J, for the best row after the re-scoring) | **12. Save**: `comparison.save(OUT_DIR / "abc")` |
| `rescored_all_corpora.csv`: the top 20 re-scored on A to J with their own formulation (`jaccard_ABC`, `jaccard_all`, `jaccard_worst`, the Jaccard of each corpus) | **2b. Re-score the top 20** |
| `inference_unseen/`: the same tables for the unseen corpus (no recovery tables), laid out like the DICES `csv_tables/<dataset>/comparison/` | **12. Save**: `unseen_comparison.save(OUT_DIR / INFERENCE_CORPUS)` |
| `figures/`: `ari_heatmap_setting0.png`, `nmi_heatmap_setting0.png` | **5.** Figures: ARI and NMI between formulations |
| `figures/`: `retention_curve.png`, `depth_cumulative.png`, `overall_dimensions.png/.pdf`, `similarity_groups.png`, `similarity_splits.png` | **8.** The inference analysis, as on DICES |
| `figures/`: `subgroup_overlap.png` | **9.** F, C and P |
| `synthetic_indicators/`: `1_indicators.csv`, `2_trees_by_number_of_dimensions.csv`, `3_{ari,nmi}_{mean_all,inference_unseen}.csv`, `4_rank_order.csv`, `5_corpora.csv`, `5_dimension_levels.csv` | **11.** Synthetic indicators (each table is saved by the cell that shows it) |

Sections 8 to 10 repeat, on the unseen corpus, the analysis of `Dices/DICES_polarized_trees_end_to_end.ipynb` (its sections 10 to 12)
with the same figure names, so the two can be compared; `Dices/dices_polarized_trees_results/README.md` has the DICES side.

The F, C and P tables of every formulation are in `abc/fcp/` (A to J) and `inference_unseen/fcp/` (unseen corpus).

    peg_comparison/
    ├── abc/                    the tables of the ten corpora, A to J (the folder keeps its old name)
    ├── inference_unseen/       the same tables for the unseen corpus
    ├── figures/
    └── synthetic_indicators/
