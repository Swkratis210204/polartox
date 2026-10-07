# PEG comparison — `polartox.peg_comparison`

`polartox.peg_comparison` compares the **PEG formulations** (`max`, `avg`,
`min`, `mean`, `harmonic`) on the same corpora, with **everything else held
fixed**. For each chosen setting and each corpus it runs one
`PolarizedTreesPipeline` per formulation, and analyses the trees they build.

**Ground truth is optional, and it decides what you get:**

- **Without ground truth** (real data such as DICES, or synthetic data used blind)
  you get the trees themselves: how many texts survive the filter, how many leaves,
  how deep, how large the groups, how much polarization is left, the PEG of the
  splits, which dimensions are used, whether trees split on one dimension or
  several, how close the trees of two formulations are, and the F, C and P tables of
  every formulation. There is no right answer to compare against, so the
  formulations are compared on the trees they build.
- **With ground truth** (`polartox.datagen`) you get all of that, plus
  **recovery** of the true dimensions (Jaccard, precision, recall, exact match),
  by the true number of active dimensions k.

The two can be mixed: some corpora with ground truth and some without.

It is the last step of the package's chain:

```text
datagen  →  polarized_tree  →  pipeline  →  benchmark  →  peg_comparison
corpora      one text's tree    a corpus      search the      the best settings,
with truth                      of trees      settings        one formulation at a time
```

## Quickstart

With ground truth, settings taken from a benchmark:

```python
from polartox import PEGComparison

comparison = PEGComparison.from_benchmark(benchmark)        # corpora, dims, scale
comparison.run(benchmark.top_configs_.head(3))              # each row's own variant is ignored
comparison.check_against_benchmark(benchmark.top_configs_)  # reproduces the benchmark's numbers

comparison.overview()          # recovery and the shape of the trees, per formulation
comparison.recovery_by_k()     # recovery by the true number of active dimensions
```

Without ground truth:

```python
comparison = PEGComparison({"DICES-350": annotations}, dims=["gender", "race", "age", "education"], scale=3)
comparison.run(dict(theta_filter=0.3, min_size_frac=0.02, max_depth=4,
                    h=0.2, relative_h=True, theta_stop=0.1, theta_pole=3))

comparison.overview()                     # the trees per formulation
comparison.retention_curve()              # texts kept for every theta_filter
comparison.depth_distribution(cumulative=True)
comparison.dimension_usage()              # which dimensions split, and split first
comparison.similarity("ari")              # how close are the trees of two formulations?
comparison.fcp_comparison(corpus="DICES-350", table="C")    # C of every formulation, side by side
```

## Inputs

- **`corpora`** — `{name: corpus}` where a corpus is the annotations alone (a
  DataFrame with `text_id`, `rating` and one column per dimension), or
  `(annotations, ground_truth)` as `AnnotatorPool.generate_dataset` returns them
  (`{text_id: {"active_dims": [...], ...}}`). A single corpus, or a
  `GeneratedDataset`, can be passed without the dict. Ground truth, when given,
  must cover every annotated text; text ids read back from JSON (strings) are matched
  to integer ids, as in the benchmark. Keep corpora separate: text ids only have to
  be unique within a corpus.
- **`settings`** (for `run`) — the non-PEG arguments of `PolarizedTreesPipeline`
  (`theta_filter`, `min_size_frac`, `max_depth`, `h`, `relative_h`, `theta_stop`,
  `theta_pole`, ...): a dict (one setting), a list of dicts, or a DataFrame such as
  `top_configs_` (one setting per row, labelled by its index; columns that are
  not pipeline arguments, such as the metrics, and empty cells are skipped).
  A `variant` in a setting is ignored: all formulations run with the rest.
  A misspelt argument in a dict raises `ValueError`, every pipeline is built
  before the first one runs, and a setting that keeps no text of a corpus fails
  with a message that names it.

Every table is keyed by **`setting`**: the index label of the settings row, the
position in a list, or `0` for a single dict. A *run* is one formulation at one setting.

## What it reports

**The trees (always).**

- `per_run()` / `overview()` — per setting, formulation and corpus (`overview` averages
  over the corpora, or the ones you name):
  retention, leaves per tree, depth (the deepest leaf, and the mean over leaves),
  annotators per leaf, residual nDFU (unweighted, and weighted by annotators), the
  share of nDFU the splits explained, the PEG of all splits and of the first, the
  share of indeterminate leaves, the trees that never split, the trees that split on
  two or more dimensions, the dimensions used per tree and the dimensions never used.
  `tree_statistics(pipeline)` gives the same for any pipeline.
- `per_text()` — the same numbers for every text (annotators, nDFU, whether it was
  retained, leaves, depth, first split, dimensions used, ...).
- `retention_curve(thetas)` — the share of texts kept for every `theta_filter`.
- `depth_distribution(setting, cumulative)` — the trees by depth.
- `dimension_usage(setting)` — per dimension: the share of trees that split on it,
  the share whose first split it is, and the splits per tree.

**How close are the trees of two runs? (always).**

- `similarity(measure, setting=None)` — a matrix over the runs: the formulations of a
  `setting`, or every (setting, formulation) without it. The value is the mean over
  the texts that both runs analysed, averaged over the corpora; `measure` is

  | measure | what it compares |
  |---|---|
  | `"ari"` | the adjusted Rand index of the groups the leaves make of a text's annotators (1 = the same groups, about 0 = chance) |
  | `"nmi"` | the normalized mutual information of the same groups (not corrected for chance, so it runs higher) |
  | `"dims"` | the Jaccard of the sets of dimensions the two trees split on |
  | `"first_split"` | the share of texts whose trees start with the same dimension |

  The first two compare what the trees do to the annotators, the last two how the
  trees are built. `ari_matrix`, `dims_agreement` and `first_split_agreement` are
  shortcuts for the formulations of one setting.
- `disagreement(setting)` — the texts ordered by how differently the formulations
  split them (lowest mean ARI first).
- `text_summary(corpus, text_id)` / `show_text(...)` — one text: one line per
  formulation (first split, dimensions used, leaves, depth) and the trees.

**F, C and P (always).**

- `fcp(setting, corpus, variant)` — the F, C and P tables of a run (of every
  formulation without `variant`).
- `fcp_comparison(setting, corpus, table, top)` — the F, C or P tables of all the
  formulations **side by side**: for F one row per dimension with the columns
  (formulation, depth); for C and P the union of the `top` first subgroups of every
  formulation, with the columns (formulation, `n_s` and `frac_toxic` / `mean_peg`),
  empty where a formulation does not find the subgroup.
- `subgroup_overlap(setting, corpus, table, top)` — do the formulations find the same
  subgroups? The Jaccard overlap of their `top` first subgroups of C (the most
  frequent) or P (the highest PEG).

**Recovery (with ground truth).**

- `per_run()` / `overview()` also have the recovery means, with the benchmark's
  statistics (median, std, quartiles, min, max).
- `recovery_by_k()` — the same by the true number of active dimensions, with the
  number of `texts`. Low precision means extra dimensions were added, low recall
  that true ones were missed.
- `ari_by_k()` — the ARI between the formulations by k.
- `check_against_benchmark(table)` — with its own formulation, a setting taken from
  `table` must reproduce the numbers the benchmark saved for it.

These raise `RuntimeError` when no corpus has ground truth.

**Saving.** `save(out_dir)` writes the tables above as CSV (the overview, per run,
per text, retention curve, and per setting the dimension usage, depth distribution
and the similarity matrices), the F, C and P tables of every run in `fcp/`, and the
recovery tables when there is ground truth.

Also in the module, and exported from `polartox`: `adjusted_rand_index`,
`normalized_mutual_information` (two partitions of the same items; the same
definitions as scikit-learn's, without depending on it), `pairwise_ari` (the ARI
between the trees of different pipelines) and `tree_statistics`.

`PEGComparison.from_benchmark(benchmark)` takes the corpora (one per `text_groups`
group, or one called `"all"`), `dims` and `scale` of a `PolarizedTreesBenchmark`.

## Notebooks

- [`peg_comparison_demo.ipynb`](peg_comparison_demo.ipynb) — the whole chain on a
  small example: generate two corpora, search settings, compare the formulations.
  Runs in about a minute.
- [`../benchmarks/notebooks/pegcomparison.ipynb`](../benchmarks/notebooks/pegcomparison.ipynb)
  — the experiment of the paper, on the saved benchmark corpora and the top
  configurations of the search.
- [`../Dices/DICES_polarized_trees_end_to_end.ipynb`](../Dices/DICES_polarized_trees_end_to_end.ipynb)
  — real data without ground truth: the formulations compared on DICES-350 and
  DICES-990, and F, C and P for every formulation.

The demo starts from `pip install polartox`; from a clone, use `pip install -e .`
to run it against this repository.
