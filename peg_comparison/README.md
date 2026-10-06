# PEG comparison — `polartox.peg_comparison`

`polartox.peg_comparison` compares the **PEG formulations** (`max`, `avg`,
`min`, `mean`, `harmonic`) on the same corpora, with **everything else held
fixed**. For each chosen setting and each corpus it runs one
`PolarizedTreesPipeline` per formulation and answers two questions:

- **Recovery** — which formulation finds the true dimensions of the texts
  (Jaccard, precision, recall, exact match), and what do its trees look like
  (leaves, depth, annotators per leaf, residual nDFU)?
- **Agreement** — do two formulations split the annotators of a text into the
  same groups? This is measured with the adjusted Rand index (ARI), which
  also lives in this module (`adjusted_rand_index`, `pairwise_ari`).

It is the last step of the package's chain:

```text
datagen  →  polarized_tree  →  pipeline  →  benchmark  →  peg_comparison
corpora      one text's tree    a corpus      search the      the best settings,
with truth                      of trees      settings        one formulation at a time
```

**Ground truth is required**, as for `PolarizedTreesBenchmark`: a comparison of
formulations is a comparison of how well they recover the true dimensions.
For a corpus without ground truth (real data such as DICES) use
`PolarizedTreesPipeline.run_full_evaluation(data)` directly; it gives the trees
and F/C/P without recovery metrics, and `pairwise_ari` still compares trees.

## Quickstart

```python
from polartox import (
    AnnotatorPool, DEFAULT_DEPTH_WEIGHTS, DEFAULT_DIMENSIONS, DEFAULT_INTENSITY_RANGE,
    PEGComparison,
)

pool = AnnotatorPool(DEFAULT_DIMENSIONS, scale=5, intensity_range=DEFAULT_INTENSITY_RANGE,
                     depth_weights=DEFAULT_DEPTH_WEIGHTS, annotators_per_identity=3)
corpus = pool.generate_dataset(n_texts=30, seed=0)          # annotations + ground truth

comparison = PEGComparison({"default": corpus}, dims=list(DEFAULT_DIMENSIONS), scale=5)
comparison.run(dict(theta_filter=0.3, min_size_frac=0.03, max_depth=6,
                    h=0.15, relative_h=True, theta_stop=0.10))

comparison.overview()          # recovery and tree shape per formulation
comparison.ari_matrix()        # mean ARI between every pair of formulations
comparison.show_text("default", 0)
```

With a benchmark, the settings come from its best configurations:

```python
comparison = PEGComparison.from_benchmark(benchmark)        # corpora, dims, scale
comparison.run(benchmark.top_configs_.head(3))              # each row's own variant is ignored
comparison.check_against_benchmark(benchmark.top_configs_)  # reproduces the benchmark's numbers
```

## Inputs

- **`corpora`** — `{name: (data, ground_truth)}`: the annotations (a DataFrame
  with `text_id`, `rating` and one column per dimension) and the ground truth
  (`{text_id: {"active_dims": [...], ...}}`), exactly what
  `AnnotatorPool.generate_dataset` returns. A single corpus, or a
  `GeneratedDataset`, can be passed without the dict. The ground truth must
  cover every annotated text; text ids read back from JSON (strings) are matched
  to integer ids, as in the benchmark. Keep corpora separate: text ids only
  have to be unique within a corpus.
- **`settings`** (for `run`) — the non-PEG arguments of `PolarizedTreesPipeline`
  (`theta_filter`, `min_size_frac`, `max_depth`, `h`, `relative_h`, `theta_stop`, ...):
  a dict (one setting), a list of dicts, or a DataFrame such as
  `top_configs_` (one setting per row, labelled by its index; columns that are
  not pipeline arguments, such as the metrics, and empty cells are skipped).
  A `variant` in a setting is ignored: all formulations run with the rest.
  A misspelt argument in a dict raises `ValueError`, and every pipeline is built
  before the first one runs, so a bad setting fails at once.

Every table is keyed by **`setting`**: the index label of the settings row, the
position in a list, or `0` for a single dict.

## API

- `PEGComparison(corpora, dims, scale, variants=PEG_VARIANTS)` — at least two
  different formulations.
- `PEGComparison.from_benchmark(benchmark, variants=PEG_VARIANTS)` — the corpora
  (one per `text_groups` group, or one called `"all"`), `dims` and `scale` of a
  `PolarizedTreesBenchmark`.
- `run(settings, verbose=False)` — runs every formulation on every corpus for every
  setting; returns the comparison. Results are in `runs_`
  (`(setting, variant, corpus) → (pipeline, results)`) and the arguments used
  in `settings_`.
- `per_run()` — one row per (setting, variant, corpus): tree diagnostics, and
  recovery means with the benchmark's statistics (median, std, quartiles, min, max).
- `overview()` — `per_run()` averaged over the corpora with equal weight.
- `recovery_by_k()` — mean recovery per true number of active dimensions, pooled
  over the texts of all corpora, with the number of `texts`. Low precision means
  extra dimensions were added, low recall that true ones were missed.
- `check_against_benchmark(table, rtol=1e-6)` — with its own formulation, a setting
  taken from `table` must reproduce the numbers the benchmark saved for it;
  raises `ValueError` otherwise, and returns the number of numbers compared.
- `ari(setting, corpus)` — `(matrix, per_text)` of `pairwise_ari`.
- `ari_matrix(setting)` — the mean ARI of every pair, averaged over the corpora.
- `ari_by_k()` — mean ARI per pair and true number of active dimensions, plus
  an `"all pairs"` row.
- `disagreement(setting)` — the texts ordered by how differently the
  formulations split them (lowest mean ARI first).
- `text_summary(corpus, text_id, setting)` / `show_text(...)` — one text: its true
  structure, one line per formulation (first split, dimensions used, leaves,
  depth, Jaccard) and the trees.
- `save(out_dir)` — writes `selected_settings.csv`, `per_run.csv`, `overview.csv`,
  `recovery_by_k.csv`, `ari_by_k.csv`, and per setting and corpus the ARI
  matrix and per-text values.

Also in the module:

- `adjusted_rand_index(labels_a, labels_b)` — the adjusted Rand index of two
  partitions (1 = the same groups, about 0 = chance), the same definition as
  scikit-learn's, without depending on it.
- `pairwise_ari(pipelines, dataset)` — the mean ARI between the trees of
  different pipelines over the texts they all analysed; the leaves of a tree
  (`PolarizedTree.leaf_labels(dataset)`) partition a text's annotators.

All three are also exported from `polartox`.

## Notebooks

- [`peg_comparison_demo.ipynb`](peg_comparison_demo.ipynb) — the whole chain on a
  small example: generate two corpora, search settings, compare the formulations.
  Runs in about a minute.
- [`../benchmarks/notebooks/pegcomparison.ipynb`](../benchmarks/notebooks/pegcomparison.ipynb)
  — the experiment of the paper, on the saved benchmark corpora and the top
  configurations of the search.

The demo starts from `pip install polartox`; from a clone, use `pip install -e .`
to run it against this repository.
