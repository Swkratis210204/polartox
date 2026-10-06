"""
polartox.peg_comparison -- compare the PEG formulations on the same corpora.

Everything is held fixed except the PEG formulation (`variant`): for every
chosen setting (typically a row of PolarizedTreesBenchmark.top_configs_) and
every corpus, one PolarizedTreesPipeline per formulation is run on the
annotations of a corpus with known ground truth (polartox.datagen). From
those runs PEGComparison derives:

    overview()          recovery and tree shape per formulation
    recovery_by_k()     the same split by the true number of active dimensions
    ari_matrix()        do two formulations split the annotators the same way?
    ari_by_k()          ... split by the true number of active dimensions
    disagreement()      the texts on which the formulations differ most
    show_text()         the trees of one text, side by side

The adjusted Rand index (adjusted_rand_index) and its comparison of
pipelines (pairwise_ari) live here too.

Flow: datagen -> polarized_tree -> pipeline -> benchmark -> peg_comparison.

Ground truth is required, as in PolarizedTreesBenchmark: a comparison of
formulations is a comparison of how well they recover the true dimensions.
"""

import inspect
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from polartox.benchmark import STAT_METRICS, STATS, as_text_id, normalize_ground_truth
from polartox.pipeline import PolarizedTreesPipeline
from polartox.polarized_tree import PEG_VARIANTS, check_variant

__all__ = ["PEGComparison", "adjusted_rand_index", "pairwise_ari"]

RECOVERY = ["jaccard", "precision", "recall", "exact_match"]

# Diagnostics of a run (key of PolarizedTreesPipeline.diagnostics) -> display name.
TREE = {
    "retention_rate": "Retention",
    "mean_leaves": "Leaves",
    "mean_depth": "Depth",
    "mean_leaf_size": "Annotators per leaf",
    "mean_residual_ndfu": "Residual nDFU",
    "mean_top_split_peg": "Top-split PEG",
    "indeterminate_rate": "Indeterminate",
}

# Pipeline arguments that PEGComparison sets itself.
_FIXED = {"self", "dims", "scale", "variant"}


# ---------------------------------------------------------------------
# Adjusted Rand index
# ---------------------------------------------------------------------

def adjusted_rand_index(labels_a, labels_b):
    """Adjusted Rand index between two partitions of the same items.

    `labels_a[i]` and `labels_b[i]` are the groups item i belongs to in the two
    partitions (any hashable labels; only which items share a label matters).
    1 means identical partitions, about 0 means no more agreement than chance,
    and negative values mean less than chance. Two partitions that each put all
    items in a single group count as identical (1.0).

    Same definition as sklearn.metrics.adjusted_rand_score, which polartox does
    not depend on.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("labels_a and labels_b must be one-dimensional and of equal length")

    n = len(a)
    if n < 2:
        return 1.0

    _, codes_a = np.unique(a, return_inverse=True)
    _, codes_b = np.unique(b, return_inverse=True)
    table = np.zeros((codes_a.max() + 1, codes_b.max() + 1), dtype=np.int64)
    np.add.at(table, (codes_a, codes_b), 1)

    sum_squares = int((table ** 2).sum())
    rows = int((table.sum(axis=1) ** 2).sum())
    columns = int((table.sum(axis=0) ** 2).sum())

    # Ordered pairs of items: together in both / only in b / only in a / in neither.
    both = sum_squares - n
    only_b = columns - sum_squares
    only_a = rows - sum_squares
    neither = n * n - only_a - only_b - sum_squares

    if only_a == 0 and only_b == 0:
        return 1.0
    return 2.0 * (both * neither - only_a * only_b) / (
        (both + only_a) * (only_a + neither) + (both + only_b) * (only_b + neither)
    )


def pairwise_ari(pipelines, dataset):
    """How much do different pipelines agree on the groups of annotators?

    `pipelines` maps a name to a PolarizedTreesPipeline that has built its trees
    on `dataset` (run_full_evaluation or build_all_trees); typically the same
    settings with a different PEG formulation. For every text that all of them
    analysed, the trees' leaves partition the text's annotators, and the
    adjusted Rand index (1 = same groups, about 0 = chance) compares two
    partitions.

    Returns (matrix, per_text): the mean ARI of every pair over those texts as a
    names x names DataFrame (1.0 on the diagonal), and the per-text values as a
    long DataFrame with columns text_id, a, b, ari.
    """
    names = list(pipelines)
    shared = sorted(set.intersection(*(set(p.trees_) for p in pipelines.values())))

    wanted = set(shared)
    texts = {text_id: rows for text_id, rows in dataset.groupby("text_id", sort=False) if text_id in wanted}
    labels = {
        name: {text_id: pipelines[name].trees_[text_id].leaf_labels(texts[text_id]) for text_id in shared}
        for name in names
    }
    per_text = pd.DataFrame(
        [
            {"text_id": text_id, "a": a, "b": b,
             "ari": adjusted_rand_index(labels[a][text_id], labels[b][text_id])}
            for text_id in shared
            for a, b in itertools.combinations(names, 2)
        ],
        columns=["text_id", "a", "b", "ari"],
    )

    matrix = pd.DataFrame(1.0, index=names, columns=names)
    for (a, b), group in per_text.groupby(["a", "b"]):
        matrix.loc[a, b] = matrix.loc[b, a] = group["ari"].mean()
    return matrix, per_text


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def _as_corpora(corpora, dims):
    """{name: (data, ground_truth)}, checked. Accepts that dict, one
    (data, ground_truth) pair, or what AnnotatorPool.generate_dataset returns."""
    if not isinstance(corpora, dict):
        corpora = {"corpus": corpora}
    if not corpora:
        raise ValueError("corpora is empty.")

    checked = {}
    for name, corpus in corpora.items():
        if isinstance(corpus, pd.DataFrame):
            raise TypeError(
                f"corpus '{name}' has no ground truth: pass (data, ground_truth) or the "
                "result of AnnotatorPool.generate_dataset. Formulations are compared on how "
                "well they recover the true dimensions."
            )
        try:
            data, truth = corpus
        except (TypeError, ValueError):
            raise TypeError(f"corpus '{name}' must be (data, ground_truth).") from None

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"corpus '{name}': the annotations must be a pandas DataFrame.")
        missing = ({"text_id", "rating"} | set(dims)) - set(data.columns)
        if missing:
            raise ValueError(f"corpus '{name}': annotations are missing columns {sorted(missing)}.")
        if data.empty:
            raise ValueError(f"corpus '{name}': annotations cannot be empty.")
        if not isinstance(truth, dict):
            raise TypeError(f"corpus '{name}': ground_truth must be a dictionary.")

        truth = normalize_ground_truth(data, truth)
        if any("active_dims" not in entry for entry in truth.values()):
            raise ValueError(f"corpus '{name}': every ground_truth entry needs 'active_dims'.")
        checked[name] = (data, truth)
    return checked


def _in_order(table, levels, **orders):
    """`table` with its index levels in the given orders (the others sorted), not
    in order of appearance."""
    keys = [
        tuple(orders[name].index(value) if name in orders else value for name, value in zip(levels, index))
        for index in table.index
    ]
    return table.iloc[sorted(range(len(table)), key=keys.__getitem__)]


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def _as_settings(settings):
    """{label: pipeline arguments}. A dict is one setting, a list of dicts several,
    and a DataFrame (such as PolarizedTreesBenchmark.top_configs_) one per row,
    labelled by its index; columns that are not pipeline arguments (metrics, rank)
    and empty cells are skipped. A `variant` is always ignored."""
    allowed = set(inspect.signature(PolarizedTreesPipeline.__init__).parameters) - _FIXED

    if isinstance(settings, pd.DataFrame):
        labelled = {label: row.to_dict() for label, row in settings.iterrows()}
        strict = False
    elif isinstance(settings, dict):
        labelled, strict = {0: settings}, True
    elif isinstance(settings, (list, tuple)):
        labelled, strict = dict(enumerate(settings)), True
    else:
        raise TypeError("settings must be a dict, a list of dicts or a DataFrame.")
    if not labelled:
        raise ValueError("settings is empty.")

    parsed = {}
    for label, setting in labelled.items():
        if strict:
            unknown = set(setting) - allowed - {"variant"}
            if unknown:
                raise ValueError(f"setting {label}: not pipeline arguments: {sorted(unknown)}.")
        parsed[label] = {
            key: _plain(value) for key, value in setting.items()
            if key in allowed and not pd.isna(value)
        }
    return parsed


# ---------------------------------------------------------------------
# PEGComparison
# ---------------------------------------------------------------------

class PEGComparison:
    """
    The PEG formulations on the same corpora, everything else fixed.

    corpora : {name: (data, ground_truth)}
        Annotations (columns text_id, rating and `dims`) and the ground truth
        of each corpus, as AnnotatorPool.generate_dataset returns them. One
        corpus may be passed without the dict.
    dims, scale :
        As in PolarizedTreesPipeline.
    variants :
        The formulations to compare (default: all of PEG_VARIANTS).

        comparison = PEGComparison(corpora, dims=DIMS, scale=5)
        comparison.run(benchmark.top_configs_.head(3))
        comparison.overview()

    Every table is keyed by `setting`: the index label of the row of the
    settings DataFrame, the position in a list, or 0 for a single dict.
    """

    def __init__(self, corpora, dims, scale, variants=PEG_VARIANTS):
        self.dims = list(dims)
        self.scale = scale
        self.variants = list(variants)
        for variant in self.variants:
            check_variant(variant)
        if len(self.variants) < 2 or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be at least two different formulations.")

        self.corpora = _as_corpora(corpora, self.dims)

        self.settings_ = {}     # setting -> pipeline arguments
        self.runs_ = {}         # (setting, variant, corpus) -> (pipeline, results)
        self._ari = {}

    @classmethod
    def from_benchmark(cls, benchmark, variants=PEG_VARIANTS):
        """The corpora, dims and scale of a PolarizedTreesBenchmark. With `text_groups`
        every group becomes a corpus; without, there is one corpus, "all"."""
        data = benchmark.annotations
        truth = benchmark._normalized_ground_truth()
        if benchmark.text_groups is None:
            corpora = {"all": (data, truth)}
        else:
            group_of = data["text_id"].map({as_text_id(t): g for t, g in benchmark.text_groups.items()})
            corpora = {
                group: (data[group_of == group], {t: truth[t] for t in data.loc[group_of == group, "text_id"].unique()})
                for group in group_of.dropna().unique()
            }
        return cls(corpora, dims=benchmark.pipeline.dims, scale=benchmark.pipeline.scale, variants=variants)

    # -- running ------------------------------------------------------

    def run(self, settings, verbose=False):
        """Run every formulation on every corpus for every setting.

        `settings` holds the non-PEG arguments of PolarizedTreesPipeline
        (theta_filter, min_size_frac, max_depth, h, relative_h, theta_stop, ...):
        a dict, a list of dicts, or a DataFrame such as top_configs_ (a row's own
        `variant` is ignored: all formulations run with its other settings).
        Returns self.
        """
        self.settings_ = _as_settings(settings)

        # Every pipeline is built before any runs, so a bad setting fails at once.
        pipelines = {
            (setting, variant, corpus): PolarizedTreesPipeline(
                dims=self.dims, scale=self.scale, variant=variant, **params
            )
            for setting, params in self.settings_.items()
            for variant in self.variants
            for corpus in self.corpora
        }

        self.runs_, self._ari = {}, {}
        for key, pipe in pipelines.items():
            data, truth = self.corpora[key[2]]
            results = pipe.run_full_evaluation(data, ground_truth=truth, verbose=False)
            self.runs_[key] = (pipe, results)
            if verbose:
                print("done:", key)
        return self

    def _require_run(self):
        if not self.runs_:
            raise RuntimeError("Call run(settings) first.")

    def _first_setting(self, setting):
        self._require_run()
        if setting is None:
            return next(iter(self.settings_))
        if setting not in self.settings_:
            raise KeyError(f"unknown setting {setting!r}; available: {list(self.settings_)}")
        return setting

    # -- recovery and tree shape -------------------------------------

    def per_run(self):
        """One row per (setting, variant, corpus): the tree diagnostics, and the
        recovery means (with the benchmark's statistics: median, std, quartiles,
        min and max for jaccard, precision and recall)."""
        self._require_run()
        rows = []
        for (setting, variant, corpus), (pipe, results) in self.runs_.items():
            leaf_size_per_tree = [
                np.mean([leaf["n"] for leaf in tree.get_leaves()]) for tree in pipe.trees_.values()
            ]
            row = {
                "setting": setting, "variant": variant, "corpus": corpus,
                **{key: results["diagnostics"][key] for key in TREE if key != "mean_leaf_size"},
                "mean_leaf_size": float(np.mean(leaf_size_per_tree)),
            }
            for metric in RECOVERY:
                values = results["recovery"][metric].astype(float)
                row[metric] = values.mean()
                if metric in STAT_METRICS:
                    row.update({f"{metric}_{stat}": function(values) for stat, function in STATS.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def overview(self):
        """Per (setting, variant), every number of per_run() averaged over the
        corpora with equal weight."""
        per_run = self.per_run()
        numbers = [c for c in per_run.columns if c not in ("setting", "variant", "corpus")]
        return per_run.groupby(["setting", "variant"], sort=False)[numbers].mean()

    def recovery_by_k(self):
        """Mean recovery per (setting, variant, true number of active dimensions),
        pooled over the texts of all corpora, with the number of `texts`. Precision
        low means extra dimensions were added, recall low that true ones were missed."""
        self._require_run()
        frames = [
            results["recovery"].assign(setting=setting, variant=variant, corpus=corpus)
            for (setting, variant, corpus), (_, results) in self.runs_.items()
        ]
        pooled = pd.concat(frames, ignore_index=True)
        pooled[RECOVERY] = pooled[RECOVERY].astype(float)
        grouped = pooled.groupby(["setting", "variant", "k_true"], sort=False)
        table = grouped[RECOVERY].mean()
        table.insert(0, "texts", grouped.size())
        return _in_order(table, ["setting", "variant", "k_true"], setting=list(self.settings_), variant=self.variants)

    def check_against_benchmark(self, table, rtol=1e-6):
        """Check that each setting taken from `table` (e.g. top_configs_, indexed as
        the settings were) reproduces, with its own `variant`, the numbers the
        benchmark saved for it. Raises ValueError listing the differences; returns
        the number of numbers compared."""
        overview = self.overview()
        compared, wrong = 0, []
        for setting in self.settings_:
            if setting not in table.index:
                continue
            row, own = table.loc[setting], table.loc[setting, "variant"]
            for column in overview.columns:
                if column in table.columns and pd.notna(row[column]):
                    compared += 1
                    if not np.isclose(overview.loc[(setting, own), column], row[column], rtol=rtol):
                        wrong.append((setting, own, column))
        if wrong:
            raise ValueError(f"differs from the benchmark in (setting, variant, column): {wrong}")
        return compared

    # -- do the formulations build the same trees? --------------------

    def _ari_of(self, setting, corpus):
        key = (setting, corpus)
        if key not in self._ari:
            data, _ = self.corpora[corpus]
            pipes = {variant: self.runs_[setting, variant, corpus][0] for variant in self.variants}
            self._ari[key] = pairwise_ari(pipes, data)
        return self._ari[key]

    def ari(self, setting=None, corpus=None):
        """(matrix, per_text) of pairwise_ari for one setting and corpus."""
        setting = self._first_setting(setting)
        if corpus is None:
            if len(self.corpora) != 1:
                raise ValueError(f"name a corpus: {list(self.corpora)}")
            corpus = next(iter(self.corpora))
        return self._ari_of(setting, corpus)

    def ari_matrix(self, setting=None):
        """Mean ARI of every pair of formulations, averaged over the corpora."""
        setting = self._first_setting(setting)
        matrices = [self._ari_of(setting, corpus)[0] for corpus in self.corpora]
        return sum(matrices) / len(matrices)

    def _ari_per_text(self):
        frames = []
        for setting in self.settings_:
            for corpus, (_, truth) in self.corpora.items():
                per_text = self._ari_of(setting, corpus)[1]
                k_of = {text_id: len(entry["active_dims"]) for text_id, entry in truth.items()}
                frames.append(per_text.assign(setting=setting, corpus=corpus,
                                              k_true=per_text["text_id"].map(k_of)))
        return pd.concat(frames, ignore_index=True)

    def ari_by_k(self):
        """Mean ARI per (setting, pair of formulations, true number of active
        dimensions), pooled over the corpora, plus the row "all pairs"."""
        self._require_run()
        per_text = self._ari_per_text()
        per_text["pair"] = per_text["a"] + " - " + per_text["b"]
        pairs = per_text.groupby(["setting", "pair", "k_true"], sort=False)["ari"].mean()
        every = per_text.groupby(["setting", "k_true"], sort=False)["ari"].mean().reset_index()
        every.insert(1, "pair", "all pairs")
        table = pd.concat([pairs.reset_index(), every], ignore_index=True).set_index(["setting", "pair", "k_true"])["ari"]
        pair_order = [f"{a} - {b}" for a, b in itertools.combinations(self.variants, 2)] + ["all pairs"]
        return _in_order(table, ["setting", "pair", "k_true"], setting=list(self.settings_), pair=pair_order)

    def disagreement(self, setting=None):
        """Texts ordered by how differently the formulations split them: the mean
        ARI over all pairs (lowest first), with the corpus and the true k."""
        setting = self._first_setting(setting)
        per_text = self._ari_per_text()
        per_text = per_text[per_text["setting"] == setting]
        return (
            per_text.groupby(["corpus", "text_id", "k_true"], sort=False)["ari"].mean()
            .rename("mean_ari").reset_index().sort_values("mean_ari", kind="stable", ignore_index=True)
        )

    # -- one text -----------------------------------------------------

    def text_summary(self, corpus, text_id, setting=None):
        """One line per formulation for one text: first split, dimensions used,
        leaves, depth and the Jaccard with the true dimensions."""
        setting = self._first_setting(setting)
        true_dims = self.corpora[corpus][1][text_id]["active_dims"]
        rows = []
        for variant in self.variants:
            tree = self.runs_[setting, variant, corpus][0].trees_.get(text_id)
            if tree is None:
                continue
            used = sorted({dim for leaf in tree.get_leaves() for dim, _ in leaf["path"]})
            rows.append({
                "formulation": variant,
                "first split": tree.get_root().get("split_dim", "none"),
                "dimensions used": ", ".join(used) or "none",
                "leaves": tree.n_leaves,
                "depth": tree.depth,
                "Jaccard": len(set(used) & set(true_dims)) / len(set(used) | set(true_dims)),
            })
        return pd.DataFrame(rows).set_index("formulation")

    def show_text(self, corpus, text_id, setting=None):
        """Print the true structure of one text, the summary of every formulation
        and the trees themselves."""
        setting = self._first_setting(setting)
        entry = self.corpora[corpus][1][text_id]
        alpha = entry.get("alpha", {})
        print("#" * 10, f"{corpus}, text {text_id} | true active dimensions:",
              ", ".join(f"{d} (alpha {alpha[d]:.2f})" if d in alpha else d for d in entry["active_dims"])
              or "none")
        print(self.text_summary(corpus, text_id, setting).round(2).to_string())
        for variant in self.variants:
            tree = self.runs_[setting, variant, corpus][0].trees_.get(text_id)
            if tree is not None:
                print(f"--- {variant} ---")
                tree.render()

    # -- saving -------------------------------------------------------

    def save(self, out_dir):
        """Write the tables to `out_dir`: selected_settings, per_run, overview,
        recovery_by_k, ari_by_k, and per setting and corpus the ARI matrix and
        the per-text values. Returns the folder."""
        self._require_run()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame.from_dict(self.settings_, orient="index").to_csv(out_dir / "selected_settings.csv")
        self.per_run().to_csv(out_dir / "per_run.csv", index=False)
        self.overview().to_csv(out_dir / "overview.csv")
        self.recovery_by_k().to_csv(out_dir / "recovery_by_k.csv")
        self.ari_by_k().to_csv(out_dir / "ari_by_k.csv")

        for setting in self.settings_:
            for corpus in self.corpora:
                matrix, per_text = self._ari_of(setting, corpus)
                matrix.to_csv(out_dir / f"ari_setting{setting}_{corpus}.csv")
                per_text.to_csv(out_dir / f"ari_per_text_setting{setting}_{corpus}.csv", index=False)
        return out_dir
