"""
polartox.peg_comparison -- compare the PEG formulations on the same corpora.

Everything is held fixed except the PEG formulation (`variant`): for every
chosen setting (typically a row of PolarizedTreesBenchmark.top_configs_) and
every corpus, one PolarizedTreesPipeline per formulation is run on the
annotations of the corpus. Ground truth is optional, and it decides what
PEGComparison can tell you.

Always (no ground truth needed -- real data included):

    overview()              the trees per formulation: retention, leaves, depth, leaf size,
                            residual nDFU, split PEG, indeterminate leaves, trees that
                            never split, trees that split on 2+ dimensions, ...
    per_text()              the same for every text
    retention_curve()       the share of texts kept for every theta_filter
    depth_distribution()    the trees by depth (optionally cumulative)
    dimension_usage()       how often every dimension splits, and splits first
    similarity()            how close are the trees of any two runs (every setting and
                            formulation): the ARI or the NMI of the groups they make of
                            the annotators, the dimensions they split on, their first split
    ari_matrix()            ... the ARI between the formulations of one setting
    dims_agreement()        ... the same dimensions?
    first_split_agreement() ... the same first split?
    disagreement()          the texts on which the formulations differ most
    fcp()                   F, C and P of every formulation
    fcp_comparison()        ... side by side
    subgroup_overlap()      do they find the same subgroups?
    show_text()             the trees of one text, side by side

With ground truth (polartox.datagen), in addition:

    overview() / per_run()  recovery (Jaccard, precision, recall, exact match)
    recovery_by_k()         the same split by the true number of active dimensions
    ari_by_k()              the ARI between formulations by k

The adjusted Rand index (adjusted_rand_index), the normalized mutual information
(normalized_mutual_information) and the comparison of pipelines (pairwise_ari)
live here too.

Flow: datagen -> polarized_tree -> pipeline -> benchmark -> peg_comparison.
"""

import inspect
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from polartox.benchmark import STAT_METRICS, STATS, as_text_id, normalize_ground_truth
from polartox.pipeline import PolarizedTreesPipeline
from polartox.polarized_tree import PEG_VARIANTS, check_variant, jaccard

__all__ = ["PEGComparison", "adjusted_rand_index", "normalized_mutual_information", "pairwise_ari", "tree_statistics"]

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

# What tree_statistics() adds to those, with display names.
TREE_MORE = {
    "mean_tree_depth": "Tree depth",
    "weighted_residual_ndfu": "Residual nDFU (annotator-weighted)",
    "explained_share": "Polarization explained",
    "mean_split_peg": "Split PEG",
    "no_split_rate": "Trees that never split",
    "multi_dim_share": "Trees splitting on 2+ dimensions",
    "mean_dims_used": "Dimensions used per tree",
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


def normalized_mutual_information(labels_a, labels_b):
    """Normalized mutual information between two partitions of the same items: the
    information the groups of one partition give about the groups of the other,
    from 0 (none) to 1 (the same groups), normalized by the mean of the two
    entropies (the default of sklearn.metrics.normalized_mutual_info_score, which
    polartox does not depend on). Unlike the adjusted Rand index it is not
    corrected for chance, so partitions with many small groups look closer than
    they are. Two partitions that each put all items in one group count as
    identical (1.0); one that does, against one that does not, as 0."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("labels_a and labels_b must be one-dimensional and of equal length")

    n = len(a)
    if n < 2:
        return 1.0

    _, codes_a = np.unique(a, return_inverse=True)
    _, codes_b = np.unique(b, return_inverse=True)
    table = np.zeros((codes_a.max() + 1, codes_b.max() + 1), dtype=np.float64)
    np.add.at(table, (codes_a, codes_b), 1)
    joint = table / n
    p_a, p_b = joint.sum(axis=1), joint.sum(axis=0)

    entropy_a = -float((p_a * np.log(p_a)).sum())
    entropy_b = -float((p_b * np.log(p_b)).sum())
    if entropy_a == 0.0 and entropy_b == 0.0:
        return 1.0
    if entropy_a == 0.0 or entropy_b == 0.0:
        return 0.0

    nonzero = joint > 0
    expected = np.outer(p_a, p_b)
    mutual = float((joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])).sum())
    return max(0.0, mutual / ((entropy_a + entropy_b) / 2.0))


# ---------------------------------------------------------------------
# What a tree looks like (no ground truth)
# ---------------------------------------------------------------------

def _text_statistics(tree):
    """The numbers of one tree."""
    leaves = tree.get_leaves()
    root = tree.get_root()
    sizes = np.array([leaf["n"] for leaf in leaves], dtype=float)
    ndfus = np.array([leaf["ndfu"] for leaf in leaves], dtype=float)
    splits = list(tree.internal_nodes())
    used = sorted({dim for _, dim, _, _, _ in splits})

    weighted = float((sizes * ndfus).sum() / sizes.sum()) if sizes.sum() else np.nan
    root_ndfu = root["ndfu"]
    return {
        "n_leaves": tree.n_leaves,
        "tree_depth": tree.depth,
        "mean_leaf_depth": float(np.mean([len(leaf["path"]) for leaf in leaves])),
        "mean_leaf_size": float(sizes.mean()),
        "root_ndfu": root_ndfu,
        "residual_ndfu": float(ndfus.mean()),
        "weighted_residual_ndfu": weighted,
        "explained_share": 1.0 - weighted / root_ndfu if root_ndfu and root_ndfu > 0 else np.nan,
        "n_splits": len(splits),
        "mean_split_peg": float(np.mean([peg for _, _, peg, _, _ in splits])) if splits else np.nan,
        "top_split_peg": root["peg"] if not root["is_leaf"] else np.nan,
        "first_split": root["split_dim"] if not root["is_leaf"] else None,
        "dims_used": used,
        "n_dims_used": len(used),
        "n_indeterminate": sum(leaf["pole"] == "indeterminate" for leaf in leaves),
    }


def tree_statistics(pipeline):
    """The structure of the trees a pipeline has built (run_full_evaluation or
    build_all_trees), as a dict; no ground truth needed.

    retention_rate, mean_leaves, mean_depth (of the leaves), mean_leaf_size,
    mean_residual_ndfu, mean_top_split_peg and indeterminate_rate are the
    diagnostics of the pipeline. In addition: n_texts and n_retained;
    mean_tree_depth (the deepest leaf of a tree); weighted_residual_ndfu (the
    nDFU left in the leaves, weighted by their annotators) and explained_share
    (the share of a text's nDFU the splits removed); mean_split_peg (all splits);
    no_split_rate (trees that never split); multi_dim_share (trees that split on
    two or more dimensions) and mean_dims_used; and dims_never_used.
    """
    stats = [_text_statistics(tree) for tree in pipeline.trees_.values()]
    n_texts = len(pipeline.overall_ndfu_) or len(pipeline.trees_)
    leaves = [leaf for tree in pipeline.trees_.values() for leaf in tree.get_leaves()]
    splits = [peg for tree in pipeline.trees_.values() for _, _, peg, _, _ in tree.internal_nodes()]
    used = {dim for row in stats for dim in row["dims_used"]}

    def mean(values):
        values = [v for v in values if v == v]
        return float(np.mean(values)) if values else np.nan

    return {
        "n_texts": n_texts,
        "n_retained": len(pipeline.retained_ids_),
        "retention_rate": len(pipeline.retained_ids_) / n_texts if n_texts else np.nan,
        "mean_leaves": mean([row["n_leaves"] for row in stats]),
        "mean_depth": mean([len(leaf["path"]) for leaf in leaves]),
        "mean_tree_depth": mean([row["tree_depth"] for row in stats]),
        "mean_leaf_size": mean([row["mean_leaf_size"] for row in stats]),
        "mean_residual_ndfu": mean([leaf["ndfu"] for leaf in leaves]),
        "weighted_residual_ndfu": mean([row["weighted_residual_ndfu"] for row in stats]),
        "explained_share": mean([row["explained_share"] for row in stats]),
        "mean_split_peg": mean(splits),
        "mean_top_split_peg": mean([row["top_split_peg"] for row in stats]),
        "indeterminate_rate": mean([leaf["pole"] == "indeterminate" for leaf in leaves]),
        "no_split_rate": mean([row["n_splits"] == 0 for row in stats]),
        "multi_dim_share": mean([row["n_dims_used"] >= 2 for row in stats]),
        "mean_dims_used": mean([row["n_dims_used"] for row in stats]),
        "dims_never_used": sorted(set(pipeline.dims) - used),
    }


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def _as_corpora(corpora, dims):
    """{name: (data, ground_truth or None)}, checked. Accepts that dict, one
    corpus, and for each corpus the annotations alone, a (data, ground_truth)
    pair, or what AnnotatorPool.generate_dataset returns."""
    if not isinstance(corpora, dict):
        corpora = {"corpus": corpora}
    if not corpora:
        raise ValueError("corpora is empty.")

    checked = {}
    for name, corpus in corpora.items():
        if isinstance(corpus, pd.DataFrame):
            data, truth = corpus, None
        else:
            try:
                data, truth = corpus
            except (TypeError, ValueError):
                raise TypeError(f"corpus '{name}' must be the annotations or (data, ground_truth).") from None

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"corpus '{name}': the annotations must be a pandas DataFrame.")
        missing = ({"text_id", "rating"} | set(dims)) - set(data.columns)
        if missing:
            raise ValueError(f"corpus '{name}': annotations are missing columns {sorted(missing)}.")
        if data.empty:
            raise ValueError(f"corpus '{name}': annotations cannot be empty.")

        if truth is not None:
            if not isinstance(truth, dict):
                raise TypeError(f"corpus '{name}': ground_truth must be a dictionary or None.")
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

    corpora : {name: corpus}
        Every corpus is the annotations (columns text_id, rating and `dims`), or
        `(annotations, ground_truth)` as AnnotatorPool.generate_dataset returns
        them. Ground truth is optional and may differ between corpora: what needs
        it (recovery, the ARI with the true structure, the by-k tables) uses the
        corpora that have it, and the rest of the analysis works without. One
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
        self._labels = {}

    @property
    def truth_corpora(self):
        """The corpora that have ground truth."""
        return [name for name, (_, truth) in self.corpora.items() if truth is not None]

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
        (theta_filter, min_size_frac, max_depth, h, relative_h, theta_stop,
        theta_pole, ...): a dict, a list of dicts, or a DataFrame such as top_configs_
        (a row's own `variant` is ignored: all formulations run with its other
        settings). Returns self.
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

        self.runs_, self._ari, self._labels = {}, {}, {}
        for key, pipe in pipelines.items():
            data, truth = self.corpora[key[2]]
            try:
                results = pipe.run_full_evaluation(data, ground_truth=truth, verbose=False)
            except RuntimeError as error:
                setting, _, corpus = key
                raise ValueError(
                    f"setting {setting!r} keeps no text of corpus {corpus!r} "
                    f"(theta_filter={pipe.theta_filter}): lower theta_filter."
                ) from error
            self.runs_[key] = (pipe, results)
            if verbose:
                print("done:", key)
        return self

    def _require_run(self):
        if not self.runs_:
            raise RuntimeError("Call run(settings) first.")

    def _require_truth(self):
        self._require_run()
        if not self.truth_corpora:
            raise RuntimeError("This needs ground truth, and no corpus has it.")

    def _first_setting(self, setting):
        self._require_run()
        if setting is None:
            return next(iter(self.settings_))
        if setting not in self.settings_:
            raise KeyError(f"unknown setting {setting!r}; available: {list(self.settings_)}")
        return setting

    def _one_corpus(self, corpus):
        if corpus is None:
            if len(self.corpora) != 1:
                raise ValueError(f"name a corpus: {list(self.corpora)}")
            return next(iter(self.corpora))
        if corpus not in self.corpora:
            raise KeyError(f"unknown corpus {corpus!r}; available: {list(self.corpora)}")
        return corpus

    # -- the trees ----------------------------------------------------

    def per_run(self):
        """One row per (setting, variant, corpus): tree_statistics() of the run, and
        for corpora with ground truth the recovery means (with the benchmark's
        statistics: median, std, quartiles, min and max for jaccard, precision and
        recall)."""
        self._require_run()
        rows = []
        for (setting, variant, corpus), (pipe, results) in self.runs_.items():
            row = {"setting": setting, "variant": variant, "corpus": corpus}
            stats = tree_statistics(pipe)
            stats["dims_never_used"] = ", ".join(stats["dims_never_used"])
            row.update(stats)
            if "recovery" in results:
                for metric in RECOVERY:
                    values = results["recovery"][metric].astype(float)
                    row[metric] = values.mean()
                    if metric in STAT_METRICS:
                        row.update({f"{metric}_{stat}": function(values) for stat, function in STATS.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def overview(self, corpora=None):
        """Per (setting, variant), every number of per_run() averaged over the
        corpora (all of them, or the names in `corpora`) with equal weight; the
        numbers of texts are summed. Corpora without ground truth have no recovery
        numbers, which are then averaged over the corpora that have them."""
        frame = self.per_run()
        if corpora is not None:
            frame = frame[frame["corpus"].isin(list(corpora))]
        counts = ["n_texts", "n_retained"]
        means = [c for c in frame.select_dtypes("number").columns if c not in ["setting", *counts]]
        grouped = frame.groupby(["setting", "variant"], sort=False)
        return pd.concat([grouped[counts].sum(), grouped[means].mean()], axis=1)

    def per_text(self):
        """One row per (setting, variant, corpus, text): its annotators, its overall
        nDFU, whether it was retained, and for a retained text the numbers of its
        tree (leaves, depth, residual nDFU, first split, dimensions used, ...). With
        ground truth also k_true and the recovery of the text."""
        self._require_run()
        frames = []
        for (setting, variant, corpus), (pipe, results) in self.runs_.items():
            data, _ = self.corpora[corpus]
            size = data.groupby("text_id").size()
            rows = []
            for text_id, ndfu in pipe.overall_ndfu_.items():
                row = {"setting": setting, "variant": variant, "corpus": corpus, "text_id": text_id,
                       "n_annotators": int(size.get(text_id, 0)), "ndfu": ndfu,
                       "retained": text_id in pipe.trees_}
                if text_id in pipe.trees_:
                    stats = _text_statistics(pipe.trees_[text_id])
                    stats["dims_used"] = ", ".join(stats["dims_used"])
                    row.update(stats)
                rows.append(row)
            frame = pd.DataFrame(rows)
            if "recovery" in results and len(frame):
                recovery = results["recovery"][["text_id", "k_true", *RECOVERY]]
                frame = frame.merge(recovery, on="text_id", how="left")
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def retention_curve(self, thetas=None):
        """The share of texts whose overall nDFU reaches each theta_filter, per
        corpus (it does not depend on the formulation): the retention one gets
        for every value of theta_filter."""
        self._require_run()
        thetas = np.round(np.arange(0.05, 1.0, 0.05), 2) if thetas is None else np.asarray(thetas, dtype=float)
        curve = {}
        for corpus in self.corpora:
            pipe = next(p for (_, _, c), (p, _) in self.runs_.items() if c == corpus)
            ndfu = np.array([v for v in pipe.overall_ndfu_.values() if v == v], dtype=float)
            curve[corpus] = [float((ndfu >= theta).mean()) if len(ndfu) else np.nan for theta in thetas]
        return pd.DataFrame(curve, index=pd.Index(thetas, name="theta_filter"))

    def depth_distribution(self, setting=None, cumulative=False):
        """The share of the trees by depth (the deepest leaf), per formulation, over
        the trees of all corpora; with cumulative=True the share of trees of at most
        that depth."""
        setting = self._first_setting(setting)
        depths = {variant: [] for variant in self.variants}
        for (s, variant, _), (pipe, _) in self.runs_.items():
            if s == setting:
                depths[variant] += [tree.depth for tree in pipe.trees_.values()]
        top = max((max(d) for d in depths.values() if d), default=0)
        table = pd.DataFrame(
            {variant: [np.mean(np.array(d) == k) if d else np.nan for k in range(top + 1)]
             for variant, d in depths.items()},
            index=pd.Index(range(top + 1), name="depth"),
        )
        return table.cumsum() if cumulative else table

    def dimension_usage(self, setting=None):
        """Per formulation and dimension, over the trees of all corpora: the share of
        trees that split on the dimension at least once (`used`), the share whose
        first split it is (`first_split`) and the splits per tree. A dimension that
        no formulation ever uses has zeros."""
        setting = self._first_setting(setting)
        rows = []
        for variant in self.variants:
            n_trees, used, first, splits = 0, dict.fromkeys(self.dims, 0), dict.fromkeys(self.dims, 0), dict.fromkeys(self.dims, 0)
            for (s, v, _), (pipe, _) in self.runs_.items():
                if s != setting or v != variant:
                    continue
                for tree in pipe.trees_.values():
                    n_trees += 1
                    root = tree.get_root()
                    if not root["is_leaf"]:
                        first[root["split_dim"]] += 1
                    for _, dim, _, _, _ in tree.internal_nodes():
                        splits[dim] += 1
                    for dim in {dim for _, dim, _, _, _ in tree.internal_nodes()}:
                        used[dim] += 1
            for dim in self.dims:
                rows.append({"variant": variant, "dim": dim,
                             "used": used[dim] / n_trees if n_trees else np.nan,
                             "first_split": first[dim] / n_trees if n_trees else np.nan,
                             "splits_per_tree": splits[dim] / n_trees if n_trees else np.nan})
        return pd.DataFrame(rows).set_index(["variant", "dim"])

    # -- do the formulations build the same trees? --------------------

    def _ari_of(self, setting, corpus):
        key = (setting, corpus)
        if key not in self._ari:
            data, _ = self.corpora[corpus]
            pipes = {variant: self.runs_[setting, variant, corpus][0] for variant in self.variants}
            self._ari[key] = pairwise_ari(pipes, data)
        return self._ari[key]

    def ari(self, setting=None, corpus=None):
        """(matrix, per_text) of pairwise_ari for one setting and corpus: do the
        formulations split the annotators of a text into the same groups?"""
        setting = self._first_setting(setting)
        return self._ari_of(setting, self._one_corpus(corpus))

    def _run_labels(self, setting, variant, corpus):
        """{text_id: the group (leaf) of every annotator of the text} of a run."""
        key = (setting, variant, corpus)
        if key not in self._labels:
            pipe, _ = self.runs_[key]
            data, _ = self.corpora[corpus]
            self._labels[key] = {
                text_id: pipe.trees_[text_id].leaf_labels(rows)
                for text_id, rows in data.groupby("text_id", sort=False) if text_id in pipe.trees_
            }
        return self._labels[key]

    def similarity(self, measure="ari", setting=None):
        """How close are the trees of two runs? A matrix over the runs, the mean over
        the texts that both analysed, then averaged over the corpora (1 on the
        diagonal, a higher value is a closer pair). A run is one formulation at one
        setting: the rows and columns are the formulations of `setting`, or without
        `setting` every (setting, formulation).

        measure :
            "ari"  the adjusted Rand index of the groups the leaves make of a text's
                   annotators (1 = the same groups, about 0 = chance);
            "nmi"  their normalized mutual information (1 = the same groups, 0 =
                   independent; not corrected for chance);
            "dims" the Jaccard of the sets of dimensions the two trees split on;
            "first_split" the share of texts whose trees start with the same dimension
                   (two trees that never split agree).
        """
        self._require_run()
        if measure not in ("ari", "nmi", "dims", "first_split"):
            raise ValueError("measure must be 'ari', 'nmi', 'dims' or 'first_split'.")
        if setting is None:
            runs = [(s, v) for s in self.settings_ for v in self.variants]
            index = pd.MultiIndex.from_tuples(runs, names=["setting", "variant"])
        else:
            setting = self._first_setting(setting)
            runs = [(setting, v) for v in self.variants]
            index = pd.Index(self.variants)

        def score(corpus, run_a, run_b, text_id):
            if measure in ("ari", "nmi"):
                labels_a, labels_b = (self._run_labels(*run, corpus)[text_id] for run in (run_a, run_b))
                return (adjusted_rand_index if measure == "ari" else normalized_mutual_information)(labels_a, labels_b)
            trees = [self.runs_[run[0], run[1], corpus][0].trees_[text_id] for run in (run_a, run_b)]
            if measure == "dims":
                return jaccard(*({dim for _, dim, _, _, _ in tree.internal_nodes()} for tree in trees))
            return float(trees[0].get_root().get("split_dim") == trees[1].get_root().get("split_dim"))

        matrices = []
        for corpus in self.corpora:
            trees = {run: set(self.runs_[run[0], run[1], corpus][0].trees_) for run in runs}
            matrix = np.eye(len(runs))
            for (i, run_a), (j, run_b) in itertools.combinations(enumerate(runs), 2):
                shared = trees[run_a] & trees[run_b]
                value = float(np.mean([score(corpus, run_a, run_b, t) for t in shared])) if shared else np.nan
                matrix[i, j] = matrix[j, i] = value
            matrices.append(matrix)
        with np.errstate(all="ignore"):
            mean = np.nanmean(np.stack(matrices), axis=0)
        return pd.DataFrame(mean, index=index, columns=index)

    def ari_matrix(self, setting=None):
        """Mean ARI of every pair of formulations of a setting (the first by
        default), averaged over the corpora: see similarity()."""
        return self.similarity("ari", self._first_setting(setting))

    def dims_agreement(self, setting=None):
        """The structure of the trees: for every pair of formulations of a setting,
        the mean Jaccard of the sets of dimensions their trees of a text split on:
        see similarity()."""
        return self.similarity("dims", self._first_setting(setting))

    def first_split_agreement(self, setting=None):
        """The structure of the trees: for every pair of formulations of a setting,
        the share of texts whose trees start with the same dimension: see
        similarity()."""
        return self.similarity("first_split", self._first_setting(setting))

    def _ari_per_text(self):
        frames = []
        for setting in self.settings_:
            for corpus, (_, truth) in self.corpora.items():
                per_text = self._ari_of(setting, corpus)[1]
                k_of = {text_id: len(entry["active_dims"]) for text_id, entry in (truth or {}).items()}
                frames.append(per_text.assign(setting=setting, corpus=corpus,
                                              k_true=per_text["text_id"].map(k_of)))
        return pd.concat(frames, ignore_index=True)

    def disagreement(self, setting=None):
        """Texts ordered by how differently the formulations split them: the mean
        ARI over all pairs (lowest first), with the corpus and, where there is
        ground truth, the true k."""
        setting = self._first_setting(setting)
        per_text = self._ari_per_text()
        per_text = per_text[per_text["setting"] == setting].assign(k_true=lambda d: d["k_true"].fillna(-1))
        table = (
            per_text.groupby(["corpus", "text_id", "k_true"], sort=False)["ari"].mean()
            .rename("mean_ari").reset_index().sort_values("mean_ari", kind="stable", ignore_index=True)
        )
        table["k_true"] = table["k_true"].where(table["k_true"] >= 0)
        return table

    # -- F, C and P ---------------------------------------------------

    def fcp(self, setting=None, corpus=None, variant=None):
        """{"F": ..., "C": ..., "P": ...} of one run (see PolarizedTreesPipeline);
        without `variant`, {variant: {"F", "C", "P"}} of every formulation."""
        setting, corpus = self._first_setting(setting), self._one_corpus(corpus)
        if variant is None:
            return {v: self.fcp(setting, corpus, v) for v in self.variants}
        results = self.runs_[setting, variant, corpus][1]
        return {name: results[name] for name in "FCP"}

    def fcp_comparison(self, setting=None, corpus=None, table="F", top=10):
        """F, C or P of all the formulations side by side.

        F: the dimension frequency, one row per dimension, the columns are
        (formulation, depth). C and P: the union of the `top` first subgroups of every
        formulation (the most frequent of C, the highest PEG of P), one row per
        subgroup, ordered by the best rank any formulation gives it; the columns are
        (formulation, n_s and frac_toxic for C, n_s and mean_peg for P), empty where a
        formulation does not find the subgroup.
        """
        setting, corpus = self._first_setting(setting), self._one_corpus(corpus)
        if table not in ("F", "C", "P"):
            raise ValueError("table must be 'F', 'C' or 'P'.")
        tables = {v: self.runs_[setting, v, corpus][1][table] for v in self.variants}

        if table == "F":
            frames = {}
            for variant, frame in tables.items():
                depths = [c for c in frame.columns if isinstance(c, (int, np.integer))]
                frames[variant] = frame[depths].reindex(self.dims).fillna(0).astype(int)
            return pd.concat(frames, axis=1)

        columns = ["n_s", "frac_toxic"] if table == "C" else ["n_s", "mean_peg"]
        best_rank = {}
        for frame in tables.values():
            for rank, subgroup in enumerate(frame.index[:top]):
                best_rank[subgroup] = min(rank, best_rank.get(subgroup, rank))
        order = sorted(best_rank, key=lambda subgroup: (best_rank[subgroup], str(subgroup)))

        frames = {}
        for variant, frame in tables.items():
            found = frame[columns].to_dict("index") if len(frame) else {}
            rows = [found.get(subgroup, dict.fromkeys(columns, np.nan)) for subgroup in order]
            frames[variant] = pd.DataFrame(rows, index=pd.Index(order, tupleize_cols=False), columns=columns)
        return pd.concat(frames, axis=1)

    def subgroup_overlap(self, setting=None, corpus=None, table="C", top=10):
        """Do the formulations find the same subgroups? The Jaccard overlap, for every
        pair, of the `top` first subgroups of the C table (the most frequent) or of
        the P table (the highest mean PEG)."""
        setting, corpus = self._first_setting(setting), self._one_corpus(corpus)
        if table not in ("C", "P"):
            raise ValueError("table must be 'C' or 'P'.")
        found = {v: set(self.runs_[setting, v, corpus][1][table].index[:top]) for v in self.variants}
        matrix = pd.DataFrame(1.0, index=self.variants, columns=self.variants)
        for a, b in itertools.combinations(self.variants, 2):
            matrix.loc[a, b] = matrix.loc[b, a] = jaccard(found[a], found[b])
        return matrix

    # -- recovery of the true structure (ground truth) ----------------

    def recovery_by_k(self):
        """Mean recovery per (setting, variant, true number of active dimensions),
        pooled over the texts of the corpora with ground truth, with the number of
        `texts`. Low precision means extra dimensions were added, low recall that true
        ones were missed."""
        self._require_truth()
        frames = [
            results["recovery"].assign(setting=setting, variant=variant, corpus=corpus)
            for (setting, variant, corpus), (_, results) in self.runs_.items()
            if "recovery" in results
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
        self._require_truth()
        overview = self.overview(self.truth_corpora)
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

    def ari_by_k(self):
        """Mean ARI between the formulations per (setting, pair of formulations, true
        number of active dimensions), pooled over the corpora with ground truth,
        plus the row "all pairs"."""
        self._require_truth()
        per_text = self._ari_per_text()
        per_text = per_text[per_text["k_true"].notna()].copy()
        per_text["pair"] = per_text["a"] + " - " + per_text["b"]
        pairs = per_text.groupby(["setting", "pair", "k_true"], sort=False)["ari"].mean()
        every = per_text.groupby(["setting", "k_true"], sort=False)["ari"].mean().reset_index()
        every.insert(1, "pair", "all pairs")
        table = pd.concat([pairs.reset_index(), every], ignore_index=True).set_index(["setting", "pair", "k_true"])["ari"]
        pair_order = [f"{a} - {b}" for a, b in itertools.combinations(self.variants, 2)] + ["all pairs"]
        return _in_order(table, ["setting", "pair", "k_true"], setting=list(self.settings_), pair=pair_order)

    # -- one text -----------------------------------------------------

    def text_summary(self, corpus, text_id, setting=None):
        """One line per formulation for one text: first split, dimensions used,
        leaves and depth, and with ground truth the Jaccard with the true dimensions."""
        setting = self._first_setting(setting)
        truth = self.corpora[corpus][1]
        true_dims = truth[text_id]["active_dims"] if truth is not None else None
        rows = []
        for variant in self.variants:
            tree = self.runs_[setting, variant, corpus][0].trees_.get(text_id)
            if tree is None:
                continue
            used = sorted({dim for leaf in tree.get_leaves() for dim, _ in leaf["path"]})
            row = {
                "formulation": variant,
                "first split": tree.get_root().get("split_dim", "none"),
                "dimensions used": ", ".join(used) or "none",
                "leaves": tree.n_leaves,
                "depth": tree.depth,
            }
            if true_dims is not None:
                row["Jaccard"] = jaccard(used, true_dims)
            rows.append(row)
        return pd.DataFrame(rows).set_index("formulation")

    def show_text(self, corpus, text_id, setting=None):
        """Print one text (its true structure, if there is ground truth), the summary
        of every formulation and the trees themselves."""
        setting = self._first_setting(setting)
        truth = self.corpora[corpus][1]
        if truth is not None:
            entry = truth[text_id]
            alpha = entry.get("alpha", {})
            print("#" * 10, f"{corpus}, text {text_id} | true active dimensions:",
                  ", ".join(f"{d} (alpha {alpha[d]:.2f})" if d in alpha else d for d in entry["active_dims"])
                  or "none")
        else:
            ndfu = self.runs_[setting, self.variants[0], corpus][0].overall_ndfu_.get(text_id, float("nan"))
            print("#" * 10, f"{corpus}, text {text_id} | nDFU {ndfu:.3f}")
        print(self.text_summary(corpus, text_id, setting).round(2).to_string())
        for variant in self.variants:
            tree = self.runs_[setting, variant, corpus][0].trees_.get(text_id)
            if tree is not None:
                print(f"--- {variant} ---")
                tree.render()

    # -- saving -------------------------------------------------------

    def save(self, out_dir):
        """Write the tables to `out_dir`: selected_settings, per_run, overview, per_text,
        retention_curve; per setting the dimension usage, the depth distribution, the
        similarity matrices (ARI, NMI, dimensions, first split; across all settings
        too) and the F, C and P tables (in fcp/); with ground truth also recovery_by_k
        and ari_by_k.
        Returns the folder."""
        self._require_run()
        out_dir = Path(out_dir)
        (out_dir / "fcp").mkdir(parents=True, exist_ok=True)

        pd.DataFrame.from_dict(self.settings_, orient="index").to_csv(out_dir / "selected_settings.csv")
        self.per_run().to_csv(out_dir / "per_run.csv", index=False)
        self.overview().to_csv(out_dir / "overview.csv")
        self.per_text().to_csv(out_dir / "per_text.csv", index=False)
        self.retention_curve().to_csv(out_dir / "retention_curve.csv")

        for setting in self.settings_:
            self.dimension_usage(setting).to_csv(out_dir / f"dimension_usage_setting{setting}.csv")
            self.depth_distribution(setting).to_csv(out_dir / f"depth_distribution_setting{setting}.csv")
            for measure in ("ari", "nmi", "dims", "first_split"):
                self.similarity(measure, setting).to_csv(out_dir / f"similarity_{measure}_setting{setting}.csv")
            for corpus in self.corpora:
                matrix, per_text = self._ari_of(setting, corpus)
                matrix.to_csv(out_dir / f"ari_setting{setting}_{corpus}.csv")
                per_text.to_csv(out_dir / f"ari_per_text_setting{setting}_{corpus}.csv", index=False)
                for variant in self.variants:
                    for name, table in self.fcp(setting, corpus, variant).items():
                        table.to_csv(out_dir / "fcp" / f"{name}_setting{setting}_{corpus}_{variant}.csv")

        if len(self.settings_) > 1:
            for measure in ("ari", "nmi", "dims", "first_split"):
                self.similarity(measure).to_csv(out_dir / f"similarity_{measure}_all_settings.csv")

        if self.truth_corpora:
            self.recovery_by_k().to_csv(out_dir / "recovery_by_k.csv")
            self.ari_by_k().to_csv(out_dir / "ari_by_k.csv")
        return out_dir
