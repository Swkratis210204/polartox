"""
polartox.polarized_tree -- single-tree construction and inspection.

Everything about ONE text's polarized tree lives here: the splitting
algorithm (detect_polarized_subgroups), its small numeric helpers
(ndfu_score, compute_peg), and the PolarizedTree class that wraps a
built tree and answers questions about it (render, inspect, walk nodes,
pull a node's/leaf's rating distribution).

Corpus-level orchestration across many texts lives in pipeline.py
(PolarizedTreesPipeline), which builds and holds PolarizedTree instances
but never reaches into their internals directly.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from ndfu import dfu, pdf


def ndfu_score(ratings, scale):
    if len(ratings) == 0:
        return float("nan")
    return dfu(pdf(list(ratings), list(range(1, scale + 1))))


def default_theta_pole(scale):
    """Ratings >= this count as toxic when no theta_pole is given."""
    return scale // 2 + 1


@lru_cache(maxsize=None)
def _ndfu_from_counts(counts, n):
    """nDFU of a rating histogram; `counts` is a tuple over the scale and
    `n` the number of ratings (which may exceed sum(counts) if some
    ratings fall outside the scale, exactly as in ndfu_score)."""
    if n == 0:
        return float("nan")
    return dfu(np.array(counts, dtype=float) / n)


def _histograms(codes, ratings, n_groups, scale):
    """Rating histogram per group: shape (n_groups, scale). Ratings that are
    not integers within 1..scale count towards no bin."""
    valid = (ratings >= 1) & (ratings <= scale) & (ratings == np.floor(ratings))
    flat = codes[valid] * scale + (ratings[valid].astype(np.int64) - 1)
    return np.bincount(flat, minlength=n_groups * scale).reshape(n_groups, scale)


def print_histogram(ratings, scale, label="ratings", indent=0, width=30):
    pad = "  " * indent
    counts = pd.Series(ratings).value_counts().reindex(range(1, scale + 1), fill_value=0)
    peak = max(counts.max(), 1)
    print(f"{pad}{label} (n={len(ratings)}):")
    for rating, count in counts.items():
        print(f"{pad}  {rating}: {'#' * round(width * count / peak)} ({count})")


PEG_VARIANTS = ("max", "avg", "min", "mean", "harmonic")

_REMOVED_VARIANTS = {
    "var": "use 'avg', the size-weighted average",
    "beta": "the max/avg-only PEGbeta was replaced by 'harmonic', the harmonic mean "
            "of max, avg and min",
}


def check_variant(variant):
    """Raise ValueError unless `variant` is one of PEG_VARIANTS."""
    if variant in PEG_VARIANTS:
        return
    options = ", ".join(repr(v) for v in PEG_VARIANTS)
    removed = _REMOVED_VARIANTS.get(variant) if isinstance(variant, str) else None
    hint = f" ({variant!r} no longer exists: {removed})" if removed else ""
    raise ValueError(f"variant must be one of {options}, got {variant!r}{hint}")


def compute_peg(node_ratings, groups, scale, variant="harmonic", beta=1.0):
    """PEG of splitting a node into `groups` (group -> ratings).

    Three base formulations compare the node's nDFU with its subgroups':
    "max" (the most polarized subgroup), "avg" (the size-weighted average)
    and "min" (the least polarized subgroup). "mean" is their arithmetic mean
    and "harmonic" their harmonic mean (0 if any of the three is 0). In
    "harmonic", beta weights avg against max and min (weights 1, beta**2, 1),
    so the default beta=1 is the plain harmonic mean; the other variants
    ignore beta.
    """
    global_ndfu = ndfu_score(node_ratings, scale)
    group_ndfus = {v: ndfu_score(r, scale) for v, r in groups.items()}
    n = len(node_ratings)
    sizes = {v: len(r) for v, r in groups.items()}

    peg = _combine_peg(global_ndfu, group_ndfus, sizes, n, variant, beta)
    return peg, global_ndfu, group_ndfus


def _combine_peg(global_ndfu, group_ndfus, sizes, n, variant, beta):
    """PEG from precomputed nDFUs; `group_ndfus` and `sizes` map group -> value."""
    peg_max = abs(global_ndfu - max(group_ndfus.values()))
    peg_avg = abs(global_ndfu - sum(sizes[v] / n * group_ndfus[v] for v in group_ndfus))
    peg_min = abs(global_ndfu - min(group_ndfus.values()))

    if variant == "max":
        peg = peg_max
    elif variant == "avg":
        peg = peg_avg
    elif variant == "min":
        peg = peg_min
    elif variant == "mean":
        peg = (peg_max + peg_avg + peg_min) / 3
    elif variant == "harmonic":
        if peg_max > 0 and peg_avg > 0 and peg_min > 0:
            peg = (2 + beta**2) / (1 / peg_max + beta**2 / peg_avg + 1 / peg_min)
        else:
            peg = 0.0
    else:
        check_variant(variant)
    return peg


def _leaf(ratings, path, ndfu_val, theta_pole, reason):
    n = len(ratings)
    p_tox = float((ratings >= theta_pole).sum()) / n if n else float("nan")
    pole = "toxic" if p_tox > 0.5 else "civil" if p_tox < 0.5 else "indeterminate"
    return {"path": list(path), "n": n, "ndfu": ndfu_val, "p_tox": p_tox, "pole": pole,
            "is_leaf": True, "stop_reason": reason}


def detect_polarized_subgroups(
    data, dims, min_size, h, max_depth, scale,
    theta_pole=None, theta_stop=0.15, variant="harmonic", beta=1.0,
    relative_h=False,
    verbose=False, return_tree=False,
):
    """
    Greedily split one text's annotators by the dimension with the highest
    PEG, recursing inside each subgroup until a stopping rule triggers.

    Parameters
    ----------
    data : DataFrame
        One text's rows: a "rating" column plus one column per dimension.
    dims : list of str
        Candidate splitting dimensions; a dimension is used at most once
        per branch.
    min_size : int or callable
        Fixed absolute minimum subgroup size, OR a callable min_size(depth)
        returning a FRACTION of the text's total annotators for that depth
        -- lets the threshold tighten as the tree goes deeper, since early
        splits (finding the 1st/2nd true cause) are reliable on large
        groups, while late splits risk mistaking residual noise (from
        imperfect intensity/alpha in the data) for a genuine extra cause.
        A dimension is skipped at a node if any of its groups is smaller.
    h : float
        Gain threshold: a node splits only if its best PEG exceeds h.
    max_depth : int
        Nodes deeper than this become leaves (the root is at depth 1).
    scale : int
        Rating scale, ratings are integers in [1, scale].
    theta_pole : int or None
        Ratings >= theta_pole count as toxic when labeling a leaf's pole.
        Defaults to scale // 2 + 1.
    theta_stop : float or None
        A node with nDFU below this is already unpolarized and becomes a
        leaf. None disables this rule.
    variant, beta :
        PEG formulation: "max", "avg", "min", "mean" or "harmonic", and the
        beta weight used by "harmonic" (see compute_peg). An unknown variant
        raises ValueError immediately, even if no node ever needs a split.
    relative_h : bool
        If True, compare best PEG / node nDFU to h instead of the raw PEG.
    verbose : bool
        Print each node's nDFU and rating histogram while building.
    return_tree : bool
        If True, return (leaves, root) instead of just the leaves.

    Returns
    -------
    list of dict, or (list of dict, dict)
        The leaves (path, n, ndfu, p_tox, pole, stop_reason), and with
        return_tree=True also the nested root node.
    """
    check_variant(variant)
    theta_pole = theta_pole if theta_pole is not None else default_theta_pole(scale)
    n_total = len(data)
    leaves = []

    def resolve_min_size(depth):
        if callable(min_size):
            return max(2, round(min_size(depth) * n_total))
        return min_size

    # Nodes are index arrays into the text's rows. Each dimension is coded
    # once (sorted, like groupby; NaN gets -1 and is dropped from groups),
    # and every split is then a histogram over (group, rating) -- no
    # per-node pandas groupby.
    all_ratings = data["rating"].to_numpy()
    coded = {}

    for dim in dims:
        codes, uniques = pd.factorize(data[dim], sort=True)
        coded[dim] = (codes.astype(np.int64), uniques)

    def split(idx, dim):
        """(group values, per-group row indexes) of a node split on `dim`."""
        codes, uniques = coded[dim]
        c = codes[idx]
        return [(uniques[g], idx[c == g]) for g in np.unique(c[c >= 0])]

    def dfs(idx, remaining_dims, depth, path):
        ratings = all_ratings[idx]
        n = len(idx)
        counts = _histograms(np.zeros(n, dtype=np.int64), ratings, 1, scale)[0]
        nd = _ndfu_from_counts(tuple(counts.tolist()), n)
        ms = resolve_min_size(depth)

        if verbose:
            print(f"\n{'  '*depth}[{' -> '.join(f'{d}={v}' for d,v in path) or 'root'}] nDFU={nd:.3f}")
            print_histogram(ratings, scale, indent=depth)

        if theta_stop is not None and nd < theta_stop:
            leaf = _leaf(ratings, path, nd, theta_pole, f"nDFU {nd:.3f} < theta_stop")
            leaves.append(leaf)
            return leaf

        if depth > max_depth or not remaining_dims:
            leaf = _leaf(ratings, path, nd, theta_pole, "max_depth/dimension exhaustion")
            leaves.append(leaf)
            return leaf

        best_dim, best_peg = None, 0
        for dim in remaining_dims:
            codes = coded[dim][0][idx]
            present = codes >= 0
            k = len(coded[dim][1])
            sizes = np.bincount(codes[present], minlength=k)
            keep = np.flatnonzero(sizes)

            if (sizes[keep] < ms).any():
                continue

            hists = _histograms(codes[present], ratings[present], k, scale)
            group_ndfus = {g: _ndfu_from_counts(tuple(hists[g].tolist()), int(sizes[g]))
                           for g in keep}
            peg = _combine_peg(nd, group_ndfus, {g: int(sizes[g]) for g in keep},
                               n, variant, beta)
            if peg > best_peg:
                best_dim, best_peg = dim, peg

        if best_dim is not None and relative_h:
            comparison_value = best_peg / nd if nd > 0 else 0
        else:
            comparison_value = best_peg

        if best_dim is None or comparison_value <= h:
            reason = "no dim passed min_size" if best_dim is None else f"best PEG {best_peg:.3f} (relative={comparison_value:.3f}) <= h"
            leaf = _leaf(ratings, path, nd, theta_pole, reason)
            leaves.append(leaf)
            return leaf

        remaining_next = [d for d in remaining_dims if d != best_dim]
        children = {v: dfs(child, remaining_next, depth + 1, path + [(best_dim, v)])
                    for v, child in split(idx, best_dim)}
        return {"path": list(path), "n": n, "ndfu": nd, "is_leaf": False,
                "split_dim": best_dim, "peg": best_peg, "children": children}

    root = dfs(np.arange(n_total), list(dims), 1, [])
    return (leaves, root) if return_tree else leaves


def render_tree_text(node, label="root", prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    if node["is_leaf"]:
        print(f"{prefix}{connector}{label} (n={node['n']}, nDFU={node['ndfu']:.3f}) -> [{node['pole']}] p_tox={node['p_tox']:.3f}")
        return
    print(f"{prefix}{connector}{label} (n={node['n']}, nDFU={node['ndfu']:.3f}) split '{node['split_dim']}' (PEG={node['peg']:.3f})")
    child_prefix = prefix + ("    " if is_last else "│   ")
    items = list(node["children"].items())
    for i, (v, child) in enumerate(items):
        render_tree_text(child, f"{node['split_dim']}={v}", child_prefix, i == len(items) - 1)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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


class PolarizedTree:
    """
    One text's polarized tree: the nested split structure (root) plus the
    flat list of leaves it bottoms out in. Built via PolarizedTree.build();
    everything else on this class answers questions about that structure
    (render it, inspect a node's rating distribution, walk internal nodes)
    without needing a corpus or a pipeline.
    """

    def __init__(self, root, leaves, text_id=None, scale=None, theta_pole=None):
        self.root = root
        self.leaves = leaves
        self.text_id = text_id
        self.scale = scale
        self.theta_pole = theta_pole

    @classmethod
    def build(cls, data, dims, min_size, h, max_depth, scale,
              theta_pole=None, theta_stop=0.15, variant="harmonic", beta=1.0,
              relative_h=False, text_id=None, verbose=False):
        leaves, root = detect_polarized_subgroups(
            data, dims, min_size, h, max_depth, scale,
            theta_pole=theta_pole, theta_stop=theta_stop, variant=variant, beta=beta,
            relative_h=relative_h, verbose=verbose, return_tree=True,
        )
        resolved_theta_pole = theta_pole if theta_pole is not None else default_theta_pole(scale)
        return cls(root, leaves, text_id=text_id, scale=scale, theta_pole=resolved_theta_pole)

    def get_root(self):
        return self.root

    def get_leaves(self):
        return self.leaves

    @property
    def n_leaves(self):
        return len(self.leaves)

    @property
    def depth(self):
        """Max leaf depth (number of splits from root to deepest leaf)."""
        return max((len(leaf["path"]) for leaf in self.leaves), default=0)

    def internal_nodes(self, root=None, depth=1, path=()):
        """Yield (depth, split_dim, peg, path, node) for every non-leaf node."""
        root = self.root if root is None else root
        if root["is_leaf"]:
            return
        yield depth, root["split_dim"], root["peg"], path, root
        for v, child in root["children"].items():
            yield from self.internal_nodes(child, depth + 1, path + ((root["split_dim"], v),))

    def find_node(self, path):
        """Return the node dict reached by following `path` (a sequence of
        (dim, value) pairs) from the root, or None if the path doesn't exist."""
        node = self.root
        for dim, value in path:
            if node["is_leaf"] or node["split_dim"] != dim or value not in node["children"]:
                return None
            node = node["children"][value]
        return node

    def leaf_labels(self, dataset):
        """For every annotator row of this text, the index (in `self.leaves`) of
        the leaf that annotator ends up in; -1 if no leaf holds the row (for
        example a missing value in a split dimension).

        Rows are those of `dataset` belonging to this text, in their original
        order, so the labels of two trees built for the same text can be
        compared with `adjusted_rand_index`.
        """
        rows = dataset
        if self.text_id is not None and "text_id" in dataset.columns:
            rows = rows[rows["text_id"] == self.text_id]

        labels = np.full(len(rows), -1, dtype=np.int64)
        coded = {}      # each dimension is coded once; a missing value gets code -1
        for k, leaf in enumerate(self.leaves):
            member = np.ones(len(rows), dtype=bool)
            for dim, value in leaf["path"]:
                if dim not in coded:
                    codes, uniques = pd.factorize(rows[dim])
                    coded[dim] = (codes, {v: i for i, v in enumerate(uniques)})
                codes, code_of = coded[dim]
                wanted = code_of.get(value, -1)
                if wanted < 0:          # this value does not occur in these rows
                    member[:] = False
                    break
                member &= codes == wanted
            labels[member] = k
        return labels

    def node_ratings(self, dataset, path=()):
        """Filter `dataset` down to the rows belonging to the node at `path`
        (empty path = root, i.e. this whole text)."""
        subgroup_data = dataset
        if self.text_id is not None and "text_id" in dataset.columns:
            subgroup_data = subgroup_data[subgroup_data["text_id"] == self.text_id]
        for dim, value in path:
            subgroup_data = subgroup_data[subgroup_data[dim] == value]
        return subgroup_data["rating"].to_numpy()

    def node_distribution(self, dataset, path=(), indent=0):
        """Print the rating histogram for one specific node (root or any
        internal/leaf node), identified by its path from the root."""
        label = " -> ".join(f"{d}={v}" for d, v in path) or "root"
        print_histogram(self.node_ratings(dataset, path), self.scale, label=label, indent=indent)

    def leaf_distributions(self, dataset):
        """Print the rating histogram for every leaf in this tree."""
        for leaf in self.leaves:
            self.node_distribution(dataset, path=leaf["path"])

    def render(self):
        render_tree_text(self.root)

    def inspect(self, dataset, show_distributions=False):
        """Print this tree, optionally with a rating histogram at every node."""

        def walk(node, path=(), depth=0):
            label = " -> ".join(f"{d}={v}" for d, v in path) or "root"
            print(f"\n{'  '*depth}[{label}] nDFU={node['ndfu']:.3f}")
            if show_distributions:
                print_histogram(self.node_ratings(dataset, path), self.scale, indent=depth)
            if node["is_leaf"]:
                print(f"{'  '*depth}  -> LEAF [{node['pole']}] p_tox={node['p_tox']:.3f} ({node['stop_reason']})")
            else:
                print(f"{'  '*depth}  split on '{node['split_dim']}' (PEG={node['peg']:.3f})")
                for v, child in node["children"].items():
                    walk(child, path + ((node["split_dim"], v),), depth + 1)

        walk(self.root)
