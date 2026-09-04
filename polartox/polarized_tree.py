"""
polartox.polarized_tree -- single-tree construction and inspection.

Everything about ONE text's polarized tree lives here: the splitting
algorithm (detect_polarized_subgroups), its small numeric helpers
(ndfu_score, compute_prg), and the PolarizedTree class that wraps a
built tree and answers questions about it (render, inspect, walk nodes,
pull a node's/leaf's rating distribution).

Corpus-level orchestration across many texts lives in polarized_trees.py
(PolarizedTreesPipeline), which builds and holds PolarizedTree instances
but never reaches into their internals directly.
"""

import pandas as pd
from ndfu import dfu, pdf


def ndfu_score(ratings, scale):
    if len(ratings) == 0:
        return float("nan")
    return dfu(pdf(list(ratings), list(range(1, scale + 1))))


def print_histogram(ratings, scale, label="ratings", indent=0, width=30):
    pad = "  " * indent
    counts = pd.Series(ratings).value_counts().reindex(range(1, scale + 1), fill_value=0)
    peak = max(counts.max(), 1)
    print(f"{pad}{label} (n={len(ratings)}):")
    for rating, count in counts.items():
        print(f"{pad}  {rating}: {'#' * round(width * count / peak)} ({count})")


def compute_prg(node_ratings, groups, scale, variant="beta", beta=1.0):
    """PRGmax, PRGvar, or PRGbeta (harmonic mean of both -- recommended default)."""
    global_ndfu = ndfu_score(node_ratings, scale)
    group_ndfus = {v: ndfu_score(r, scale) for v, r in groups.items()}
    n = len(node_ratings)

    prg_max = abs(global_ndfu - max(group_ndfus.values()))
    prg_var = abs(global_ndfu - sum(len(r) / n * group_ndfus[v] for v, r in groups.items()))

    if variant == "max":
        prg = prg_max
    elif variant == "var":
        prg = prg_var
    elif variant == "beta":
        denom = beta**2 * prg_max + prg_var
        prg = (1 + beta**2) * prg_max * prg_var / denom if denom > 0 else 0.0
    else:
        raise ValueError("variant must be 'max', 'var', or 'beta'")
    return prg, global_ndfu, group_ndfus


def _leaf(node_data, path, ndfu_val, theta_pole, reason):
    ratings = node_data["rating"].to_numpy()
    n = len(ratings)
    p_tox = float((ratings >= theta_pole).sum()) / n if n else float("nan")
    pole = "toxic" if p_tox > 0.5 else "civil" if p_tox < 0.5 else "indeterminate"
    return {"path": list(path), "n": n, "ndfu": ndfu_val, "p_tox": p_tox, "pole": pole,
            "is_leaf": True, "stop_reason": reason}


def detect_polarized_subgroups(
    data, dims, min_size, h, max_depth, scale,
    theta_pole=None, theta_stop=0.15, variant="beta", beta=1.0,
    relative_h=False,   # NEW
    verbose=False, return_tree=False,
):
    """
    ...
    min_size : int or callable
        Fixed absolute minimum subgroup size, OR a callable min_size(depth)
        returning a FRACTION of the text's total annotators for that depth
        -- lets the threshold tighten as the tree goes deeper, since early
        splits (finding the 1st/2nd true cause) are reliable on large
        groups, while late splits risk mistaking residual noise (from
        imperfect intensity/alpha in the data) for a genuine extra cause.
    """
    theta_pole = theta_pole if theta_pole is not None else scale // 2 + 1
    n_total = len(data)
    leaves = []

    def resolve_min_size(depth):
        if callable(min_size):
            return max(2, round(min_size(depth) * n_total))
        return min_size

    def dfs(node_data, remaining_dims, depth, path):
        ratings = node_data["rating"].to_numpy()
        nd = ndfu_score(ratings, scale)
        ms = resolve_min_size(depth)

        if verbose:
            print(f"\n{'  '*depth}[{' -> '.join(f'{d}={v}' for d,v in path) or 'root'}] nDFU={nd:.3f}")
            print_histogram(ratings, scale, indent=depth)

        if theta_stop is not None and nd < theta_stop:
            leaf = _leaf(node_data, path, nd, theta_pole, f"nDFU {nd:.3f} < theta_stop")
            leaves.append(leaf)
            return leaf

        if depth > max_depth or not remaining_dims:
            leaf = _leaf(node_data, path, nd, theta_pole, "max_depth/dimension exhaustion")
            leaves.append(leaf)
            return leaf

        best_dim, best_prg = None, 0
        for dim in remaining_dims:
            groups = {v: g["rating"].to_numpy() for v, g in node_data.groupby(dim)}
            if any(len(g) < ms for g in groups.values()):
                continue
            prg, _, _ = compute_prg(ratings, groups, scale, variant, beta)
            if prg > best_prg:
                best_dim, best_prg = dim, prg

        if best_dim is not None and relative_h:
            comparison_value = best_prg / nd if nd > 0 else 0
        else:
            comparison_value = best_prg

        if best_dim is None or comparison_value <= h:
            reason = "no dim passed min_size" if best_dim is None else f"best PRG {best_prg:.3f} (relative={comparison_value:.3f}) <= h"
            leaf = _leaf(node_data, path, nd, theta_pole, reason)
            leaves.append(leaf)
            return leaf

        remaining_next = [d for d in remaining_dims if d != best_dim]
        children = {v: dfs(g, remaining_next, depth + 1, path + [(best_dim, v)])
                    for v, g in node_data.groupby(best_dim)}
        return {"path": list(path), "n": len(ratings), "ndfu": nd, "is_leaf": False,
                "split_dim": best_dim, "prg": best_prg, "children": children}

    root = dfs(data, list(dims), 1, [])
    return (leaves, root) if return_tree else leaves


def render_tree_text(node, label="root", prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    if node["is_leaf"]:
        print(f"{prefix}{connector}{label} (n={node['n']}, nDFU={node['ndfu']:.3f}) -> [{node['pole']}] p_tox={node['p_tox']:.3f}")
        return
    print(f"{prefix}{connector}{label} (n={node['n']}, nDFU={node['ndfu']:.3f}) split '{node['split_dim']}' (PRG={node['prg']:.3f})")
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
              theta_pole=None, theta_stop=0.15, variant="beta", beta=1.0,
              relative_h=False, text_id=None, verbose=False):
        leaves, root = detect_polarized_subgroups(
            data, dims, min_size, h, max_depth, scale,
            theta_pole=theta_pole, theta_stop=theta_stop, variant=variant, beta=beta,
            relative_h=relative_h, verbose=verbose, return_tree=True,
        )
        resolved_theta_pole = theta_pole if theta_pole is not None else scale // 2 + 1
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
        """Yield (depth, split_dim, prg, path, node) for every non-leaf node."""
        root = self.root if root is None else root
        if root["is_leaf"]:
            return
        yield depth, root["split_dim"], root["prg"], path, root
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
                print(f"{'  '*depth}  split on '{node['split_dim']}' (PRG={node['prg']:.3f})")
                for v, child in node["children"].items():
                    walk(child, path + ((node["split_dim"], v),), depth + 1)

        walk(self.root)
