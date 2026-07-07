"""
polartox.polarized_trees -- Polarized Trees: tree construction (Steps 1-5)
plus corpus-level metrics (Step 6), optionally enriched with ground truth.

Requires: pip install polartox[ndfu]
"""

import pandas as pd
import numpy as np
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
    """
    PRG for one candidate split, per the paper's definitions:
        PRGmax(A, dim)  = |nDFUglobal(A) - max_k nDFU(gk)|
        PRGvar(A, dim)  = |nDFUglobal(A) - sum_k (Nk/N) * nDFU(gk)|
        PRGbeta(A, dim) = (1+beta^2) * PRGmax * PRGvar / (beta^2 * PRGmax + PRGvar)

    variant : "max", "var", or "beta" (recommended default -- see README
    "Departures from the paper": PRGmax alone was found to pick a spurious
    dimension over a real one in a concrete tested case; PRGbeta corrected
    this while avoiding PRGvar's tendency to produce values too small to
    clear a sensible h).
    """
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
    relative_h=False,
    verbose=False, return_tree=False,
):
    """
    Steps 3-5: build one Polarized Tree for a single text's annotations.

    theta_stop (not in the paper): stops a node immediately, before
    searching for any split, if the node's own nDFU is already below this
    value. Set to None to disable (not recommended -- fragments
    already-resolved nodes on pure sampling noise otherwise).

    relative_h (not in the paper): if True, a candidate split is accepted
    only if best_prg / node's own nDFU > h, instead of comparing best_prg
    to h directly. Recommended True -- see README for validation.

    variant="beta" (harmonic mean of PRGmax/PRGvar) empirically
    outperformed PRGmax alone (the paper's stated default) on synthetic
    validation data.
    """
    theta_pole = theta_pole if theta_pole is not None else scale // 2 + 1
    leaves = []

    def dfs(node_data, remaining_dims, depth, path):
        ratings = node_data["rating"].to_numpy()
        nd = ndfu_score(ratings, scale)

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
            if any(len(g) < min_size for g in groups.values()):
                continue
            prg, _, _ = compute_prg(ratings, groups, scale, variant, beta)
            if prg > best_prg:
                best_dim, best_prg = dim, prg

        comparison_value = (best_prg / nd if nd > 0 else 0) if (best_dim is not None and relative_h) else best_prg

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


class PolarizedTreesPipeline:
    """
    Full pipeline: Step 1-2 (filter) -> Step 3-5 (build trees) -> Step 6 (metrics).

    F/C/P and diagnostics() always work, with or without ground truth.
    Pass ground_truth to run_full_evaluation() to additionally get
    recovery metrics (jaccard/precision/recall) and validation columns
    added to F/C/P -- only meaningful on synthetic data with known answers.

    min_size_frac (recommended) sets subgroup-size threshold as a fraction
    of each text's own annotator count, so the same config works across
    datasets with different annotator counts. Pass min_size for a fixed
    absolute count instead.

    theta_stop stops a node immediately, before searching for any split,
    if the node's own nDFU is already below this value. This is an
    ABSOLUTE floor -- it can cut off a branch before a real, weaker
    co-active cause is tested, if a stronger cause already resolved most
    of the disagreement. Lower values (e.g. 0.10) recover some of these
    cases; see relative_h for a complementary fix. See
    detect_polarized_subgroups for full details on both.

    relative_h (recommended True): compares a candidate split's PRG to h
    as a fraction of the node's own remaining nDFU, instead of as a raw
    absolute value. Tested to improve recovery on every metric across
    multiple synthetic corpora with no regressions; see README.
    """

    def __init__(self, dims, scale, theta_filter, h, max_depth,
                 min_size=None, min_size_frac=0.03,
                 variant="beta", beta=1.0, theta_pole=None, theta_stop=0.15,
                 relative_h=False):
        self.dims = list(dims)
        self.scale = scale
        self.theta_filter = theta_filter
        self.h = h
        self.max_depth = max_depth
        self.variant = variant
        self.beta = beta
        self.theta_pole = theta_pole if theta_pole is not None else scale // 2 + 1
        self.theta_stop = theta_stop
        self.relative_h = relative_h

        self.min_size = min_size
        self.min_size_frac = min_size_frac if min_size is None else None

        self.trees_ = {}
        self.overall_ndfu_ = {}
        self.retained_ids_ = []

    def _min_size_for(self, n):
        return max(2, round(self.min_size_frac * n)) if self.min_size_frac else self.min_size

    def filter_polarized_texts(self, dataset):
        """Steps 1-2: compute nDFU per text, keep only those >= theta_filter."""
        self.overall_ndfu_ = {tid: ndfu_score(g["rating"].to_numpy(), self.scale)
                               for tid, g in dataset.groupby("text_id")}
        self.retained_ids_ = [t for t, nd in self.overall_ndfu_.items() if nd >= self.theta_filter]
        return self.retained_ids_

    def build_all_trees(self, dataset, text_ids=None):
        """Steps 3-5: build a tree for every retained text."""
        text_ids = text_ids or self.retained_ids_
        if not text_ids:
            raise RuntimeError("Call filter_polarized_texts(dataset) first, or pass text_ids.")
        self.trees_ = {}
        for tid in text_ids:
            text_data = dataset[dataset["text_id"] == tid]
            min_size = self._min_size_for(len(text_data))
            self.trees_[tid] = detect_polarized_subgroups(
                text_data, self.dims, min_size, self.h, self.max_depth, self.scale,
                self.theta_pole, self.theta_stop, self.variant, self.beta,
                relative_h=self.relative_h, return_tree=True,
            )
        return self.trees_

    def _internal_nodes(self, root, depth=1, path=()):
        if root["is_leaf"]:
            return
        yield depth, root["split_dim"], root["prg"], path, root
        for v, child in root["children"].items():
            yield from self._internal_nodes(child, depth + 1, path + ((root["split_dim"], v),))

    def dimension_frequency(self, ground_truth=None):
        """Step 6.1 (F): splitting-dimension frequency by depth."""
        rows = [{"text_id": t, "depth": d, "dim": dim}
                for t, (_, root) in self.trees_.items()
                for d, dim, prg, path, node in self._internal_nodes(root)]
        if not rows:
            return pd.DataFrame()
        F = pd.DataFrame(rows).pivot_table(index="dim", columns="depth", values="text_id",
                                            aggfunc="count", fill_value=0)
        if ground_truth is not None:
            truly_active = {d for gt in ground_truth.values() for d in gt["active_dims"]}
            F["ever_truly_active"] = [d in truly_active for d in F.index]
        return F

    def subgroup_pole_consistency(self, ground_truth=None):
        """Step 6.2 (C): pole stability per intersectional subgroup."""
        rows = [{"subgroup": tuple(sorted(leaf["path"])), "text_id": t, "pole": leaf["pole"]}
                for t, (leaves, _) in self.trees_.items() for leaf in leaves
                if leaf["pole"] != "indeterminate"]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        C = df.groupby("subgroup").agg(
            n_s=("text_id", "count"),
            frac_toxic=("pole", lambda s: (s == "toxic").mean()),
            frac_civil=("pole", lambda s: (s == "civil").mean()),
        ).sort_values("n_s", ascending=False)

        if ground_truth is not None:
            def true_lean_agrees(row):
                subgroup, text_id, pole = row["subgroup"], row["text_id"], row["pole"]
                gt = ground_truth[text_id]
                leans = set()
                for dim, value in subgroup:
                    if dim in gt["active_dims"]:
                        leans.add(gt["lean"][dim].get(value))
                if len(leans) != 1:
                    return np.nan
                return list(leans)[0] == pole
            df["agrees"] = df.apply(true_lean_agrees, axis=1)
            C["true_lean_match_rate"] = df.groupby("subgroup")["agrees"].mean()
        return C

    def subgroup_prg(self, ground_truth=None):
        """Step 6.3 (P): mean PRG of the split producing each subgroup."""
        rows = []
        for t, (_, root) in self.trees_.items():
            for d, dim, prg, path, node in self._internal_nodes(root):
                for v, child in node["children"].items():
                    if child["is_leaf"]:
                        rows.append({"subgroup": tuple(sorted(path + ((dim, v),))),
                                     "text_id": t, "dim": dim, "prg": prg})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        P = df.groupby("subgroup").agg(n_s=("prg", "count"), mean_prg=("prg", "mean")).sort_values("mean_prg", ascending=False)

        if ground_truth is not None:
            def true_alpha(row):
                gt = ground_truth[row["text_id"]]
                return gt["alpha"].get(row["dim"]) if row["dim"] in gt["active_dims"] else np.nan
            df["alpha"] = df.apply(true_alpha, axis=1)
            P["mean_true_alpha"] = df.groupby("subgroup")["alpha"].mean()
        return P

    def diagnostics(self):
        """Ground-truth-free corpus diagnostics (usable on real data)."""
        n_leaves, depths, residual, top_prgs, indet, used = [], [], [], [], [], set()
        for t, (leaves, root) in self.trees_.items():
            n_leaves.append(len(leaves))
            for leaf in leaves:
                residual.append(leaf["ndfu"])
                depths.append(len(leaf["path"]))
                indet.append(leaf["pole"] == "indeterminate")
            if not root["is_leaf"]:
                top_prgs.append(root["prg"])
            used |= {dim for d, dim, prg, path, node in self._internal_nodes(root)}
        n_total = len(self.overall_ndfu_) or len(self.trees_)
        return {
            "retention_rate": len(self.retained_ids_) / n_total,
            "mean_leaves": float(np.mean(n_leaves)) if n_leaves else np.nan,
            "mean_depth": float(np.mean(depths)) if depths else np.nan,
            "mean_residual_ndfu": float(np.mean(residual)) if residual else np.nan,
            "mean_top_split_prg": float(np.mean(top_prgs)) if top_prgs else np.nan,
            "indeterminate_rate": float(np.mean(indet)) if indet else np.nan,
            "dims_never_used": sorted(set(self.dims) - used),
        }

    def recovery_metrics(self, ground_truth):
        """Precision/recall/jaccard/exact_match per text -- synthetic data only."""
        rows = []
        for t, (leaves, _) in self.trees_.items():
            true_d = set(ground_truth[t]["active_dims"])
            found_d = {d for leaf in leaves for d, v in leaf["path"]}
            precision = len(true_d & found_d) / len(found_d) if found_d else float(not true_d)
            recall = len(true_d & found_d) / len(true_d) if true_d else float(not found_d)
            rows.append({"text_id": t, "k_true": len(true_d), "true_dims": sorted(true_d),
                         "found_dims": sorted(found_d), "jaccard": jaccard(true_d, found_d),
                         "precision": precision, "recall": recall,
                         "exact_match": sorted(true_d) == sorted(found_d)})
        return pd.DataFrame(rows)

    def inspect_tree(self, text_id, dataset, show_distributions=False):
        """Print one text's tree, optionally with a rating histogram at every node."""
        leaves, root = self.trees_[text_id]

        def walk(node, path=(), depth=0):
            subgroup_data = dataset[dataset["text_id"] == text_id]
            for dim, v in path:
                subgroup_data = subgroup_data[subgroup_data[dim] == v]
            label = " -> ".join(f"{d}={v}" for d, v in path) or "root"
            print(f"\n{'  '*depth}[{label}] nDFU={node['ndfu']:.3f}")
            if show_distributions:
                print_histogram(subgroup_data["rating"].to_numpy(), self.scale, indent=depth)
            if node["is_leaf"]:
                print(f"{'  '*depth}  -> LEAF [{node['pole']}] p_tox={node['p_tox']:.3f} ({node['stop_reason']})")
            else:
                print(f"{'  '*depth}  split on '{node['split_dim']}' (PRG={node['prg']:.3f})")
                for v, child in node["children"].items():
                    walk(child, path + ((node["split_dim"], v),), depth + 1)

        walk(root)

    def run_full_evaluation(self, dataset, ground_truth=None, verbose=True):
        """Runs everything. F/C/P + diagnostics always; + recovery if ground_truth given."""
        self.filter_polarized_texts(dataset)
        self.build_all_trees(dataset)

        results = {
            "F": self.dimension_frequency(ground_truth),
            "C": self.subgroup_pole_consistency(ground_truth),
            "P": self.subgroup_prg(ground_truth),
            "diagnostics": self.diagnostics(),
        }

        if ground_truth is not None:
            results["recovery"] = self.recovery_metrics(ground_truth)

        if verbose:
            print("=== diagnostics ===")
            for k, v in results["diagnostics"].items():
                print(f"  {k}: {v}")
            if ground_truth is not None:
                r = results["recovery"]
                print("\n=== recovery ===")
                print(f"  mean jaccard:     {r['jaccard'].mean():.4f}")
                print(f"  mean precision:   {r['precision'].mean():.4f}")
                print(f"  mean recall:      {r['recall'].mean():.4f}")
                print(f"  exact match rate: {r['exact_match'].mean():.4f}")

        return results
