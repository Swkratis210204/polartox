"""
polartox.pipeline -- corpus-level orchestration (Steps 1-6):
filter polarized texts, build a PolarizedTree per text, aggregate
corpus-level metrics, optionally enriched with ground truth.

Single-tree construction/inspection lives in polartox.polarized_tree
(PolarizedTree, detect_polarized_subgroups); re-exported here for
convenience.

Requires: pip install polartox[ndfu]
"""

import pandas as pd
import numpy as np

from polartox.polarized_tree import (
    ndfu_score,
    print_histogram,
    compute_prg,
    detect_polarized_subgroups,
    render_tree_text,
    jaccard,
    PolarizedTree,
)

__all__ = [
    "ndfu_score", "print_histogram", "compute_prg", "detect_polarized_subgroups",
    "render_tree_text", "jaccard", "PolarizedTree", "PolarizedTreesPipeline",
]


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

    This class owns corpus-level orchestration and aggregation only; each
    text's tree is a PolarizedTree instance (polartox.polarized_tree),
    accessed through its public API (get_root/get_leaves/internal_nodes),
    never by reaching into raw node dicts directly.
    """

    def __init__(self, dims, scale, theta_filter, h, max_depth,
             min_size=None, min_size_frac=0.03, min_size_frac_schedule=None,
             variant="beta", beta=1.0, theta_pole=None, theta_stop=0.15,
             relative_h=False):
        """
        min_size_frac_schedule : (base, step) tuple or None
            If given, min_size_frac tightens with depth: frac(depth) =
            base + step * (depth - 1). Overrides min_size_frac and min_size.
            Empirically motivated: a single fixed threshold either lets noise
            through deep in the tree (too loose) or blocks legitimate late
            splits in texts with 3+ true causes (too strict) -- see project
            notes for the k=2 vs k=3/4 tradeoff this resolves.
        """
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

        if min_size_frac_schedule is not None:
            base, step = min_size_frac_schedule
            self.min_size = lambda depth: base + step * (depth - 1)
            self.min_size_frac = None
        else:
            self.min_size = min_size
            self.min_size_frac = min_size_frac if min_size is None else None

        self.trees_ = {}
        self.overall_ndfu_ = {}
        self.retained_ids_ = []

    def _min_size_for(self, n):
        if callable(self.min_size):
            return self.min_size  # pass the callable straight through
        return max(2, round(self.min_size_frac * n)) if self.min_size_frac else self.min_size

    def filter_polarized_texts(self, dataset):
        """Steps 1-2: compute nDFU per text, keep only those >= theta_filter."""
        self.overall_ndfu_ = {tid: ndfu_score(g["rating"].to_numpy(), self.scale)
                               for tid, g in dataset.groupby("text_id")}
        self.retained_ids_ = [t for t, nd in self.overall_ndfu_.items() if nd >= self.theta_filter]
        return self.retained_ids_

    def build_all_trees(self, dataset, text_ids=None):
        """Steps 3-5: build a PolarizedTree for every retained text."""
        text_ids = text_ids or self.retained_ids_
        if not text_ids:
            raise RuntimeError("Call filter_polarized_texts(dataset) first, or pass text_ids.")
        self.trees_ = {}
        for tid in text_ids:
            text_data = dataset[dataset["text_id"] == tid]
            min_size = self._min_size_for(len(text_data))
            self.trees_[tid] = PolarizedTree.build(
                text_data, self.dims, min_size, self.h, self.max_depth, self.scale,
                theta_pole=self.theta_pole, theta_stop=self.theta_stop,
                variant=self.variant, beta=self.beta,
                relative_h=self.relative_h, text_id=tid,
            )
        return self.trees_

    def dimension_frequency(self, ground_truth=None):
        """Step 6.1 (F): splitting-dimension frequency by depth."""
        rows = [{"text_id": t, "depth": d, "dim": dim}
                for t, tree in self.trees_.items()
                for d, dim, prg, path, node in tree.internal_nodes()]
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
                for t, tree in self.trees_.items() for leaf in tree.get_leaves()
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
        for t, tree in self.trees_.items():
            for d, dim, prg, path, node in tree.internal_nodes():
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
        for t, tree in self.trees_.items():
            n_leaves.append(tree.n_leaves)
            for leaf in tree.get_leaves():
                residual.append(leaf["ndfu"])
                depths.append(len(leaf["path"]))
                indet.append(leaf["pole"] == "indeterminate")
            if not tree.get_root()["is_leaf"]:
                top_prgs.append(tree.get_root()["prg"])
            used |= {dim for d, dim, prg, path, node in tree.internal_nodes()}
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
        for t, tree in self.trees_.items():
            true_d = set(ground_truth[t]["active_dims"])
            found_d = {d for leaf in tree.get_leaves() for d, v in leaf["path"]}
            precision = len(true_d & found_d) / len(found_d) if found_d else float(not true_d)
            recall = len(true_d & found_d) / len(true_d) if true_d else float(not found_d)
            rows.append({"text_id": t, "k_true": len(true_d), "true_dims": sorted(true_d),
                         "found_dims": sorted(found_d), "jaccard": jaccard(true_d, found_d),
                         "precision": precision, "recall": recall,
                         "exact_match": sorted(true_d) == sorted(found_d)})
        return pd.DataFrame(rows)

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
