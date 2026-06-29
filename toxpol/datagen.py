"""
Synthetic annotation dataset generator for studying demographic polarization in human labeling.

Builds a pool of annotators with explicit demographic identities and generates annotation
datasets where each text is independently assigned a severity tier (high, moderate, or low
polarization). Within the High and Moderate tiers, every value of every dimension receives a
random weight specific to that text; an annotator's combined weight (the geometric mean of
their weights across all dimensions) determines whether they fall on the toxic or civil side
of that text's median threshold. This produces genuine intersectional structure: ratings
depend on an annotator's full demographic profile, not a single dimension, and different
texts can be explained by different combinations of dimensions at different depths.
"""

import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd


# Default demographic dimensions used in the paper
DEFAULT_DIMENSIONS = {
    "gender": ["male", "female", "non-binary"],
    "politics": ["left", "center", "right"],
    "age": ["<25", "25-50", ">50"],
    "education": ["low", "medium", "high"],
    "orientation": ["heterosexual", "lgbtq+"],
}


class AnnotatorPool:
    """
    Synthetic annotator pool for generating polarized rating datasets.

    Builds a Cartesian-product pool of demographic identities, then generates
    annotation datasets where each text is independently assigned a severity
    tier:

    - **High**: full-strength weight-based bias config, threshold at the
      median, wide non-overlapping toxic/civil ranges. Strong bimodal split.
    - **Moderate**: same weight mechanism, narrower overlapping ranges,
      producing a softer signal.
    - **Low**: the negative-control tier. A configurable fraction of Low
      texts use no demographic split at all (single random peak plus
      spread); the remainder use the weight mechanism with heavily
      overlapping ranges. Both sub-cases land near zero polarization.

    Parameters
    ----------
    dimensions : dict[str, list[str]]
        Mapping from dimension name to the list of possible values.
        Example: {"politics": ["left", "center", "right"], "age": ["<25", ">25"]}

    exclude : list[str] | None
        Dimension names to drop before building identities. Useful for ablations.

    annotators_per_identity : int
        How many annotators share each unique demographic combination.
        Pool size = product(len(v) for v in dimensions.values()) * annotators_per_identity.

    scale : int
        Maximum value on the rating scale (ratings are integers in [1, scale]).

    toxic_range : tuple[int, int]
        (low, high) inclusive range from which High-tier toxic-pole annotators draw ratings.

    civil_range : tuple[int, int]
        (low, high) inclusive range from which High-tier civil-pole annotators draw ratings.

    neutral_range : tuple[int, int]
        Reserved for future use; not currently sampled from directly.

    Examples
    --------
    >>> from toxpol.datagen import AnnotatorPool, DEFAULT_DIMENSIONS
    >>> pool = AnnotatorPool(DEFAULT_DIMENSIONS)
    >>> dataset, bias_configs = pool.generate_dataset(n_texts=50, n_annotators_per_text=100)
    >>> dataset.head()
    """

    def __init__(
        self,
        dimensions,
        exclude=None,
        annotators_per_identity=10,
        scale=5,
        toxic_range=(4, 5),
        civil_range=(1, 2),
        neutral_range=(3, 3),
    ):
        self.annotators_per_identity = annotators_per_identity
        self.identities, self.active_dims = self._get_identities(dimensions, exclude)
        self.pool = self._build_pool()

        self.scale = scale
        self.toxic_range = toxic_range
        self.civil_range = civil_range
        self.neutral_range = neutral_range

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _get_identities(self, dimensions, exclude=None):
        active_dims = {k: v for k, v in dimensions.items() if k not in (exclude or [])}
        identities = [
            dict(zip(active_dims.keys(), combo))
            for combo in itertools.product(*active_dims.values())
        ]
        return identities, active_dims

    def _build_pool(self):
        pool = []
        for identity in self.identities:
            for _ in range(self.annotators_per_identity):
                pool.append(identity.copy())
        pool = pd.DataFrame(pool)
        pool.index.name = "annotator_id"
        return pool

    # ------------------------------------------------------------------
    # Bias mechanics (High / Moderate / Low-weighted tiers)
    # ------------------------------------------------------------------

    def _generate_bias_config(self):
        """
        Assign a random weight to each dimension value for a single text.

        Higher weight pushes an annotator toward the toxic pole, lower
        weight toward civil, relative to the rest of the pool. Weights
        are independent across dimensions and regenerated fresh for
        every text.
        """
        config = {}
        for dim, values in self.active_dims.items():
            config[dim] = {value: random.uniform(0.5, 2.0) for value in values}
        return config

    def _get_combined_weight(self, annotator, bias_config):
        """Geometric mean of weights across all dimensions for one annotator."""
        score = 1.0
        for dim in self.active_dims:
            score *= bias_config[dim][annotator[dim]]
        return score ** (1 / len(self.active_dims))

    def _annotate(self, annotator, bias_config, threshold, toxic_range, civil_range, noise=0.05):
        """
        Rate toxic if the annotator's combined weight exceeds the text's
        threshold, civil otherwise. Global noise adds random deviations.
        """
        score = self._get_combined_weight(annotator, bias_config)
        rating_range = toxic_range if score > threshold else civil_range

        if random.random() < noise:
            return random.randint(1, self.scale)

        return random.randint(*rating_range)

    def _annotate_low_unimodal(self, peak, spread, noise=0.05):
        """
        Single shared peak anywhere on the scale, with natural spread.
        No demographic split: every annotator draws from the same
        distribution regardless of identity. Used for the unimodal
        sub-case of the Low severity tier.
        """
        if random.random() < noise:
            return random.randint(1, self.scale)
        rating = int(round(np.random.normal(loc=peak, scale=spread)))
        return int(np.clip(rating, 1, self.scale))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_texts=100,
        n_annotators_per_text=100,
        noise=0.05,
        high_ratio=0.60,
        moderate_ratio=0.20,
        low_ratio=0.20,
        low_unimodal_share=0.40,
    ):
        """
        Generate a synthetic annotation dataset with three polarization
        severity tiers, mixed according to the given ratios. Each text is
        an independent draw: its tier, and (for High/Moderate/Low-weighted
        texts) its bias config, are generated fresh per text.

        Parameters
        ----------
        n_texts : int
            Number of texts to generate. Must be >= 1.

        n_annotators_per_text : int
            Annotators sampled per text (without replacement).
            Must be <= pool size (annotators_per_identity * number_of_identities).

        noise : float in [0, 1]
            Probability that any given annotator's rating is instead drawn
            uniformly at random from the full scale.

        high_ratio, moderate_ratio, low_ratio : float
            Mixing ratios for the three severity tiers. Must sum to 1.0.

        low_unimodal_share : float in [0, 1]
            Within the Low tier, the fraction of texts that use the true
            unimodal (no demographic split) sub-case rather than the
            heavily-overlapped weighted sub-case.

        Returns
        -------
        dataset : pd.DataFrame
            One row per (text_id, annotator_id) pair. Columns:
            text_id, annotator_id, <all active dimension columns>, rating.

        bias_configs : dict[int, dict]
            Ground truth per text_id: its severity tier, sub-case (where
            applicable), bias config (or None for true unimodal), and
            threshold (where applicable).
        """
        if n_annotators_per_text > len(self.pool):
            raise ValueError(
                f"n_annotators_per_text ({n_annotators_per_text}) exceeds pool size "
                f"({len(self.pool)}). Reduce n_annotators_per_text or increase "
                f"annotators_per_identity."
            )
        assert abs(high_ratio + moderate_ratio + low_ratio - 1.0) < 1e-6, \
            "high_ratio + moderate_ratio + low_ratio must sum to 1.0"

        records = []
        bias_configs = {}

        for text_id in range(n_texts):
            tier = random.choices(
                ["high", "moderate", "low"],
                weights=[high_ratio, moderate_ratio, low_ratio]
            )[0]

            sampled = self.pool.sample(n=n_annotators_per_text, replace=False)

            # Low tier splits further into a true-unimodal sub-case and a
            # heavily-overlapping weighted sub-case, both landing near zero
            if tier == "low" and random.random() < low_unimodal_share:
                peak = random.randint(1, self.scale)
                spread = random.uniform(0.6, 1.2)
                bias_configs[text_id] = {
                    "tier": tier, "subcase": "unimodal", "config": None,
                    "peak": peak, "spread": spread
                }
                for annotator_id, annotator in sampled.iterrows():
                    rating = self._annotate_low_unimodal(peak, spread, noise)
                    records.append({
                        "text_id": text_id, "annotator_id": annotator_id,
                        **annotator.to_dict(), "rating": rating
                    })
                continue

            # High / Moderate, and the weighted sub-case of Low: generate
            # a weight-based bias config and tier-specific pole ranges
            bias_config = self._generate_bias_config()
            scores = [
                self._get_combined_weight(annotator, bias_config)
                for _, annotator in self.pool.iterrows()
            ]
            threshold = np.median(scores)

            if tier == "high":
                toxic_range, civil_range = self.toxic_range, self.civil_range
                subcase = None
            elif tier == "moderate":
                toxic_range, civil_range = (4, 5), (1, 3)
                subcase = None
            else:  # low, weighted sub-case
                toxic_range, civil_range = (3, 5), (1, 3)
                subcase = "weighted"

            bias_configs[text_id] = {
                "tier": tier, "subcase": subcase, "config": bias_config, "threshold": threshold
            }

            for annotator_id, annotator in sampled.iterrows():
                rating = self._annotate(annotator, bias_config, threshold, toxic_range, civil_range, noise)
                records.append({
                    "text_id": text_id, "annotator_id": annotator_id,
                    **annotator.to_dict(), "rating": rating
                })

        dataset = pd.DataFrame(records)
        return dataset, bias_configs

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    @property
    def pool_size(self):
        """Total number of annotators in the pool."""
        return len(self.pool)

    @property
    def n_identities(self):
        """Number of unique demographic identity combinations."""
        return len(self.identities)

    def summary(self):
        """Print a brief summary of the pool configuration."""
        print(f"Active dimensions : {list(self.active_dims.keys())}")
        print(f"Unique identities : {self.n_identities}")
        print(f"Annotators/identity: {self.annotators_per_identity}")
        print(f"Pool size          : {self.pool_size}")
        print(f"Rating scale       : 1-{self.scale}")
        print(f"  toxic_range      : {self.toxic_range}")
        print(f"  civil_range      : {self.civil_range}")
        print(f"  neutral_range    : {self.neutral_range}")

    def describe_bias(self, bias_configs, text_id=0):
        """Pretty-print one text's bias config (tier, sub-case, weights) as a readable table."""
        cfg = bias_configs[text_id]
        print(f"Text {text_id} -- tier: {cfg['tier']}" + (f" ({cfg['subcase']})" if cfg.get('subcase') else ""))
        if cfg["config"] is None:
            print(f"  unimodal peak={cfg['peak']}  spread={cfg['spread']:.2f}")
            return
        print(f"  threshold (median combined weight): {cfg['threshold']:.3f}")
        for dim, weights in cfg["config"].items():
            weight_str = "  ".join(f"{v}={w:.2f}" for v, w in weights.items())
            print(f"  {dim:<12} {weight_str}")

    def analyze(self, dataset, bias_configs):
        """
        Compute nDFU scores for every text, overall and per dimension value.

        Requires the `ndfu` package (`pip install toxpol-nlp[ndfu]`).

        Returns
        -------
        dict
            results[text_id]["overall"] -> float
            results[text_id][dim][value] -> float
        """
        try:
            from ndfu import dfu
        except ImportError:
            raise ImportError(
                "ndfu is required for analyze(). "
                "Install it with: pip install toxpol-nlp[ndfu]"
            )

        def _ndfu(ratings):
            counts = np.bincount(ratings, minlength=self.scale + 1)[1:]
            hist = counts / counts.sum()
            return dfu(hist)

        results = {}
        for text_id, text_data in dataset.groupby("text_id"):
            text_results = {"overall": _ndfu(text_data["rating"].values)}
            for dim in self.active_dims:
                text_results[dim] = {
                    value: _ndfu(group["rating"].values)
                    for value, group in text_data.groupby(dim)
                }
            results[text_id] = text_results
        return results

    def summarize(self, dataset, bias_configs, text_id=0):
        """
        Print nDFU scores for one text, grouped by dimension, with its
        severity tier shown. Calls analyze() internally. Requires `ndfu`.
        """
        results = self.analyze(dataset, bias_configs)
        text = results[text_id]
        tier = bias_configs[text_id]["tier"]
        print(f"Text {text_id} (tier: {tier}) -- overall nDFU: {text['overall']:.3f}\n")
        for dim, values in text.items():
            if dim == "overall":
                continue
            print(f"{dim}:")
            for value, score in values.items():
                print(f"  {value}: {score:.3f}")
            print()

    def summarize_all(self, dataset, bias_configs):
        """
        Print mean nDFU per severity tier and per dimension value,
        aggregated across all texts. Calls analyze() internally.
        Requires `ndfu`.
        """
        results = self.analyze(dataset, bias_configs)
        n_texts = len(results)

        overall_scores = [results[t]["overall"] for t in results]
        print(f"Overall nDFU -- mean: {np.mean(overall_scores):.3f}  "
              f"median: {np.median(overall_scores):.3f}  "
              f"(across {n_texts} texts)\n")

        tier_counts = Counter(bias_configs[t]["tier"] for t in bias_configs)
        print(f"Tier counts: {dict(tier_counts)}\n")

        for tier in ("high", "moderate", "low"):
            scores = [results[t]["overall"] for t in results if bias_configs[t]["tier"] == tier]
            if scores:
                print(f"{tier:<10} mean: {np.mean(scores):.3f}  "
                      f"min: {np.min(scores):.3f}  max: {np.max(scores):.3f}  (n={len(scores)})")
        print()

        for dim in self.active_dims:
            print(f"\n{dim}:")
            for value in self.active_dims[dim]:
                scores = [results[t][dim][value] for t in results if value in results[t][dim]]
                if scores:
                    print(f"  {value:<15} mean: {np.mean(scores):.3f}  "
                          f"min: {np.min(scores):.3f}  max: {np.max(scores):.3f}")