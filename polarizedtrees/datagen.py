"""
Synthetic annotation dataset generator for studying demographic polarization in human labeling.

Builds a pool of annotators with explicit demographic identities and generates structured
disagreement patterns where rating behavior is governed by per-dimension bias configurations.
"""

import itertools
import random

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
    annotation datasets where each dimension is randomly assigned either a
    "polarizing" role (splitting annotators into toxic/civil poles) or a
    "unimodal" role (converging all annotators toward one rating range).

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
        (low, high) inclusive range from which toxic-pole annotators draw ratings.

    civil_range : tuple[int, int]
        (low, high) inclusive range from which civil-pole annotators draw ratings.

    neutral_range : tuple[int, int]
        (low, high) inclusive range used when a unimodal dimension converges to "neutral".

    Examples
    --------
    >>> from polarizedtrees.datagen import AnnotatorPool, DEFAULT_DIMENSIONS
    >>> pool = AnnotatorPool(DEFAULT_DIMENSIONS)
    >>> dataset, bias_config = pool.generate_dataset(n_texts=50, n_annotators_per_text=100)
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
    # Bias configuration
    # ------------------------------------------------------------------

    def _generate_bias_config(self, polarizing_prob=0.7):
        """
        Randomly assign each active dimension a role for one dataset instance.

        A "polarizing" dimension splits its values into a toxic pole and a civil
        pole. An "unimodal" dimension converges all annotators toward one range.

        Returns
        -------
        dict
            Keys are dimension names. Each value is a dict with:
            - role: "polarizing" | "unimodal"
            - toxic_pole / civil_pole (if polarizing): lists of dimension values
            - convergence (if unimodal): "toxic" | "civil" | "neutral"
        """
        config = {}
        for dim, values in self.active_dims.items():
            role = random.choices(
                ["polarizing", "unimodal"],
                weights=[polarizing_prob, 1 - polarizing_prob],
            )[0]
            if role == "polarizing":
                shuffled = values.copy()
                random.shuffle(shuffled)
                split = random.randint(1, len(shuffled) - 1)
                config[dim] = {
                    "role": "polarizing",
                    "toxic_pole": shuffled[:split],
                    "civil_pole": shuffled[split:],
                }
            else:
                config[dim] = {
                    "role": "unimodal",
                    "convergence": random.choice(["toxic", "civil", "neutral"]),
                }
        return config

    # ------------------------------------------------------------------
    # Per-annotator rating
    # ------------------------------------------------------------------

    def _annotate(self, annotator, bias_config, noise=0.1):
        """
        Produce one rating for a single annotator given the active bias config.

        Each polarizing dimension casts a vote (toxic or civil) based on which
        pole the annotator's value falls in. The majority vote determines the
        rating range. Unimodal dimensions serve as a fallback when no polarizing
        dimension applies. With probability `noise`, a fully random rating is
        returned instead, simulating genuine outlier disagreement.
        """
        votes = []
        for dim, config in bias_config.items():
            if config["role"] == "polarizing":
                if annotator[dim] in config["toxic_pole"]:
                    votes.append("toxic")
                elif annotator[dim] in config["civil_pole"]:
                    votes.append("civil")

        if not votes:
            for dim, config in bias_config.items():
                if config["role"] == "unimodal":
                    votes.append(config["convergence"])

        toxic_votes = votes.count("toxic")
        civil_votes = votes.count("civil")
        neutral_votes = votes.count("neutral")

        if toxic_votes > civil_votes and toxic_votes > neutral_votes:
            rating_range = self.toxic_range
        elif civil_votes > toxic_votes and civil_votes > neutral_votes:
            rating_range = self.civil_range
        else:
            rating_range = self.neutral_range

        if random.random() < noise:
            return random.randint(1, self.scale)

        return random.randint(*rating_range)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_texts=100,
        n_annotators_per_text=100,
        noise=0.1,
        polarizing_prob=0.7,
    ):
        """
        Generate a synthetic annotation dataset.

        A single bias configuration is drawn for the entire dataset (all texts
        share the same demographic polarization structure). Each text is then
        annotated by a random subset of the pool.

        Parameters
        ----------
        n_texts : int
            Number of texts to annotate. Must be >= 1.

        n_annotators_per_text : int
            Annotators sampled per text (without replacement).
            Must be <= pool size (annotators_per_identity * number_of_identities).

        noise : float in [0, 1]
            Probability that any annotator ignores the bias config and draws
            a uniformly random rating instead.

        polarizing_prob : float in [0, 1]
            Prior probability that each dimension is assigned a "polarizing"
            role in the bias config (vs. "unimodal").

        Returns
        -------
        dataset : pd.DataFrame
            One row per (text_id, annotator_id) pair. Columns:
            text_id, annotator_id, <all active dimension columns>, rating.

        bias_config : dict
            The bias configuration used for this dataset. See
            `_generate_bias_config` for the structure.
        """
        if n_annotators_per_text > len(self.pool):
            raise ValueError(
                f"n_annotators_per_text ({n_annotators_per_text}) exceeds pool size "
                f"({len(self.pool)}). Reduce n_annotators_per_text or increase "
                f"annotators_per_identity."
            )

        bias_config = self._generate_bias_config(polarizing_prob)

        records = []
        for text_id in range(n_texts):
            sampled = self.pool.sample(n=n_annotators_per_text, replace=False)
            for annotator_id, annotator in sampled.iterrows():
                rating = self._annotate(annotator, bias_config, noise)
                records.append(
                    {
                        "text_id": text_id,
                        "annotator_id": annotator_id,
                        **annotator.to_dict(),
                        "rating": rating,
                    }
                )

        return pd.DataFrame(records), bias_config

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
        print(f"Rating scale       : 1–{self.scale}")
        print(f"  toxic_range      : {self.toxic_range}")
        print(f"  civil_range      : {self.civil_range}")
        print(f"  neutral_range    : {self.neutral_range}")
