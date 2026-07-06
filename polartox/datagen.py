"""
polartox.datagen -- synthetic annotation dataset generator for the
Polarized Trees framework.

Mechanism: per text, a random subset of k demographic dimensions is chosen
to drive polarization (k=0 is the unimodal negative control). For each
active dimension, values are split into toxic-leaning / civil-leaning
groups, and each dimension's pull is governed by a single continuous
intensity parameter alpha, interpolating between the uniform distribution
(no signal, alpha=0) and a fully deterministic one-hot distribution at the
relevant pole (maximal signal, alpha=1). Each identity's final rating
distribution is the elementwise product of its active-dimension shapes --
reinforcing when leans agree, muted when they conflict. Every annotator in
the pool rates every text unless a smaller per-text sample size is given.
"""

import itertools
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Reference configuration (opt-in, not applied implicitly)
# ─────────────────────────────────────────────

DEFAULT_DIMENSIONS = {
    "gender":      ["male", "female", "non-binary"],
    "politics":    ["left", "center", "right"],
    "age":         ["<25", "25-50", ">50"],
    "education":   ["low", "medium", "high"],
    "orientation": ["heterosexual", "lgbtq+"],
}

DEFAULT_DEPTH_WEIGHTS = {0: 0.05, 1: 0.20, 2: 0.30, 3: 0.25, 4: 0.20}
DEFAULT_INTENSITY_RANGE = (0.3, 1.0)


class GeneratedDataset:
    """
    Result of AnnotatorPool.generate_dataset(...).

    Behaves like the (dataset, ground_truth) tuple for backward-compatible
    unpacking:

        dataset, ground_truth = pool.generate_dataset(...)

    ...but also supports direct inspection when kept as one object:

        result = pool.generate_dataset(...)
        result.head()
        result.describe_text(0)
    """

    def __init__(self, data, ground_truth):
        self.data = data
        self.ground_truth = ground_truth

    def __iter__(self):
        return iter((self.data, self.ground_truth))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        n_texts = len(self.ground_truth)
        return f"GeneratedDataset({n_texts} texts, {len(self.data)} rows)\n{self.data.head()!r}"

    def head(self, n=5):
        return self.data.head(n)

    def tail(self, n=5):
        return self.data.tail(n)

    def sample(self, n=5, **kwargs):
        return self.data.sample(n, **kwargs)

    def text_ids_by_k(self, k, show=10):
        """
        Return all text_ids with exactly k active dimensions. Prints a
        short summary (count + first `show` ids) instead of dumping the
        full list, since there are often hundreds of matches.
        """
        ids = [tid for tid, cfg in self.ground_truth.items() if len(cfg["active_dims"]) == k]
        preview = ids[:show]
        suffix = f", ... ({len(ids) - show} more)" if len(ids) > show else ""
        print(f"k={k}: {len(ids)} texts -> {preview}{suffix}")
        return ids

    def describe_text(self, text_id):
        """
        Pretty-print one text's ground truth config (active dims, lean,
        alpha, or peak/spread for k=0) alongside its observed rating counts.
        """
        cfg = self.ground_truth[text_id]
        text_rows = self.data[self.data["text_id"] == text_id]

        print(f"Text {text_id}  ({len(text_rows)} annotators)")
        if not cfg["active_dims"]:
            print("  k = 0 -- negative control, no dimension is active")
            print(f"  shared peak = {cfg['peak']}, spread = {cfg['spread']:.2f}")
        else:
            print(f"  k = {len(cfg['active_dims'])} active dimension(s)")
            print("  alpha = how strongly a dimension pulls toward its pole (0 = no effect, 1 = fully deterministic)")
            for dim in cfg["active_dims"]:
                lean_str = ", ".join(f"{v} -> {l}" for v, l in cfg["lean"][dim].items())
                print(f"    - {dim} (alpha={cfg['alpha'][dim]:.2f}): {lean_str}")

        counts = text_rows["rating"].value_counts().sort_index()
        counts_str = ", ".join(f"{rating}: {count}" for rating, count in counts.items())
        print(f"  rating counts -> {counts_str}")

class AnnotatorPool:
    """
    Synthetic annotator pool for generating multidimensional polarized
    rating datasets, for validating the Polarized Trees framework.

    All structural parameters are mandatory -- there are no silent
    fallbacks. If you want the reference configuration, pass the
    DEFAULT_* constants explicitly, e.g. dimensions=DEFAULT_DIMENSIONS.

    Parameters
    ----------
    dimensions : dict[str, list[str]]
        Mapping from dimension name to its possible values.
    scale : int
        Rating scale, ratings are integers in [1, scale].
    intensity_range : tuple[float, float]
        (alpha_min, alpha_max), each in [0, 1]. For every active dimension
        in a text, alpha is drawn uniformly from this range and controls
        how strongly that dimension pulls annotators toward its assigned
        pole: alpha=0 contributes no signal (identical to the uniform
        distribution), alpha=1 is a fully deterministic pole.
    depth_weights : dict[int, float]
        P(k active dimensions) for k = 0 .. len(dimensions). Must sum to 1.
    annotators_per_identity : int
        How many annotators share each unique demographic combination.
    alpha_window : float
        For texts with 2+ active dimensions, each dimension's alpha is
        drawn within +/- alpha_window of a shared per-text base value,
        rather than fully independently. Reduces (does not eliminate) an
        "absorption" failure mode where a strong dimension's signal buries
        a much weaker co-active one during detection -- see
        polartox.polarized_trees for details.
    """

    def __init__(
        self,
        dimensions,
        scale,
        intensity_range,
        depth_weights,
        annotators_per_identity,
    ):
        self.dimensions = dimensions
        self.dim_names = list(self.dimensions.keys())
        self.scale = scale
        self.ratings = np.arange(1, scale + 1)

        alpha_min, alpha_max = intensity_range
        assert 0 <= alpha_min <= alpha_max <= 1, "intensity_range must be within [0, 1]"
        self.intensity_range = intensity_range

        self.depth_weights = depth_weights
        self.annotators_per_identity = annotators_per_identity

        self.pool = self._build_pool()

        self.pool_codes = {}
        for dim, values in self.dimensions.items():
            value_to_idx = {v: i for i, v in enumerate(values)}
            self.pool_codes[dim] = self.pool[dim].map(value_to_idx).to_numpy()

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_pool(self):
        identities = [
            dict(zip(self.dim_names, combo))
            for combo in itertools.product(*self.dimensions.values())
        ]
        records = []
        aid = 0
        for identity in identities:
            for _ in range(self.annotators_per_identity):
                records.append({"annotator_id": aid, **identity})
                aid += 1
        return pd.DataFrame(records).set_index("annotator_id")

    @property
    def pool_size(self):
        return len(self.pool)

    @property
    def n_identities(self):
        return self.pool_size // self.annotators_per_identity

    # ------------------------------------------------------------------
    # Per-text config: which dims are active, their lean split, their intensity
    # ------------------------------------------------------------------

    def _sample_text_config(self, rng):
        """
        Ground truth for one text.
        active_dims: [] means k=0, the unimodal negative control.
        lean: {dim: {value: 'toxic'/'civil'}}
        alpha: {dim: float in intensity_range}, one independent draw per
               active dimension.
        """
        ks = list(self.depth_weights.keys())
        ps = list(self.depth_weights.values())
        k = rng.choice(ks, p=ps)

        active_dims = list(rng.choice(self.dim_names, size=k, replace=False)) if k > 0 else []

        lean, alpha = {}, {}
        alpha_min, alpha_max = self.intensity_range

        for dim in active_dims:
            values = self.dimensions[dim][:]
            rng.shuffle(values)
            n = len(values)
            split = n // 2 if n % 2 == 0 else rng.choice([n // 2, n // 2 + 1])
            toxic_vals, civil_vals = values[:split], values[split:]
            lean[dim] = {v: "toxic" for v in toxic_vals} | {v: "civil" for v in civil_vals}
            alpha[dim] = float(rng.uniform(alpha_min, alpha_max))

        return {"active_dims": active_dims, "lean": lean, "alpha": alpha}

    def _pole_shape(self, alpha, pole):
        """
        Interpolate between the uniform distribution (alpha=0) and a fully
        deterministic one-hot distribution at the given pole (alpha=1).
        pole='civil' concentrates at rating 1, pole='toxic' at rating=scale.
        """
        uniform = np.ones(self.scale) / self.scale
        extreme = np.zeros(self.scale)
        extreme[0 if pole == "civil" else -1] = 1.0
        return alpha * extreme + (1 - alpha) * uniform

    def _unimodal_negative_control(self, rng):
        peak = int(rng.integers(1, self.scale + 1))
        spread = float(rng.uniform(0.6, 1.2))
        return peak, spread

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(self, n_texts, n_annotators_per_text=None, noise=0.05, seed=0):
        """
        Generate a synthetic annotation dataset.

        n_annotators_per_text : int or None
            How many annotators rate each text, sampled fresh (without
            replacement) per text. If None, or if the requested number
            is >= the pool size, the full pool is used for every text.
        noise : float in [0, 1]
            Probability that any given annotator's rating is instead drawn
            uniformly at random from the full scale.

        Returns
        -------
        GeneratedDataset
            Unpacks as (dataset, ground_truth) for backward compatibility,
            or use directly for result.head(), result.describe_text(id), etc.
        """
        rng = np.random.default_rng(seed)
        scale = self.scale
        pool_size = self.pool_size

        if n_annotators_per_text is None or n_annotators_per_text >= pool_size:
            n_per_text = pool_size
        else:
            n_per_text = n_annotators_per_text

        pool_index = self.pool.index.to_numpy()
        pool_identity_cols = {dim: self.pool[dim].to_numpy() for dim in self.dim_names}

        all_annotator_ids = np.empty(n_texts * n_per_text, dtype=pool_index.dtype)
        all_ratings = np.empty(n_texts * n_per_text, dtype=np.int64)
        all_identity = {dim: np.empty(n_texts * n_per_text, dtype=object) for dim in self.dim_names}
        text_id_col = np.repeat(np.arange(n_texts), n_per_text)

        ground_truth = {}

        for text_id in range(n_texts):
            start = text_id * n_per_text
            end = start + n_per_text

            if n_per_text == pool_size:
                sampled_idx = np.arange(pool_size)
            else:
                sampled_idx = rng.choice(pool_size, size=n_per_text, replace=False)

            annotator_ids = pool_index[sampled_idx]
            for dim in self.dim_names:
                all_identity[dim][start:end] = pool_identity_cols[dim][sampled_idx]
            all_annotator_ids[start:end] = annotator_ids

            config = self._sample_text_config(rng)

            if not config["active_dims"]:
                peak, spread = self._unimodal_negative_control(rng)
                ground_truth[text_id] = {"active_dims": [], "peak": peak, "spread": spread}
                draws = rng.normal(peak, spread, size=n_per_text)
                ratings = np.clip(np.round(draws), 1, scale).astype(np.int64)
            else:
                ground_truth[text_id] = config
                dist = np.ones((n_per_text, scale))
                for dim in config["active_dims"]:
                    values = self.dimensions[dim]
                    alpha = config["alpha"][dim]
                    value_shapes = np.empty((len(values), scale))
                    for i, v in enumerate(values):
                        pole = config["lean"][dim][v]
                        value_shapes[i] = self._pole_shape(alpha, pole)
                    codes = self.pool_codes[dim][sampled_idx]
                    dist *= value_shapes[codes]
                dist /= dist.sum(axis=1, keepdims=True)

                cdf = np.cumsum(dist, axis=1)
                u = rng.random(n_per_text)[:, None]
                ratings = (u < cdf).argmax(axis=1) + 1

            noise_mask = rng.random(n_per_text) < noise
            if noise_mask.any():
                ratings = ratings.copy()
                ratings[noise_mask] = rng.integers(1, scale + 1, size=noise_mask.sum())

            all_ratings[start:end] = ratings

        data = {"text_id": text_id_col, "annotator_id": all_annotator_ids}
        for dim in self.dim_names:
            data[dim] = all_identity[dim]
        data["rating"] = all_ratings

        dataset = pd.DataFrame(data)
        return GeneratedDataset(dataset, ground_truth)

    def summary(self):
        print(f"Dimensions        : {self.dim_names}")
        print(f"Unique identities : {self.n_identities}")
        print(f"Annotators/identity: {self.annotators_per_identity}")
        print(f"Pool size         : {self.pool_size}")
        print(f"Rating scale      : 1-{self.scale}")
        print(f"Intensity range   : {self.intensity_range}")
        print(f"Depth weights     : {self.depth_weights}")


# ─────────────────────────────────────────────
# Example instantiation
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pool = AnnotatorPool(
        dimensions=DEFAULT_DIMENSIONS,
        scale=5,
        intensity_range=DEFAULT_INTENSITY_RANGE,
        depth_weights=DEFAULT_DEPTH_WEIGHTS,
        annotators_per_identity=10,
    )
    pool.summary()

    result = pool.generate_dataset(n_texts=5000, n_annotators_per_text=None, noise=0.05, seed=42)
    print(f"\nDataset shape: {result.data.shape}")
    print(result.head())
    result.describe_text(0)