# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — Unreleased

### Changed
- Full rewrite and rename from `polartox`. This is a new baseline, not a
  continuation of `polartox`'s version history.
- Synthetic data generation mechanism replaced entirely: severity tiers
  (High/Moderate/Low) and geometric-mean-of-weights + median threshold are
  replaced by a k-active-dimensions design, where each active dimension's
  pull is governed by a single continuous intensity parameter (`alpha`),
  and identities' rating distributions are the elementwise product of their
  active-dimension shapes.
- `bias_configs` renamed to `ground_truth`; structure changed accordingly
  (see `data_gen/README.md` migration notes).
- `generate_dataset` now returns a `GeneratedDataset` object (still unpacks
  as `(dataset, ground_truth)` for backward compatibility), adding
  `.head()`, `.tail()`, `.sample()`, `.describe_text()`, and
  `.text_ids_by_k()`.
- nDFU is no longer reimplemented in this package. `polartox` depends on
  the collaborative [`ndfu`](https://github.com/ipavlopoulos/ndfu) package
  (Pavlopoulos & Likas, 2024) directly, installable via the `[ndfu]` extra.

### Removed
- `pool.describe_bias`, `pool.analyze`, `pool.summarize`, `pool.summarize_all`.
- `toxic_range` / `civil_range` / `neutral_range` / `exclude` /
  `high_ratio` / `moderate_ratio` / `low_ratio` / `low_unimodal_share`
  parameters.

### Why
The previous weight-averaging mechanism capped achievable nDFU well below 1
— a Central Limit Theorem artifact of averaging many annotators sampled
from a large pool. The new multiplicative mechanism composes signal
instead of diluting it, reaching the full nDFU range.

---

## Prior history (`polartox`, deprecated)
See the [`polartox` PyPI page](https://pypi.org/project/polartox/) for
changelog entries prior to this rewrite.