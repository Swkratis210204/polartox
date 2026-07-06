# polartox

![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)
[![PyPI](https://img.shields.io/badge/PyPI-0.2.0-blue)](https://pypi.org/project/polartox/)

NLP toolkit for **annotator polarization research**. Provides tools for synthetic dataset generation and polarization detection in human annotation studies.

## Install

```bash
pip install polartox

# with nDFU support (Pavlopoulos & Likas, 2024 -- github.com/ipavlopoulos/ndfu)
pip install "polartox[ndfu]"
```

## Repository Structure

```
data_gen/          synthetic annotation dataset generator
polarized_trees/   Polarized Trees detection algorithm
polartox/          installable package (source code)
```

### `data_gen/`
Tools for generating synthetic annotation datasets with **injected, known polarization**. Real annotation data cannot provide ground truth for which demographic dimensions drive disagreement — this module does. The generated datasets are the primary validation input for the Polarized Trees algorithm.

→ See [`data_gen/README.md`](data_gen/README.md) for the full API and usage.

### `polarized_trees/`
The Polarized Trees detection algorithm. Given an annotation dataset, recursively partitions annotators by demographic dimension to find the subgroups most polarized on a given text (paper Steps 1–6): which dimensions drive disagreement, and which intersectional subgroups diverge most strongly.

→ See [`polarized_trees/README.md`](polarized_trees/README.md) for the full API, usage, and a comparison of three empirically-validated departures from the paper's literal specification.

## Tools

| Module | Description | Status |
|---|---|---|
| `polartox.datagen` | Synthetic annotator pool with injected, ground-truth polarization | Stable |
| `polartox.polarized_trees` | Polarized Trees detection algorithm | Stable (baseline) |

nDFU scoring is provided by the collaborative [`ndfu`](https://github.com/ipavlopoulos/ndfu) package (Pavlopoulos & Likas, 2024) rather than reimplemented here — install via the `[ndfu]` extra above.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.