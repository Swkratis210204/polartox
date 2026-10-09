# Top-20 configurations: which PEG formulations?

**All 20 of the top-ranked configurations use `harmonic` (11) or `weighted` (9).**
None of them uses `max`, `min` or `mean`.

| variant | in top 20 | best rank | configs evaluated | best Jaccard | mean Jaccard | median Jaccard |
|---|---|---|---|---|---|---|
| harmonic | 11 | 1 | 161 | 0.8924 | 0.7727 | 0.8234 |
| weighted | 9 | 9 | 175 | 0.8794 | 0.7256 | 0.8037 |
| max | 0 | - | 155 | 0.8503 | 0.7950 | 0.7991 |
| mean | 0 | - | 142 | 0.8429 | 0.8041 | 0.8098 |
| min | 0 | - | 167 | 0.8287 | 0.7905 | 0.7971 |

The same numbers are in `top20_formulations.csv`; the 20 configurations
themselves, with all their metrics, are in `top_configurations.csv`
(row position 0 to 19 = rank 1 to 20).

## Other things true of the top 20

- `relative_h=True` and `theta_stop=0.1` in every one of the 20.
- `harmonic` rows use `h` of 0.15 or 0.20; `weighted` rows use `h` of 0.10 or 0.15.
- Ranks 1, 2 and 3 have identical scores (Jaccard 0.8924, precision 0.9521,
  recall 0.9280) and differ only in `min_size_frac` and `max_depth`, which
  rarely change recovery on this synthetic data. The order among tied rows is
  arbitrary. The top 20 holds only 11 distinct Jaccard values.
- 19 of the 20 rows are distinct settings of the non-PEG hyperparameters
  (`theta_filter`, `min_size_frac`, `max_depth`, `h`, `relative_h`,
  `theta_stop`).

## How these results were produced

- Random search: 800 of the 3,240 configurations (seed 0), selected by mean
  Jaccard over the three benchmark corpora A, B and C; runtime 314 s.
- polartox 0.7.0 (editable install of this repository), numpy 2.5.3,
  pandas 3.0.6, ndfu 0.9.3, scikit-learn 1.9.1.
- Run date: 2026-10-02.
- Because only a quarter of the grid was sampled, each non-PEG setting was
  evaluated with only some of the five formulations. This ranking therefore
  says which configurations scored best in the sample, not which formulation is
  best at a fixed setting; that comparison is the next step.
