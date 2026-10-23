"""Runtime of Polarized Trees with the selected configuration, on one CPU core.

    python scripts/runtime_benchmark.py [--dices-dir DIR] [--reps N] [--synthetic-reps N]

Prints wall-clock seconds, in four parts:
  1. time to build one tree, per formulation, on the polarized texts of DICES-350 and DICES-990;
  2. total time to build all those trees, per formulation and for all five;
  3. one synthetic corpus (100 texts, 1,620 annotators per text), per formulation;
  4. growth with the number of annotators per text (50, 100, 500, 1,620).

One warm-up pass is discarded, then --reps (DICES) or --synthetic-reps (synthetic) passes are averaged.
Per-tree time is PolarizedTree.build only; "all trees" is PolarizedTreesPipeline.build_all_trees.
DICES_350.csv and DICES_990.csv (the original files, rater_id not needed) are read from --dices-dir.
Nothing is written to disk.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"                                   # before numpy is imported

import argparse
import platform
import statistics as st
import time
from pathlib import Path

import numpy as np
import pandas as pd

from polartox.datagen import DEFAULT_DIMENSIONS, AnnotatorPool
from polartox.pipeline import PolarizedTreesPipeline
from polartox.polarized_tree import PEG_VARIANTS, PolarizedTree

ROOT = Path(__file__).resolve().parent.parent

# The selected configuration (everything except the PEG formulation)
CONFIG = dict(theta_filter=0.30, min_size_frac=0.02, max_depth=4, h=0.20, relative_h=True, theta_stop=0.10)

# DICES preprocessing, as in the DICES notebook
DICES_DIMS = ["gender", "race", "age", "education"]
EXCLUDED = {"rater_gender": ["Nonbinary", "Self-describe (below)"], "rater_race": ["Other", "Multiracial"], "rater_education": ["Other"]}
RENAMED = {
    "rater_race": {"Asian/Asian subcontinent": "Asian", "Black/African American": "Black",
                   "LatinX, Latino, Hispanic or Spanish Origin": "Hispanic/Latino"},
    "rater_age": {"gen z": "Gen Z", "millenial": "Millennial", "gen x+": "Gen X+"},
    "rater_education": {"College degree or higher": "Degree+", "High school or below": "High School or Below"},
}
RATING = {"No": 1, "Unsure": 2, "Yes": 3}
DICES_SCALE, DICES_THETA_POLE = 3, 3

# Synthetic corpus A (benchmark_config.py)
SYN_DIMS = list(DEFAULT_DIMENSIONS)
SYN_SCALE, SYN_DEPTH_WEIGHTS = 5, {0: 0.15, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.10}


def load_dices(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Q3_unfair_bias_overall": "Q3_bias_overall"})   # the original DICES-990 file's name
    df = df[["item_id", "rater_gender", "rater_race", "rater_age", "rater_education", "Q3_bias_overall"]]
    for col, cats in EXCLUDED.items():
        df = df[~df[col].isin(cats)]
    for col, mapping in RENAMED.items():
        df[col] = df[col].replace(mapping)
    out = df.rename(columns={"item_id": "text_id", "rater_gender": "gender", "rater_race": "race",
                             "rater_age": "age", "rater_education": "education"})
    out["rating"] = out["Q3_bias_overall"].map(RATING)
    out = out.reset_index(drop=True)
    out["annotator_id"] = np.arange(len(out))
    return out[["text_id", "annotator_id"] + DICES_DIMS + ["rating"]]


def pipeline(variant, dims, scale, theta_pole=None):
    return PolarizedTreesPipeline(dims=dims, scale=scale, variant=variant, theta_pole=theta_pole, **CONFIG)


def time_trees(pipe, data, ids, reps):
    """({text: [seconds of PolarizedTree.build per repeat]}, [seconds of build_all_trees per repeat])"""
    groups = {t: data[data["text_id"] == t] for t in ids}
    per = {t: [] for t in ids}
    total = []
    for rep in range(reps + 1):                            # repeat 0 is the warm-up
        for t in ids:
            g = groups[t]
            t0 = time.perf_counter()
            PolarizedTree.build(g, pipe.dims, pipe._min_size_for(len(g)), pipe.h, pipe.max_depth, pipe.scale,
                                theta_pole=pipe.theta_pole, theta_stop=pipe.theta_stop, variant=pipe.variant,
                                beta=pipe.beta, relative_h=pipe.relative_h, text_id=t)
            if rep:
                per[t].append(time.perf_counter() - t0)
    for rep in range(reps + 1):
        t0 = time.perf_counter()
        pipe.build_all_trees(data, ids)
        if rep:
            total.append(time.perf_counter() - t0)
    return per, total


def summary(per):
    means = [st.mean(v) for v in per.values()]
    return st.mean(means), st.median(means), max(means), max(x for v in per.values() for x in v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dices-dir", type=Path, default=ROOT.parent, help="folder with DICES_350.csv and DICES_990.csv")
    ap.add_argument("--reps", type=int, default=5, help="timed repeats on DICES")
    ap.add_argument("--synthetic-reps", type=int, default=2, help="timed repeats on the synthetic corpora")
    args = ap.parse_args()

    print(f"{platform.processor()} | {platform.platform()} | python {platform.python_version()} | "
          f"numpy {np.__version__} | pandas {pd.__version__} | logical cpus {os.cpu_count()}")
    try:                                                   # one core, if psutil is available
        import psutil
        psutil.Process().cpu_affinity([2 if (os.cpu_count() or 1) > 2 else 0])
        print("pinned to cpu", psutil.Process().cpu_affinity())
    except Exception as exc:
        print("not pinned to one core (", type(exc).__name__, "): BLAS threads are limited to 1, the code is single-threaded")
    print(f"configuration: {CONFIG}\n")

    print(f"== 1/2. DICES polarized texts, {args.reps} timed repeats after one warm-up (per-tree times in ms)")
    totals = {}
    for name in ("DICES-350", "DICES-990"):
        data = load_dices(args.dices_dir / f"DICES_{name[-3:]}.csv")
        for v in PEG_VARIANTS:
            pipe = pipeline(v, DICES_DIMS, DICES_SCALE, DICES_THETA_POLE)
            ids = pipe.filter_polarized_texts(data)
            per, total = time_trees(pipe, data, ids, args.reps)
            mean, median, mx, slowest = summary(per)
            totals[name, v] = st.mean(total)
            print(f"{name:10s} {v:9s} texts {len(ids):4d} | per tree ms: mean {mean*1e3:6.2f} median {median*1e3:6.2f} "
                  f"max {mx*1e3:6.2f} slowest single run {slowest*1e3:7.2f} | all trees s: {totals[name, v]:.3f}")
    print("\ntotal s for all trees (350 / 990 / both):")
    for v in PEG_VARIANTS:
        print(f"  {v:9s} {totals['DICES-350', v]:.3f} / {totals['DICES-990', v]:.3f} / {totals['DICES-350', v] + totals['DICES-990', v]:.3f}")
    t350 = sum(totals["DICES-350", v] for v in PEG_VARIANTS)
    t990 = sum(totals["DICES-990", v] for v in PEG_VARIANTS)
    print(f"  all five  {t350:.3f} / {t990:.3f} / {t350 + t990:.3f}")

    pool = AnnotatorPool(DEFAULT_DIMENSIONS, SYN_SCALE, (0.3, 1.0), SYN_DEPTH_WEIGHTS, 10)
    print(f"\n== 3. synthetic corpus, 100 texts x 1,620 annotators, all 100 texts built, {args.synthetic_reps} timed repeats")
    corpus = pool.generate_dataset(100, n_annotators_per_text=None, noise=0.05, seed=1)
    data = corpus.data if hasattr(corpus, "data") else corpus
    ids = sorted(data["text_id"].unique())
    for v in PEG_VARIANTS:
        per, total = time_trees(pipeline(v, SYN_DIMS, SYN_SCALE), data, ids, args.synthetic_reps)
        mean, median, mx, _ = summary(per)
        print(f"{v:9s} per tree s: mean {mean:.4f} median {median:.4f} max {mx:.4f} | 100 trees s: {st.mean(total):.2f}")

    print("\n== 4. growth with the number of annotators per text (100 texts each, all built)")
    for n in (50, 100, 500, 1620):
        gen = pool.generate_dataset(100, n_annotators_per_text=n, noise=0.05, seed=1)
        data = gen.data if hasattr(gen, "data") else gen
        ids = sorted(data["text_id"].unique())
        cells = []
        for v in ("min", "harmonic", "max"):
            per, total = time_trees(pipeline(v, SYN_DIMS, SYN_SCALE), data, ids, args.synthetic_reps)
            cells.append(f"{v}: {summary(per)[0]*1e3:.2f} ms/tree, {st.mean(total):.2f} s/100 trees")
        print(f"annotators {n:5d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
