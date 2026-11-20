"""Figure `dices_retention.png` of the paper: the distribution of the text-level nDFU in
DICES-350 and DICES-990, one panel on top of the other (one column wide).

Uses the same preparation as DICES_polarized_trees_end_to_end.ipynb (sections 2-4 and 6):
drop the excluded SCD categories, map No/Unsure/Yes to 1/2/3, nDFU per item.

Run:  python Dices/make_dices_retention_figure.py [DIR_WITH_DICES_CSVS]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ndfu import dfu

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../Aposteriori/datasets/dices")
OUT = Path(__file__).resolve().parents[1] / "paper" / "src" / "images" / "dices_retention.png"
THETA = 0.30
RATING_MAP = {"No": 1, "Unsure": 2, "Yes": 3}
EXCLUDED = {"rater_gender": ["Nonbinary", "Self-describe (below)"],
            "rater_race": ["Other", "Multiracial"],
            "rater_education": ["Other"]}
BLUE, MUTED, GRID = "#2a78d6", "#52514e", "#e6e5e1"


def item_ndfu(path):
    df = pd.read_csv(path, usecols=["item_id", "rater_gender", "rater_race", "rater_age",
                                    "rater_education", "Q3_bias_overall"])
    for col, cats in EXCLUDED.items():
        df = df[~df[col].isin(cats)]
    df["signal"] = df["Q3_bias_overall"].map(RATING_MAP)
    out = []
    for _, g in df.groupby("item_id", sort=True):
        hist = g["signal"].value_counts().reindex([1, 2, 3], fill_value=0).to_numpy(float)
        out.append(float(dfu(hist / hist.sum())))
    return np.array(out)


data = {"DICES-350": item_ndfu(DATA_DIR / "DICES_350.csv"),
        "DICES-990": item_ndfu(DATA_DIR / "DICES_990.csv")}

fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.8), facecolor="white")
for ax, (name, values) in zip(axes, data.items()):
    kept = int((values >= THETA).sum())
    ax.hist(values, bins=25, range=(0, 1), color=BLUE, edgecolor="white", linewidth=0.5)
    ax.axvline(THETA, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{name}: {kept} of {len(values)} texts at or above {THETA}",
                 loc="left", fontsize=8.5, color="#0b0b0b")
    ax.set_xlim(0, 1)
    ax.set_ylabel("number of texts", color=MUTED, fontsize=8.5)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    print(name, "texts:", len(values), "kept:", kept)
axes[1].set_xlabel("nDFU", color=MUTED, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
