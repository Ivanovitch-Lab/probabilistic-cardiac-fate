#!/usr/bin/env python3
"""
Figure2a.py
===========
Reproduces Figure 2a:
scatter plot of clone size vs number of cardiac regions occupied per clone,
with the Small/Large size threshold at 30 cells.

Input
-----
../data/Supplementary_Table_S1.csv   Meilhac clone-by-region table.

Output
------
../figures/Figure2a.png

Run
---
    python Figure2a.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RANDOM_SEED    = 0
SIZE_THRESHOLD = 30
MAX_SIZE       = 92

REGION_COLS = ["R1_OFT", "R2_RV", "R3_LV", "R4_AVC", "R5_AB", "R6_Atria"]

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_HERE, "..", "data", "Supplementary_Table_S1.csv")
OUT_PNG    = os.path.join(_HERE, "..", "figures", "Figure2a.png")

SMALL_COLOR = "#7FA8C8"
LARGE_COLOR = "#E68A00"
SMALL_LABEL = "#1F4E79"


def load_data():
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: cannot find {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df = df[df["clone_id"].notna()].copy()
    df = df[df["size"] <= MAX_SIZE].copy()
    for col in REGION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["size"]    = pd.to_numeric(df["size"], errors="coerce")
    df["breadth"] = (df[REGION_COLS] > 0).sum(axis=1)
    return df


def plot(df, ax):
    rng = np.random.default_rng(RANDOM_SEED)
    small = df[df["size"] <= SIZE_THRESHOLD].copy()
    large = df[df["size"] >  SIZE_THRESHOLD].copy()
    small["y"] = small["breadth"] + rng.uniform(-0.18, 0.18, size=len(small))
    large["y"] = large["breadth"] + rng.uniform(-0.14, 0.14, size=len(large))

    ax.axvspan(0, SIZE_THRESHOLD, color="#F4F6F8", zorder=0)
    ax.axvline(SIZE_THRESHOLD, color="#888888", linestyle="--", linewidth=0.8, zorder=1)

    rho_all,   p_all   = spearmanr(df["size"],    df["breadth"])
    rho_small, p_small = spearmanr(small["size"], small["breadth"])
    rho_large, p_large = spearmanr(large["size"], large["breadth"])
    pstr = lambda p: "p<0.001" if p < 0.001 else f"p={p:.2f}"

    ax.scatter(small["size"], small["y"],
               c=SMALL_COLOR, s=14, alpha=0.75,
               linewidths=0.3, edgecolors="white", zorder=3,
               label=f"Small  (≤{SIZE_THRESHOLD}, n={len(small)}):  "
                     f"ρ={rho_small:.2f}, {pstr(p_small)}")
    ax.scatter(large["size"], large["y"],
               c=LARGE_COLOR, s=24, alpha=0.88,
               linewidths=0.35, edgecolors="white", zorder=4,
               label=f"Large  (>{SIZE_THRESHOLD}, n={len(large)}):  "
                     f"ρ={rho_large:.2f}, {pstr(p_large)}")

    def _arrow(xy, xytext, text, color, ha="left", va="center"):
        ax.annotate(text, xy=xy, xytext=xytext,
                    fontsize=6.0, fontweight="bold", color=color,
                    ha=ha, va=va,
                    arrowprops=dict(
                        arrowstyle="-|>,head_width=0.30,head_length=0.45",
                        color=color, lw=1.2, alpha=0.95, shrinkA=2, shrinkB=3))

    def _row_for(d, predicate):
        sub = d[predicate(d)]
        return sub.iloc[0] if not sub.empty else None

    callouts = [
        (small, 11, 3, (4, 3.55),  "11 cells\nAVC+AB+Atria",            SMALL_LABEL, "left",   "center"),
        (small, 30, 4, (29, 4.65), "30 cells\nLV+AVC+AB+Atria",         SMALL_LABEL, "right",  "center"),
        (large, 48, 5, (36, 5.7),  "48 cells\nRV+LV+AVC+AB+Atria",      LARGE_COLOR, "left",   "center"),
        (large, 57, 4, (65, 4.7),  "57 cells\nOFT+RV+LV+Atria",         LARGE_COLOR, "left",   "center"),
        (large, 75, 3, (82, 3.7),  "75 cells\nLV+AB+Atria",             LARGE_COLOR, "left",   "center"),
        (large, 54, 2, (55, 2.75), "54 cells\nOFT+RV",                  LARGE_COLOR, "left",   "center"),
    ]
    for df_sub, sz, br, xy_to, txt, col, ha, va in callouts:
        r = _row_for(df_sub, lambda d, sz=sz, br=br: (d["size"] == sz) & (d["breadth"] == br))
        if r is not None:
            _arrow((r["size"], r["y"]), xy_to, txt, col, ha, va)
    r = _row_for(large, lambda d: (d["size"] == 42) & (d["breadth"] == 1))
    if r is not None:
        _arrow((r["size"], r["y"]), (r["size"], 0.55),
               "42 cells\nAtria only", LARGE_COLOR, "center", "top")

    ax.text(0.985, 0.04,
            f"All n={len(df)}:  ρ={rho_all:.2f}, {pstr(p_all)}",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=6.0, color="#444444", fontstyle="italic")

    ax.set_xlabel("Clone size (cells)", fontsize=8)
    ax.set_ylabel("Number of cardiac regions per clone", fontsize=8)
    ax.set_yticks(range(1, 7))
    ax.set_ylim(0.0, 6.7)
    ax.set_xlim(0, MAX_SIZE + 4)
    ax.tick_params(labelsize=7, length=2.5, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0),
              fontsize=6.5, frameon=False,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.0)


def main():
    print("=" * 70)
    print("FIGURE 2a — clone size vs regional breadth")
    print("=" * 70)
    df = load_data()
    print(f"Loaded {len(df)} clones (excluding the 142-cell clone)")
    small = df[df["size"] <= SIZE_THRESHOLD]
    large = df[df["size"] >  SIZE_THRESHOLD]
    print(f"  Small (≤{SIZE_THRESHOLD}): n = {len(small)}")
    print(f"  Large (>{SIZE_THRESHOLD}): n = {len(large)}")

    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
    plot(df, ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
