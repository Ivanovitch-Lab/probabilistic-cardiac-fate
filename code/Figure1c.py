#!/usr/bin/env python3
"""
Figure1c.py
===========
Reproduces Figure 1c of Ivanovitch (BioEssays 2026):
strip plot of myocardial cells per clone from the Abukar et al. (2025) live-
imaging dataset, split by cardiac identity (AVC/LV vs Atria/inflow) and
coloured by whether the clone is single-fate or multi-fated.

Input
-----
../data/Abukar_clone_data.csv
    Per-clone summary of the live-imaging tracking data (221 clones × 10
    columns: clone_id, total_cells, LV, Atria, Meso, Endocardium,
    Pericardium, Endothelial, ExE Meso, Endoderm).

Output
------
../figures/Figure1c.png

Run
---
    python Figure1c.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


RANDOM_SEED = 0

_HERE       = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_HERE, "..", "data", "Abukar_clone_data.csv")
OUT_PNG     = os.path.join(_HERE, "..", "figures", "Figure1c.png")

# Differentiated non-myocardial fates used to classify clones as multi-fated.
# "Meso" is undifferentiated mesoderm and is not counted as a fate alternative.
DIFFERENTIATED_NON_MYO = [
    "Endocardium", "Pericardium", "Endothelial", "ExE Meso", "Endoderm",
]

MONO_COLOR  = "#7FA8C8"   # single-fate (myocardial-only) clones
MULTI_COLOR = "#E68A00"   # multi-fated clones


def load_clones():
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: cannot find {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    df["myo"]      = df["LV"] + df["Atria"]
    df["is_multi"] = (df[DIFFERENTIATED_NON_MYO] > 0).any(axis=1)
    df["cardiac"]  = np.where(df["LV"]    > 0, "LV",
                       np.where(df["Atria"] > 0, "Atria", "None"))

    n_myo       = (df["myo"] > 0).sum()
    n_myo_multi = ((df["myo"] > 0) & df["is_multi"]).sum()
    print(f"Loaded {len(df)} clones from {INPUT_FILE}")
    print(f"  Myocardial clones (≥1 LV or Atria cell): {n_myo}")
    print(f"    LV-containing: {(df['LV']>0).sum()}")
    print(f"    Atria-only   : {((df['Atria']>0) & (df['LV']==0)).sum()}")
    print(f"  Of myocardial clones, multi-fate "
          f"(differentiated non-myo co-fate, Meso excluded): {n_myo_multi}")
    return df


def plot_panel_strip(ax, clones, rng):
    """Strip plot — myocardial cells per clone, split by cardiac fate."""
    myo = clones[clones["cardiac"] != "None"].copy()
    x_pos     = {"LV": 0, "Atria": 1}
    label_for = {"LV": "AVC/LV", "Atria": "Atria (inflow)"}
    myo["x_base"] = myo["cardiac"].map(x_pos)
    myo["x"] = myo["x_base"] + rng.uniform(-0.34, 0.34, size=len(myo))
    myo["y"] = myo["myo"]    + rng.uniform(-0.18, 0.18, size=len(myo))

    mono  = myo[~myo["is_multi"]]
    multi = myo[ myo["is_multi"]]

    # Faint vertical guides per column
    for x in x_pos.values():
        ax.axvline(x, color="#EEEEEE", lw=0.7, zorder=0)

    # Median bars per cardiac group
    for fate, x in x_pos.items():
        med = myo[myo["cardiac"] == fate]["myo"].median()
        ax.hlines(med, x - 0.42, x + 0.42, color="#444444", lw=1.1, zorder=2)

    ax.scatter(mono["x"], mono["y"],
               c=MONO_COLOR, s=18, alpha=0.85,
               linewidths=0.3, edgecolors="white", zorder=3,
               label=f"Single fate (n={len(mono)})")
    ax.scatter(multi["x"], multi["y"],
               c=MULTI_COLOR, s=18, alpha=0.92,
               linewidths=0.3, edgecolors="white", zorder=4,
               label=f"Multi-fated (n={len(multi)})")

    lv_sizes = myo[myo["cardiac"] == "LV"]["myo"]
    at_sizes = myo[myo["cardiac"] == "Atria"]["myo"]
    _, p = mannwhitneyu(lv_sizes, at_sizes, alternative="two-sided")
    p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.text(0.985, 0.985,
            f"AVC/LV vs Atria\nMann–Whitney U: {p_str}",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=5.5, color="#444444", fontstyle="italic",
            linespacing=1.3)

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0),
              fontsize=5.5, frameon=False,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.0)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([label_for[k] for k in x_pos.keys()],
                       fontsize=7, fontweight="bold")
    ax.set_xlim(-0.7, 1.75)
    ax.set_ylabel("Myocardial cells per clone", fontsize=6.5)
    ymax = int(myo["myo"].max()) + 1
    ax.set_yticks(range(0, ymax + 2, 2))
    ax.set_ylim(0, ymax + 1)
    ax.tick_params(axis="y", labelsize=5.5, length=2.5, pad=2)
    ax.tick_params(axis="x", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("=" * 70)
    print("FIGURE 1c — Myocardial clone size from live imaging (Abukar et al.)")
    print("=" * 70)
    print()

    clones = load_clones()
    rng    = np.random.default_rng(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(3.1, 2.8), dpi=300)
    plot_panel_strip(ax, clones, rng)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
