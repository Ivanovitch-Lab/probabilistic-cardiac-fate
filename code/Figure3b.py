#!/usr/bin/env python3
"""
Figure3b.py
===========
Reproduces Figure 3b of the accompanying manuscript:
log2 fold-enrichment of each k=3 cluster's mean fate composition relative
to the mean across all clustered intermediate states. Three rows (one per
cluster) × six columns (one per terminal fate).

Input
-----
../data/Supplementary_Table_S4.csv   (produced by Figure3a.py)

Output
------
../figures/Figure3b.png

Run
---
    python Figure3b.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIONS = ["OFT", "RV", "LV", "AVC", "AB", "Atria"]

CLUSTER_PALETTE = {1: "#CC79A7", 2: "#E69F00", 3: "#0088FF"}
CLUSTER_LABELS  = {1: "Atria/OFT", 2: "AVC/AB", 3: "LV/RV"}
CLUSTER_ORDER   = [1, 2, 3]   # top → bottom row order

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_HERE, "..", "data", "Supplementary_Table_S4.csv")
OUT_PNG    = os.path.join(_HERE, "..", "figures", "Figure3b.png")


def main():
    print("=" * 70)
    print("FIGURE 3b — log2 fold-enrichment per k=3 cluster")
    print("=" * 70)
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: cannot find {INPUT_FILE}. Run Figure3a.py first.")

    df = pd.read_csv(INPUT_FILE)
    inter = df[df["cluster_k3"].notna()].copy()
    inter["cluster_k3"] = inter["cluster_k3"].astype(int)
    print(f"Loaded {len(inter)} clustered intermediate states from {INPUT_FILE}")

    fcols   = [f"frac_{f}" for f in REGIONS]
    overall = inter[fcols].mean()

    fold = np.zeros((3, len(REGIONS)))
    for i, c in enumerate(CLUSTER_ORDER):
        sub = inter[inter["cluster_k3"] == c]
        for j, col in enumerate(fcols):
            cmean = sub[col].mean()
            omean = overall[col]
            fold[i, j] = (
                np.log2(cmean / omean) if omean > 0 and cmean > 0
                else (-3.0 if cmean == 0 else 3.0)
            )

    vmax = max(np.abs(fold).max(), 1.0)
    # Parameters copied verbatim from 6. Plot_spectrum_of_states.py
    # (plot_enrichment_figure) which produced the published Figure 3b.
    fig, ax = plt.subplots(figsize=(2.6, 3.35), dpi=300)
    im = ax.imshow(fold, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(REGIONS)))
    ax.set_xticklabels(REGIONS, fontsize=6.5, rotation=35, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels([CLUSTER_LABELS[c] for c in CLUSTER_ORDER], fontsize=6.5)
    for tick, c in zip(ax.get_yticklabels(), CLUSTER_ORDER):
        tick.set_color(CLUSTER_PALETTE[c])

    ax.tick_params(axis="x", pad=1)

    ax.set_title("log₂ fold-enrichment", fontsize=7, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.085, pad=0.04)
    cb.set_label("log₂ fold", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    print(f"\nCluster sizes:  " +
          ", ".join(f"{CLUSTER_LABELS[c]}: n={(inter['cluster_k3']==c).sum()}"
                    for c in CLUSTER_ORDER))
    print(f"\nlog2 fold-enrichment (rows = cluster, cols = fate):")
    print(pd.DataFrame(fold, index=[CLUSTER_LABELS[c] for c in CLUSTER_ORDER],
                       columns=REGIONS).round(2).to_string())

    plt.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
