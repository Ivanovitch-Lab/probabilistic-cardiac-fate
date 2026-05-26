#!/usr/bin/env python3
"""
SuppFigure1.py
==============
Reproduces Supplementary Figure 1 of Ivanovitch (BioEssays 2026):
sensitivity of clonal-coupling z-scores to the Small/Large size threshold.
Runs the Curveball permutation null at five cutoffs (20, 25, 30, 35, 40
cells), in each Small and Large bin, and draws a 2 × 5 grid of heatmaps.

This script also writes the long-form CSV of every (cutoff × bin ×
region-pair) statistic as Supplementary Table S2 — used downstream by
Table1.py / Table2.py.

Input
-----
../data/Supplementary_Table_S1.csv

Outputs
-------
../figures/SuppFigure1.png
../data/Supplementary_Table_S2.csv

Run
---
    python SuppFigure1.py     # ~3 min for 5 cutoffs × 2 bins of Curveball
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


RANDOM_SEED     = 0
N_PERMUTATIONS  = 100_000
N_STEPS_BETWEEN = 5
CUTOFFS         = [20, 25, 30, 35, 40]
MAX_SIZE        = 92
FDR_ALPHA       = 0.05
Z_VMIN, Z_VMAX  = -7, 7

REGION_COLS   = ["R1_OFT", "R2_RV", "R3_LV", "R4_AVC", "R5_AB", "R6_Atria"]
REGION_LABELS = ["OFT", "RV", "LV", "AVC", "AB", "Atria"]

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_HERE, "..", "data", "Supplementary_Table_S1.csv")
OUT_PNG    = os.path.join(_HERE, "..", "figures", "SuppFigure1.png")
OUT_CSV    = os.path.join(_HERE, "..", "data", "Supplementary_Table_S2.csv")


def load_data():
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: cannot find {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df = df[df["clone_id"].notna()].copy()
    df = df[df["size"] <= MAX_SIZE].copy()
    for col in REGION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    return df


def _curveball_step(m, rng):
    n = m.shape[0]
    i, j = rng.choice(n, size=2, replace=False)
    only_i = np.where((m[i] == 1) & (m[j] == 0))[0]
    only_j = np.where((m[j] == 1) & (m[i] == 0))[0]
    if len(only_i) == 0 or len(only_j) == 0:
        return
    k = rng.integers(1, min(len(only_i), len(only_j)) + 1)
    si = rng.choice(only_i, size=k, replace=False)
    sj = rng.choice(only_j, size=k, replace=False)
    m[i, si] = 0;  m[i, sj] = 1
    m[j, sj] = 0;  m[j, si] = 1


def compute_zscores(subset_df, rng):
    binary = (subset_df[REGION_COLS].values > 0).astype(int)
    n_clones, n_reg = binary.shape
    if n_clones == 0:
        return np.full((n_reg, n_reg), np.nan)
    obs = (binary.T @ binary).astype(float)
    p_sum = np.zeros_like(obs); p_sq = np.zeros_like(obs)
    cur = binary.copy()
    for _ in range(5 * n_clones):
        _curveball_step(cur, rng)
    for _ in range(N_PERMUTATIONS):
        for _ in range(N_STEPS_BETWEEN):
            _curveball_step(cur, rng)
        pc = (cur.T @ cur).astype(float)
        p_sum += pc; p_sq += pc ** 2
    mean = p_sum / N_PERMUTATIONS
    std  = np.sqrt(np.maximum((p_sq / N_PERMUTATIONS) - mean ** 2, 1e-8))
    z = (obs - mean) / std
    np.fill_diagonal(z, np.nan)
    return z


def _bh(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        v = min((n / (i + 1)) * p[order[i]], prev)
        q[i] = v; prev = v
    out = np.empty(n); out[order] = q
    return out


def compute_fdr(z_matrix):
    n = len(REGION_LABELS)
    pairs, pvals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            z = z_matrix[i, j]
            p = 2.0 * (1.0 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            pairs.append((i, j)); pvals.append(p)
    valid = [k for k, p in enumerate(pvals) if not np.isnan(p)]
    q_arr = np.full(len(pvals), np.nan)
    if valid:
        q_v = _bh(np.array([pvals[k] for k in valid]))
        for k, q in zip(valid, q_v):
            q_arr[k] = q
    q_matrix = np.full((n, n), np.nan)
    for k, (i, j) in enumerate(pairs):
        q_matrix[i, j] = q_arr[k]; q_matrix[j, i] = q_arr[k]
    return q_matrix


def plot_cell(ax, z, q, title, show_x=True, show_y=True):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D0D0D0")
    im = ax.imshow(z, cmap=cmap, vmin=Z_VMIN, vmax=Z_VMAX, aspect="equal")
    n_reg = len(REGION_LABELS)
    if show_x:
        ax.set_xticks(range(n_reg))
        ax.set_xticklabels(REGION_LABELS, rotation=45, ha="right", fontsize=6.5)
    else:
        ax.set_xticks([])
    if show_y:
        ax.set_yticks(range(n_reg))
        ax.set_yticklabels(REGION_LABELS, fontsize=6.5)
    else:
        ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5); spine.set_edgecolor("#888888")
    for i in range(n_reg):
        for j in range(n_reg):
            if i != j and not np.isnan(q[i, j]) and q[i, j] <= FDR_ALPHA:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    return im


def main():
    print("=" * 70)
    print("SUPPLEMENTARY FIGURE 1 — sensitivity to size-bin cutoff")
    print("=" * 70)
    df = load_data()
    print(f"Loaded {len(df)} clones (excluding the 142-cell clone)")

    rng = np.random.default_rng(RANDOM_SEED)
    print(f"\nCurveball ({N_PERMUTATIONS:,} permutations × {len(CUTOFFS)} cutoffs "
          "× 2 bins). Takes ~3 min...")

    results = {}
    rows = []
    for cutoff in CUTOFFS:
        small = df[df["size"] <= cutoff].copy()
        large = df[df["size"] >  cutoff].copy()
        print(f"  cutoff={cutoff:>2}: Small n={len(small):>3}, Large n={len(large):>3}")
        for label, subset in [("Small", small), ("Large", large)]:
            z = compute_zscores(subset, rng)
            q = compute_fdr(z)
            results[(cutoff, label)] = (z, q, len(subset))
            for i in range(len(REGION_LABELS)):
                for j in range(i + 1, len(REGION_LABELS)):
                    if np.isnan(z[i, j]):
                        continue
                    p_raw = 2.0 * (1.0 - norm.cdf(abs(z[i, j])))
                    rows.append({
                        "cutoff":  cutoff,
                        "bin":     label,
                        "region1": REGION_LABELS[i],
                        "region2": REGION_LABELS[j],
                        "pair":    f"{REGION_LABELS[i]}-{REGION_LABELS[j]}",
                        "z":       float(z[i, j]),
                        "p_raw":   float(p_raw),
                        "p_FDR":   float(q[i, j]),
                        "sig":     bool(q[i, j] <= FDR_ALPHA),
                    })

    long_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    long_df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Wrote: {OUT_CSV}  ({len(long_df)} rows)")

    print("\nSignificant pairs (FDR ≤ 0.05):")
    sig = long_df[long_df["sig"]].sort_values(["bin", "cutoff", "pair"])
    for _, r in sig.iterrows():
        print(f"  cutoff={r['cutoff']:>2}  {r['bin']:>5}  "
              f"{r['pair']:<11}  z={r['z']:>6.3f}  FDR={r['p_FDR']:.2e}")

    # 2 rows × 5 cols heatmap grid: top row Small, bottom row Large
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), dpi=300,
                             constrained_layout=True)
    fig.suptitle("Sensitivity of clonal-coupling z-scores to size-bin cutoff",
                 fontsize=11, fontweight="bold")

    im_ref = None
    for col_i, cutoff in enumerate(CUTOFFS):
        for row_i, label in enumerate(["Small", "Large"]):
            ax = axes[row_i, col_i]
            z, q, n = results[(cutoff, label)]
            op = "≤" if label == "Small" else ">"
            title = f"Cutoff = {cutoff} cells\nn = {n}"
            im = plot_cell(ax, z, q, title,
                           show_x=(row_i == 1),
                           show_y=(col_i == 0))
            if im_ref is None:
                im_ref = im

    # Row labels
    fig.text(0.005, 0.74, "Small bin", rotation=90, va="center", ha="center",
             fontsize=10, fontweight="bold")
    fig.text(0.005, 0.30, "Large bin", rotation=90, va="center", ha="center",
             fontsize=10, fontweight="bold")

    cb = fig.colorbar(im_ref, ax=axes.ravel().tolist(),
                      fraction=0.012, pad=0.02, aspect=40)
    cb.set_label("Clonal coupling z-score", fontsize=9)
    cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
    cb.ax.tick_params(labelsize=8)

    fig.text(0.5, -0.02,
             "* FDR ≤ 0.05  •  Grey cells = self-pairs (excluded)  "
             f"•  Random seed = {RANDOM_SEED}  "
             f"•  Permutations = {N_PERMUTATIONS:,}",
             ha="center", fontsize=8, style="italic", color="#444444")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
