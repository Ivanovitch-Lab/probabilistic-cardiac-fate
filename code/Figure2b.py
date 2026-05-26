#!/usr/bin/env python3
"""
Figure2b.py
===========
Reproduces Figure 2b of Ivanovitch (BioEssays 2026):
pairwise clonal-coupling z-score heatmaps for Small (≤ 30 cells) and Large
(> 30 cells) clones, at the main size cutoff of 30 cells. Asterisks mark
pairs significant after BH FDR correction (FDR ≤ 0.05).

Input
-----
../data/Supplementary_Table_S1.csv

Output
------
../figures/Figure2b.png

Run
---
    python Figure2b.py        # ~30 s for the Curveball permutation null
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
SIZE_THRESHOLD  = 30
MAX_SIZE        = 92
FDR_ALPHA       = 0.05
Z_VMIN, Z_VMAX  = -7, 7

REGION_COLS   = ["R1_OFT", "R2_RV", "R3_LV", "R4_AVC", "R5_AB", "R6_Atria"]
REGION_LABELS = ["OFT", "RV", "LV", "AVC", "AB", "Atria"]

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_HERE, "..", "data", "Supplementary_Table_S1.csv")
OUT_PNG    = os.path.join(_HERE, "..", "figures", "Figure2b.png")


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


# ─── Curveball null model (Strona et al. 2014) ──────────────────────────────
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


# ─── Benjamini–Hochberg FDR ─────────────────────────────────────────────────
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


# ─── Plotting ───────────────────────────────────────────────────────────────
def plot_heatmap(ax, z, q, title, show_yticks=True):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D0D0D0")
    im = ax.imshow(z, cmap=cmap, vmin=Z_VMIN, vmax=Z_VMAX, aspect="equal")
    n_reg = len(REGION_LABELS)
    ax.set_xticks(range(n_reg))
    ax.set_xticklabels(REGION_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_reg))
    ax.set_yticklabels(REGION_LABELS if show_yticks else [""] * n_reg, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5); spine.set_edgecolor("#888888")
    for i in range(n_reg):
        for j in range(n_reg):
            if i != j and not np.isnan(q[i, j]) and q[i, j] <= FDR_ALPHA:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    return im


def main():
    print("=" * 70)
    print("FIGURE 2b — clonal-coupling z-score heatmaps (cutoff = 30 cells)")
    print("=" * 70)
    df = load_data()
    small = df[df["size"] <= SIZE_THRESHOLD].copy()
    large = df[df["size"] >  SIZE_THRESHOLD].copy()
    print(f"Loaded {len(df)} clones. Small n={len(small)}, Large n={len(large)}")

    print(f"\nCurveball permutations ({N_PERMUTATIONS:,} × 2 bins) ...")
    rng = np.random.default_rng(RANDOM_SEED)
    z_small = compute_zscores(small, rng); q_small = compute_fdr(z_small)
    z_large = compute_zscores(large, rng); q_large = compute_fdr(z_large)

    print("\nSignificant pairs (FDR ≤ 0.05):")
    for label, z, q in [("Small", z_small, q_small), ("Large", z_large, q_large)]:
        for i in range(len(REGION_LABELS)):
            for j in range(i + 1, len(REGION_LABELS)):
                if not np.isnan(q[i, j]) and q[i, j] <= FDR_ALPHA:
                    print(f"  {label}: {REGION_LABELS[i]}–{REGION_LABELS[j]}  "
                          f"z={z[i,j]:.3f}  FDR={q[i,j]:.2e}")

    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=300,
                                     constrained_layout=True)
    plot_heatmap(ax_s, z_small, q_small,
                 f"Small clones (≤{SIZE_THRESHOLD} cells, n = {len(small)})",
                 show_yticks=True)
    im = plot_heatmap(ax_l, z_large, q_large,
                      f"Large clones (>{SIZE_THRESHOLD} cells, n = {len(large)})",
                      show_yticks=False)
    cb = fig.colorbar(im, ax=[ax_s, ax_l], fraction=0.04, pad=0.02, aspect=25)
    cb.set_label("Clonal coupling z-score", fontsize=8)
    cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, -0.04,
             "* FDR ≤ 0.05  •  Grey cells = self-pairs (excluded)  "
             f"•  Random seed = {RANDOM_SEED}  "
             f"•  Permutations = {N_PERMUTATIONS:,}",
             ha="center", fontsize=7, style="italic", color="#444444")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
