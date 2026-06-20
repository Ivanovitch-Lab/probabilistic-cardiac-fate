#!/usr/bin/env python3
"""
Figure4b.py

Top-5 binary restriction-sequence topologies by best-path score,
drawn as a compact strip of mini-cladograms. Acts as the visual anchor
between Figure 4a (the strip plot of best-path scores across the
full topology landscape) and Figures 4c/4d (the detailed rank-1 and
rank-3 cladograms): each panel here is one of the five highest-scoring
binary trees among those satisfying all three biological filters (strict
bifurcation, every bifurcation clone-supported, monotone median clone
size), exhaustively enumerated over the full ~21M-combination path space
(_topology_utils.enumerate_full_space).

Best-path score is the same metric used on Figure 4a's x-axis —
score of the SINGLE BEST 6-path combination producing this topology,
divided by the score of the overall best combination across all
combinations (= 1.0 by construction). Annotated on each panel.

The full 17-topology credible set is shown in SuppFigure 3.

Output: ../figures/Figure4b.png
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import compute_edge_supports, build_graph
from _sequence_utils import derive_restriction_sequence
from _topology_utils import (
    all_paths_per_terminal, enumerate_full_space, load_clone_regions,
    draw_clado, CLUSTER_PALETTE, ARC_COLOR,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, "..", "figures", "Figure4b.png")

N_TOP = 5
DPI_OUT = 300


def main():
    edge_prob = compute_edge_supports()
    G = build_graph(edge_prob)
    clones_df = load_clone_regions()

    # Global best lp = rank-1 combination's joint lp (= sum of per-terminal
    # rank-1 log path-scores). This is the denominator of best-path.
    ppt = all_paths_per_terminal(G)
    terms = sorted(ppt.keys())
    global_best_lp = sum(math.log(max(ppt[t][0][1], 1e-30)) for t in terms)

    print("Enumerating the full combination space (no top-K cutoff) ...")
    res = enumerate_full_space(G, clones_df, progress=True)
    survivors = res["survivors"]
    n = min(N_TOP, len(survivors))
    top = survivors[:n]

    print(f"\nTop {n} of {len(survivors)} monotone-median survivors:")
    for i, e in enumerate(top, 1):
        rel = math.exp(e["best_lp"] - global_best_lp)
        print(f"  S{i}: best_lp={e['best_lp']:.3f}  "
              f"best-path={rel:.3f}  weight={e['weight']:.3%}  "
              f"n_combos={e['n_combos']:,}")

    # ── 1 x N_TOP strip layout, sized for main-figure economy ───────────
    panel_w, panel_h = 1.45, 1.55
    fig_w = panel_w * n + 0.3
    fig_h = panel_h + 0.6   # room for top S-label band + bottom legend
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), dpi=DPI_OUT)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, e in enumerate(top):
        ax = axes_flat[i]
        seq = derive_restriction_sequence(e["combo_paths"])
        draw_clado(ax, seq, title="", n_combos=1, log_p=e["best_lp"],
                   margin=1.0, clones_df=clones_df)
        rel = math.exp(e["best_lp"] - global_best_lp)
        ax.text(0.5, 1.02, f"S{i + 1}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", transform=ax.transAxes,
                color="#333333")
        ax.text(0.5, 1.005, f"L = {rel:.2f}",
                ha="center", va="top", fontsize=5.5,
                transform=ax.transAxes, color="#888888")

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    legend_handles = [
        mpatches.Patch(facecolor=CLUSTER_PALETTE[3], edgecolor="white",
                       linewidth=1, label="LV / RV"),
        mpatches.Patch(facecolor=CLUSTER_PALETTE[2], edgecolor="white",
                       linewidth=1, label="AVC / AB"),
        mpatches.Patch(facecolor=CLUSTER_PALETTE[1], edgecolor="white",
                       linewidth=1, label="OFT / Atria"),
        Line2D([0], [0], color=ARC_COLOR, lw=1.5, alpha=0.85,
               label="Co-occurrence (FDR < 0.05)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#DDDDDD", markeredgecolor="#BBBBBB",
               markersize=6,
               label="Bifurcation node (label = median clone size)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6,
               frameon=False, handletextpad=0.4, columnspacing=1.2,
               labelspacing=0.3)
    plt.tight_layout(pad=0.4, rect=(0, 0.12, 1, 0.94))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=DPI_OUT, facecolor="white",
                bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
