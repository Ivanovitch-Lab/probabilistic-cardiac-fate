#!/usr/bin/env python3
"""
SuppFigure4.py

Comprehensive overview of every distinct strictly-bifurcating
restriction-sequence topology that survives all biological filters
(strict bifurcation, every bifurcation clone-supported, monotone median
clone size). Built from an exhaustive scan of the full ~21M-combination
space (see _topology_utils.enumerate_full_space), not a top-K shortlist.
The five highest-ranked of these are shown in Figure 4b of the main text;
this supplement displays the full credible set so the reader can audit
the complete plausible landscape that the main-text figures sample from.

Each survivor is rendered as a small cladogram in the visual style of
Figure 4c (rank-1) and Figure 4d (rank-3).

Output: ../figures/SuppFigure4.png
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
    enumerate_full_space, load_clone_regions, draw_clado,
    CLUSTER_PALETTE, ARC_COLOR,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, "..", "figures", "SuppFigure4.png")

MAX_DRAW = 20  # safety cap; assert below if survivor count ever exceeds this
N_COLS = 6     # grid is 6 wide on A4 portrait; rows scale with count
DPI_OUT = 200


def _fmt_pct(w):
    """Compact percentage for tiny support weights."""
    if w >= 5e-4:
        return f"{w:.2%}"
    return "<0.05%"


def main():
    edge_prob = compute_edge_supports()
    G = build_graph(edge_prob)
    clones_df = load_clone_regions()
    print(f"Loaded {len(clones_df)} clones for size annotation.")

    print("\nEnumerating the FULL combination space (no top-K cutoff) ...")
    res = enumerate_full_space(G, clones_df, progress=True)
    terms = res["terms"]
    survivors = res["survivors"]
    f = res["funnel"]

    print(f"\n=== Funnel over ALL {res['total']:,} combinations ===")
    print(f"  {f['distinct']:>4} distinct topologies (any shape)")
    print(f"  {f['bifurcating']:>4} strictly bifurcating "
          f"(support weight {res['weight_binary']:.2%})")
    print(f"  {f['supported']:>4} + every bifurcation clone-supported "
          f"(support weight {res['weight_supported']:.2%})")
    print(f"  {f['monotone']:>4} + monotone median clone size "
          f"(support weight {res['weight_monotone']:.2%})  <- drawn")
    print(f"\n  Top-5 of the drawn set carry {res['weight_top5']:.2%} of total support.")
    print(f"  Strictly-bifurcating trees carry only "
          f"{res['weight_binary']:.2%} of total support; the remaining "
          f"{1 - res['weight_binary']:.1%} sits on MULTIFURCATING topologies.")
    print(f"  The single most-probable reconstruction (MAP) is "
          f"{'a binary tree.' if res['map_is_binary'] else 'MULTIFURCATING (not a binary tree).'}")

    n_show = len(survivors)
    assert n_show <= MAX_DRAW, (
        f"{n_show} survivors but layout caps at {MAX_DRAW}. Either widen "
        f"the cap (and re-check the grid) or tighten the filters."
    )

    lp_top = survivors[0]["best_lp"]

    # Layout: pack panels into a roughly-square grid, never wider than N_COLS.
    if n_show <= N_COLS:
        n_cols, n_rows = n_show, 1
    else:
        n_cols, n_rows = N_COLS, math.ceil(n_show / N_COLS)
    panel_w = 9.5 / N_COLS
    fig_w = max(4.5, panel_w * n_cols)
    fig_h = 1.95 * n_rows + (1.4 if n_rows == 1 else 1.8)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h), dpi=DPI_OUT)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, e in enumerate(survivors):
        ax = axes_flat[i]
        seq = derive_restriction_sequence(e["combo_paths"])
        draw_clado(ax, seq, title="", n_combos=1, log_p=e["best_lp"],
                   margin=1.0, clones_df=clones_df,
                   cooccurrence_sisters_only=True)
        rel = math.exp(e["best_lp"] - lp_top)  # relative score (vs top of drawn set)
        ax.text(0.5, 1.005, f"{i + 1}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                transform=ax.transAxes, color="#333333")
        ax.text(0.5, 0.995, f"S={rel:.2f}  w={_fmt_pct(e['weight'])}",
                ha="center", va="top", fontsize=5.5,
                transform=ax.transAxes, color="#888888")

    for ax in axes_flat[n_show:]:
        ax.axis("off")

    fig.suptitle(
        f"All {n_show} biologically-plausible binary restriction-sequence "
        f"topologies (of {res['total']:,} path combinations)\n"
        f"strictly bifurcating · every bifurcation clone-supported · "
        f"monotone median clone size — yet all binary trees together carry "
        f"only {res['weight_binary']:.1%} of total support\n"
        f"(S = relative score; w = marginal support weight)",
        fontsize=9, fontweight="bold", y=1.0,
    )

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
               markersize=7,
               label="Bifurcation node  (label = median clone size)"),
    ]
    legend_y = 0.04 if n_rows == 1 else 0.02
    rect_bottom = 0.20 if n_rows == 1 else 0.13
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, legend_y), ncol=3,
               fontsize=7, frameon=False, handletextpad=0.5,
               columnspacing=1.6, labelspacing=0.4)
    plt.tight_layout(pad=0.6, rect=(0, rect_bottom, 1, 0.90))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=DPI_OUT, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
