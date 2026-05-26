#!/usr/bin/env python3
"""
Figure4b.py

Vertical cladogram of the rank-1 plausible restriction sequence — the
highest-joint-score topology among the candidate restriction sequences
that pass the three biological filters (strictly bifurcating, every
bifurcation clone-supported, monotone median clone size).

Root at TOP (multi-potent epiblast progenitor), terminal fates at BOTTOM —
matching the orientation of the main hierarchy network figure (Figure 3c).

Restriction sequence (top → bottom):
  1. Proximal | Distal                          (first split)
  2. LV separates from the proximal pool        (earliest committed fate)
  3. Atria separates from the junctional pool
  4. AB | AVC   and   RV | OFT                  (final fate commitment)

Significant Small-clone co-occurrences (FDR < 0.05) shown as arcs below
the terminal nodes — both significant pairs (AB–AVC, OFT–RV) sit within
their respective clades as direct sister leaves.

Output: ../figures/Figure4b.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, "..", "figures", "Figure4b.png")

CLUSTER_PALETTE = {
    1: "#CC79A7",
    2: "#E69F00",
    3: "#0088FF",
}

# ── Leaf positions  (x = horizontal fate layout, y fixed at bottom) ───────────
# Order left→right mirrors original cladogram y-positions (Atria→OFT)
LEAVES = {
    "Atria": dict(x=0.08, y=0.14, cluster=1, median=8),
    "AB":    dict(x=0.24, y=0.14, cluster=2, median=4),
    "AVC":   dict(x=0.40, y=0.14, cluster=2, median=6),
    "LV":    dict(x=0.56, y=0.14, cluster=3, median=8),
    "RV":    dict(x=0.74, y=0.14, cluster=3, median=5),
    "OFT":   dict(x=0.88, y=0.14, cluster=1, median=6),
}

# ── Internal nodes  (x = midpoint of children, y = restriction depth) ─────────
# y: higher = earlier restriction (root at top)
NODES = {
    "N_JUNC":  dict(x=0.32, y=0.38),   # AB | AVC
    "N_DIST":  dict(x=0.81, y=0.38),   # RV | OFT
    "N_PROX2": dict(x=0.20, y=0.56),   # Atria | (AB+AVC)
    "N_PROX":  dict(x=0.38, y=0.74),   # (AB+AVC+Atria) | LV
    "Root":    dict(x=0.60, y=0.90),   # Proximal | Distal
}

# Tree topology: c1=left child, c2=right child
TREE = {
    "Root":    dict(c1="N_PROX",  c2="N_DIST"),
    "N_DIST":  dict(c1="RV",      c2="OFT"),
    "N_PROX":  dict(c1="N_PROX2", c2="LV"),
    "N_PROX2": dict(c1="Atria",   c2="N_JUNC"),
    "N_JUNC":  dict(c1="AB",      c2="AVC"),
}

NODE_R   = 0.058
INT_R    = 0.036
LINE_COL = "#BBBBBB"
LINE_LW  = 1.4

# ── Median clone size at each bifurcation node ──
# Strictly hierarchical: each node uses ONLY clones within its own subtree.
# Root:    5- and 6-fate clones (n=5, most multipotent)         → 85 cells
# N_PROX:  LV + any proximal, no distal (n=6)                  → 38 cells
# N_PROX2: (AB or AVC) + Atria, no LV, no distal (n=5)         → 20 cells
# N_JUNC:  pure AB+AVC clones (n=4)                            → 14 cells
# N_DIST:  pure RV+OFT clones (n=10)                           → 18 cells
NODE_CLONE_DATA = {
    "Root":    dict(median=85,  n=5),
    "N_PROX":  dict(median=38,  n=6),
    "N_PROX2": dict(median=20,  n=5),
    "N_JUNC":  dict(median=14,  n=4),
    "N_DIST":  dict(median=18,  n=10),
}


COOCCURRENCES = [
    dict(pair=("AB",  "AVC"), z=4.96, fdr="1.1×10⁻⁵", rad=-0.80),
    dict(pair=("RV",  "OFT"), z=4.38, fdr="8.9×10⁻⁵", rad=-0.80),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _pos(name):
    if name in LEAVES:
        return LEAVES[name]["x"], LEAVES[name]["y"]
    return NODES[name]["x"], NODES[name]["y"]


def _draw_tree(ax, name):
    """Recursive: horizontal crossbar at node y, vertical drops to children."""
    if name not in TREE:
        return
    node = TREE[name]
    xN, yN = _pos(name)
    c1, c2 = node["c1"], node["c2"]
    xC1, yC1 = _pos(c1)
    xC2, yC2 = _pos(c2)

    x_lo, x_hi = min(xC1, xC2), max(xC1, xC2)

    # Horizontal crossbar
    ax.plot([x_lo, x_hi], [yN, yN], color=LINE_COL, lw=LINE_LW,
            solid_capstyle="round", zorder=1)
    # Vertical drops to each child
    for xC, yC in [(xC1, yC1), (xC2, yC2)]:
        ax.plot([xC, xC], [yN, yC], color=LINE_COL, lw=LINE_LW,
                solid_capstyle="round", zorder=1)

    _draw_tree(ax, c1)
    _draw_tree(ax, c2)


def main() -> None:
    # Half-A4 width (~105 mm) to sit beside the k3 hierarchy backbone figure.
    fig, ax = plt.subplots(figsize=(4.13, 4.5), dpi=300)
    ax.set_xlim(-0.28, 1.08)
    # Lower bound just past the cell-count labels — no wasted vertical space
    # so the legend can sit close beneath the cladogram with a small gap.
    ax.set_ylim(-0.04, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Background zones ───────────────────────────────────────────────────────
    # Proximal (left): Atria, AB, AVC, LV
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.06, 0.02), 0.70, 0.95,
        boxstyle="round,pad=0.005",
        facecolor="#F8F8F8", edgecolor="none", alpha=0.85, zorder=0))
    # Distal (right): RV, OFT
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.66, 0.02), 0.38, 0.95,
        boxstyle="round,pad=0.005",
        facecolor="#EEEEFF", edgecolor="none", alpha=0.85, zorder=0))
    # Boundary dashed line — clipped to the background zone (data coords)
    ax.plot([0.65, 0.65], [0.02, 0.97], color="#CCCCDD",
            lw=0.8, linestyle="--", zorder=0)

    # Zone labels (top)
    ax.text(0.30, 0.99, "PROXIMAL", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#777777")
    ax.text(0.80, 0.99, "DISTAL",   ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#7777AA")

    # ── Restriction arrow (labelled "Restriction sequence") ───────────────────
    ax.annotate("", xy=(-0.10, 0.12), xytext=(-0.10, 0.92),
                arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=1.0))
    ax.text(-0.13, 0.52, "Restriction sequence",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#888888", rotation=90)

    # ── Clone-size / stage gradient bar ──────────────────────────────────────
    # Warm (large clones, epiblast) at top → cool (small clones, mesoderm) at bottom.
    # Gradient fades through the transition zone (N_PROX 38 cells ↔ N_PROX2 20 cells).
    c_top = np.array([250, 190, 130]) / 255   # warm orange
    c_bot = np.array([140, 190, 240]) / 255   # cool blue
    bx0, bx1 = -0.27, -0.255
    by_top, by_bot = 0.95, 0.10
    n_grad = 40
    for i in range(n_grad):
        t  = i / n_grad
        y0 = by_bot + t       * (by_top - by_bot)
        y1 = by_bot + (t + 1/n_grad) * (by_top - by_bot)
        c  = tuple(c_bot * (1 - t) + c_top * t)
        ax.add_patch(mpatches.Rectangle(
            (bx0, y0), bx1 - bx0, y1 - y0,
            facecolor=c, edgecolor="none", alpha=0.60, zorder=0))

    # Dashed lines bracketing the transition zone (between N_PROX y=0.74 and N_PROX2 y=0.56)
    for y_dash in [0.74, 0.56]:
        ax.plot([bx0 - 0.005, bx1 + 0.005], [y_dash, y_dash],
                color="#BBBBBB", lw=0.6, linestyle="--", zorder=1)

    # Zone labels
    ax.text((bx0 + bx1) / 2, by_top + 0.01, "Large clones",
            ha="center", va="bottom", fontsize=6.5, color="#B8602A",
            fontweight="bold")
    ax.text((bx0 + bx1) / 2, by_bot - 0.01, "Small clones",
            ha="center", va="top", fontsize=6.5, color="#2A6AB8",
            fontweight="bold")

    # ── Cladogram lines ────────────────────────────────────────────────────────
    _draw_tree(ax, "Root")

    # ── Co-occurrence arcs (below leaf row) ────────────────────────────────────
    # Neutral dark-grey arcs: the signal here is statistical (FDR<0.05),
    # independent of cluster identity, so a neutral colour avoids implying
    # the arc itself carries cluster meaning.
    arc_color = "#333333"
    for cooc in COOCCURRENCES:
        n1, n2 = cooc["pair"]
        r1, r2 = LEAVES[n1], LEAVES[n2]
        ax.add_patch(FancyArrowPatch(
            posA=(r1["x"], r1["y"]),
            posB=(r2["x"], r2["y"]),
            connectionstyle=f"arc3,rad={cooc['rad']}",
            arrowstyle="-",
            color=arc_color, linewidth=1.5, alpha=0.85, zorder=3,
        ))

    # ── Internal nodes ─────────────────────────────────────────────────────────
    for name, node in NODES.items():
        if name == "Root":
            continue
        ax.add_patch(plt.Circle((node["x"], node["y"]), INT_R,
            facecolor="#DDDDDD", edgecolor="white",
            linewidth=0.8, zorder=4))
        ax.text(node["x"], node["y"] - INT_R - 0.018,
                f'{NODE_CLONE_DATA[name]["median"]}',
                ha="center", va="top", fontsize=6.5, fontweight="bold",
                color="#777777", zorder=5)

    # ── Leaf nodes ─────────────────────────────────────────────────────────────
    for name, leaf in LEAVES.items():
        x, y = leaf["x"], leaf["y"]
        color = CLUSTER_PALETTE[leaf["cluster"]]
        ax.add_patch(plt.Circle((x, y), NODE_R,
            facecolor=color, edgecolor="white",
            linewidth=1.4, alpha=0.92, zorder=5))
        ax.text(x, y, name, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white", zorder=6)
        ax.text(x, y - NODE_R - 0.018, f'{leaf["median"]}',
                ha="center", va="top", fontsize=5.5, fontweight="bold",
                color="#777777", zorder=5)

    # ── Root node ──────────────────────────────────────────────────────────────
    rx, ry = NODES["Root"]["x"], NODES["Root"]["y"]
    ax.add_patch(plt.Circle((rx, ry), INT_R,
        facecolor="#AAAAAA", edgecolor="white",
        linewidth=1.4, alpha=0.88, zorder=5))
    ax.text(rx, ry, "All", ha="center", va="center",
            fontsize=5.5, fontweight="bold", color="white", zorder=6)
    ax.text(rx, ry - INT_R - 0.018,
            f'{NODE_CLONE_DATA["Root"]["median"]}',
            ha="center", va="top", fontsize=6.5, fontweight="bold",
            color="#777777", zorder=5)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=CLUSTER_PALETTE[3], edgecolor="white", lw=1,
                       label="LV / RV"),
        mpatches.Patch(facecolor=CLUSTER_PALETTE[2], edgecolor="white", lw=1,
                       label="AVC / AB"),
        mpatches.Patch(facecolor=CLUSTER_PALETTE[1], edgecolor="white", lw=1,
                       label="OFT / Atria"),
        Line2D([0], [0], color=arc_color, lw=1.5, alpha=0.85,
               label="Significant co-occurrence  (FDR < 0.05)"),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor="#DDDDDD", markeredgecolor="#BBBBBB",
               markersize=7, label="Bifurcation node  (label = median clone size)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.08),
        ncol=1, fontsize=7.0, frameon=False, handlelength=1.6,
        handletextpad=0.5, labelspacing=0.5,
    )

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
