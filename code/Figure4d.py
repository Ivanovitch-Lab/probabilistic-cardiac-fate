#!/usr/bin/env python3
"""
Figure4d.py

Vertical cladogram of the rank-3 plausible restriction sequence — the
third-highest-joint-score topology among the candidate restriction
sequences that pass the three biological filters (strictly bifurcating,
every bifurcation clone-supported, monotone median clone size).

Root at TOP (multi-potent epiblast progenitor), terminal fates at BOTTOM —
same orientation as Figure 4c (rank-1 plausible cladogram).

The Rank-3 topology differs from Rank 1 in where LV branches: here LV
groups with the OFT–RV (right-ventricular) pole rather than with the
AB–AVC–Atria (junctional) pole. The two FDR-significant sister pairs
(AB–AVC, OFT–RV) are preserved.

Restriction sequence (top → bottom):
  1. {AB,AVC,Atria}  |  {LV,OFT,RV}            (first split)
  2. {AB,AVC} | Atria   AND   LV | {OFT,RV}     (second split)
  3. AB | AVC          AND   OFT | RV           (final fate commitment)

Output: ../figures/Figure4d.png
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
OUT_PNG = os.path.join(_HERE, "..", "figures", "Figure4d.png")

CLUSTER_PALETTE = {
    1: "#CC79A7",
    2: "#E69F00",
    3: "#0088FF",
}

# ── Leaf positions  (x = horizontal fate layout, y fixed at bottom) ───────────
# Leaf order left→right: junctional clade first (Atria, AB, AVC),
# then right-ventricular clade (LV, RV, OFT). AB-AVC and RV-OFT stay
# adjacent so the FDR-significant arcs sit between sisters.
LEAVES = {
    "Atria": dict(x=0.08, y=0.14, cluster=1, median=8),
    "AB":    dict(x=0.22, y=0.14, cluster=2, median=4),
    "AVC":   dict(x=0.36, y=0.14, cluster=2, median=6),
    "LV":    dict(x=0.55, y=0.14, cluster=3, median=8),
    "RV":    dict(x=0.74, y=0.14, cluster=3, median=5),
    "OFT":   dict(x=0.88, y=0.14, cluster=1, median=6),
}

# ── Internal nodes  (x = midpoint of children, y = restriction depth) ─────────
# y is set to match Figure 4c's restriction-depth scale, so the two
# cladograms can sit side-by-side with bifurcation nodes at directly
# comparable y-positions (root y=0.90, mid level y=0.56, late level y=0.38).
# N_RIGHT (median=24) sits between Fig 4c's N_PROX (38, y=0.74) and
# N_PROX2 (20, y=0.56) at y=0.65.
NODES = {
    "N_JUNC":  dict(x=0.29,  y=0.38),   # AB | AVC                    (median 14)
    "N_DIST":  dict(x=0.81,  y=0.38),   # OFT | RV                    (median 18)
    "N_LEFT":  dict(x=0.185, y=0.56),   # (AB+AVC) | Atria            (median 20)
    "N_RIGHT": dict(x=0.68,  y=0.65),   # LV | (OFT+RV)               (median 24)
    "Root":    dict(x=0.43,  y=0.90),   # left clade | right clade    (median 85)
}

# Tree topology: c1=left child, c2=right child
TREE = {
    "Root":    dict(c1="N_LEFT",  c2="N_RIGHT"),
    "N_LEFT":  dict(c1="N_JUNC",  c2="Atria"),
    "N_JUNC":  dict(c1="AB",      c2="AVC"),
    "N_RIGHT": dict(c1="LV",      c2="N_DIST"),
    "N_DIST":  dict(c1="RV",      c2="OFT"),
}

NODE_R   = 0.058
INT_R    = 0.036
LINE_COL = "#BBBBBB"
LINE_LW  = 1.4

# ── Median clone size at each bifurcation node ──
# Strictly hierarchical: each node uses ONLY clones within its own subtree.
# Root:    5- and 6-fate clones (n=5, most multipotent)                → 85 cells
# N_LEFT:  Atria + (AB or AVC), no LV/OFT/RV (n=5)                     → 20 cells
# N_JUNC:  pure AB+AVC clones (n=4)                                    → 14 cells
# N_RIGHT: LV + (OFT or RV), no AB/AVC/Atria (n=3)                     → 24 cells
# N_DIST:  pure OFT+RV clones (n=10)                                   → 18 cells
NODE_CLONE_DATA = {
    "Root":    dict(median=85, n=5),
    "N_LEFT":  dict(median=20, n=5),
    "N_JUNC":  dict(median=14, n=4),
    "N_RIGHT": dict(median=24, n=3),
    "N_DIST":  dict(median=18, n=10),
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

    ax.plot([x_lo, x_hi], [yN, yN], color=LINE_COL, lw=LINE_LW,
            solid_capstyle="round", zorder=1)
    for xC, yC in [(xC1, yC1), (xC2, yC2)]:
        ax.plot([xC, xC], [yN, yC], color=LINE_COL, lw=LINE_LW,
                solid_capstyle="round", zorder=1)

    _draw_tree(ax, c1)
    _draw_tree(ax, c2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.13, 4.5), dpi=300)
    ax.set_xlim(-0.28, 1.08)
    ax.set_ylim(-0.04, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Background zones (no big "PROXIMAL/DISTAL" labels — the Rank-3
    #    split groups LV with the OFT/RV clade, which is not the standard
    #    proximal/distal anatomical division). Vertical extent matches
    #    Figure 4c so the two panels read at the same depth scale. ──────────
    # Left clade: Atria, AB, AVC
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.06, 0.02), 0.50, 0.95,
        boxstyle="round,pad=0.005",
        facecolor="#F8F8F8", edgecolor="none", alpha=0.85, zorder=0))
    # Right clade: LV, RV, OFT
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.46, 0.02), 0.58, 0.95,
        boxstyle="round,pad=0.005",
        facecolor="#EEEEFF", edgecolor="none", alpha=0.85, zorder=0))
    # Boundary dashed line
    ax.plot([0.45, 0.45], [0.02, 0.97], color="#CCCCDD",
            lw=0.8, linestyle="--", zorder=0)

    # Zone labels — identical style/colors to Figure 4c (PROXIMAL/DISTAL),
    # just shifted to the Rank-3 clade midpoints.
    ax.text(0.19, 0.99, "PROXIMAL", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#777777")
    ax.text(0.72, 0.99, "DISTAL",   ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#7777AA")

    # ── Restriction arrow ────────────────────────────────────────────────────
    ax.annotate("", xy=(-0.10, 0.12), xytext=(-0.10, 0.92),
                arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=1.0))
    ax.text(-0.13, 0.52, "Restriction sequence",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#888888", rotation=90)

    # ── Clone-size / stage gradient bar ──────────────────────────────────────
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

    # Bracket the transition zone — same y as Figure 4c (between
    # N_PROX y=0.74 and N_PROX2 y=0.56) so the gradient bars align.
    for y_dash in [0.74, 0.56]:
        ax.plot([bx0 - 0.005, bx1 + 0.005], [y_dash, y_dash],
                color="#BBBBBB", lw=0.6, linestyle="--", zorder=1)

    ax.text((bx0 + bx1) / 2, by_top + 0.01, "Large clones",
            ha="center", va="bottom", fontsize=6.5, color="#B8602A",
            fontweight="bold")
    ax.text((bx0 + bx1) / 2, by_bot - 0.01, "Small clones",
            ha="center", va="top", fontsize=6.5, color="#2A6AB8",
            fontweight="bold")

    # ── Cladogram lines ──────────────────────────────────────────────────────
    _draw_tree(ax, "Root")

    # ── Co-occurrence arcs (below leaf row) ──────────────────────────────────
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

    # ── Internal nodes ───────────────────────────────────────────────────────
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

    # ── Leaf nodes ───────────────────────────────────────────────────────────
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

    # ── Root node ────────────────────────────────────────────────────────────
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

    # ── Legend ───────────────────────────────────────────────────────────────
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
