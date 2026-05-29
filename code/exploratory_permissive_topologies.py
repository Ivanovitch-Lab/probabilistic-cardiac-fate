#!/usr/bin/env python3
"""
exploratory_permissive_topologies.py
====================================
EXPLORATORY — not a manuscript figure / not part of the paper.

Personal visualisation aid. Renders, on a single A4 page, every
strictly-bifurcating, clone-supported topology that survives the MOST
PERMISSIVE size rule (instead of the manuscript's monotone-median rule),
so the full breadth of the binary-tree landscape can be eyeballed. Kept in
the repo only for transparency; deliberately NOT wired into reproduce.sh,
and its output lands in figures/exploratory/.

Permissive size rule (per parent->child edge, child = node/leaf with >=1
assigned clone): there exists ANY clone at the parent STRICTLY larger than
ANY clone at the child  <=>  max(parent sizes) > min(child sizes). No
expected/absolute size; pure existence; strict. The weakest "sizes step
down somewhere along every lineage" criterion.

Funnel (all ~21.15M path combinations): 92 strictly bifurcating -> 50
clone-supported -> {17 monotone-median (manuscript Figure 4a), 40 under
this permissive rule}.

Co-occurrence arcs (AB-AVC, OFT-RV) are drawn only where the pair are
direct sister leaves in that topology (draw_clado cooccurrence_sisters_only).
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
    enumerate_full_space, load_clone_regions, _build_children, draw_clado,
    CLUSTER_PALETTE, ARC_COLOR,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "figures", "exploratory")
OUT_PNG = os.path.join(OUT_DIR, "EXPLORATORY_permissive_topologies_A4.png")
N_COLS = 6
DPI_OUT = 200
A4_PORTRAIT = (8.27, 11.69)  # inches


def _fmt_pct(w):
    return f"{w:.2%}" if w >= 5e-4 else "<0.05%"


def _size_lists(clones_df, children_of, root):
    """{node: [clone sizes]} via the SAME assignment as compute_lca_medians."""
    K = len(root)
    out = {}
    for _, clone in clones_df.iterrows():
        rs = clone["region_set"]
        if not rs:
            continue
        if len(rs) >= K - 1:
            out.setdefault(root, []).append(clone["size"]); continue
        cur = root
        while True:
            descended = False
            for k in children_of.get(cur, []):
                if rs <= k:
                    cur = k; descended = True; break
            if not descended:
                break
        if cur == root:
            continue
        out.setdefault(cur, []).append(clone["size"])
    return out


def _passes_permissive(children_of, root, clones_df):
    sl = _size_lists(clones_df, children_of, root)
    for parent, kids in children_of.items():
        p = sl.get(parent, [])
        if not p:
            return False
        for k in kids:
            c = sl.get(k, [])
            if not c:
                continue
            if not (max(p) > min(c)):   # any parent clone strictly > any child clone
                return False
    return True


def _draw_all(entries, out_png):
    n = len(entries)
    n_cols = N_COLS
    n_rows = math.ceil(n / n_cols)
    lp_top = entries[0]["best_lp"]
    clones_df = load_clone_regions()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=A4_PORTRAIT, dpi=DPI_OUT)
    axes_flat = np.atleast_1d(axes).flatten()
    for i, e in enumerate(entries):
        ax = axes_flat[i]
        seq = derive_restriction_sequence(e["combo_paths"])
        draw_clado(ax, seq, title="", n_combos=1, log_p=e["best_lp"],
                   margin=1.0, clones_df=clones_df,
                   cooccurrence_sisters_only=True)
        rel = math.exp(e["best_lp"] - lp_top)
        ax.text(0.5, 1.01, f"S{e['rank']}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", transform=ax.transAxes,
                color="#333333")
        ax.text(0.5, 1.0, f"L={rel:.2f}  w={_fmt_pct(e['weight'])}",
                ha="center", va="top", fontsize=5, transform=ax.transAxes,
                color="#999999")
    for ax in axes_flat[n:]:
        ax.axis("off")

    # Discreet, non-intrusive label (no red banner).
    fig.text(0.01, 0.997, "exploratory — not a manuscript figure",
             ha="left", va="top", fontsize=7, style="italic", color="#999999")
    fig.suptitle(
        f"Binary topologies under the most-permissive size rule "
        f"(any parent clone > any child clone, strict)\n"
        f"{n} of 50 clone-supported trees  ·  manuscript Figure 4a keeps only "
        f"the 17 monotone-median ones\n"
        f"(L = likelihood relative to top tree; w = marginal support weight)",
        fontsize=8.5, y=0.992,
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
               label="Co-occurrence (FDR<0.05), drawn only when sisters"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DDDDDD",
               markeredgecolor="#BBBBBB", markersize=7,
               label="Bifurcation node  (label = median clone size)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=7, frameon=False,
               handletextpad=0.5, columnspacing=1.4, labelspacing=0.4)
    plt.tight_layout(pad=0.5, rect=(0, 0.035, 1, 0.95))
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_png, dpi=DPI_OUT, facecolor="white")
    plt.close()
    print(f"  saved {out_png}")


def main():
    G = build_graph(compute_edge_supports())
    clones_df = load_clone_regions()
    print("Enumerating full space ...")
    res = enumerate_full_space(G, clones_df, progress=True)
    supported = res["supported"]
    f = res["funnel"]
    print(f"\nFunnel: {f['bifurcating']} bifurcating -> {f['supported']} "
          f"supported -> {f['monotone']} monotone-median (manuscript).")

    permissive = []
    for e in supported:
        seq = derive_restriction_sequence(e["combo_paths"])
        ch = _build_children(seq)
        root = frozenset(seq[0]["fates_before_split"])
        if _passes_permissive(ch, root, clones_df):
            permissive.append(e)
    permissive.sort(key=lambda e: -e["best_lp"])
    for i, e in enumerate(permissive, 1):
        e["rank"] = i
    wt = sum(e["weight"] for e in permissive)
    print(f"{len(permissive)} / {len(supported)} pass the permissive rule "
          f"(support weight {wt:.2%}).")

    if permissive:
        print("\nRendering single A4 EXPLORATORY figure ...")
        _draw_all(permissive, OUT_PNG)


if __name__ == "__main__":
    main()
