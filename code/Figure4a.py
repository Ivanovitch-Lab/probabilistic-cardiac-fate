#!/usr/bin/env python3
"""
Figure4a.py

Overview of every distinct strictly-bifurcating restriction-sequence
topology that survives all biological filters (strict bifurcation,
every bifurcation clone-supported, monotone median clone size). Each
plausible topology is rendered as a small cladogram, sharing the visual
style used by Figure 4b (rank-1) and Figure 4c (rank-3). Together with
Table 4 this panel summarises the inference landscape — Figure 4b and
4c then zoom into individual topologies, and Figure 4d shows the
graph-level path realization.
"""

import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import (
    compute_edge_supports, build_graph,
)
from _sequence_utils import derive_restriction_sequence
from _topology_utils import (
    signature,
    all_paths_per_terminal, top_k_combinations,
    _build_children, _is_strictly_bifurcating,
    _all_internal_nodes_supported, _is_median_monotonic,
    load_clone_regions, compute_lca_medians,
    draw_clado,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, "..", "figures", "Figure4a.png")

# Search depth in joint-score combinations. After biological filters
# (strict bifurcation, every bifurcation supported, monotone median) only a
# handful survive; the figure renders all of them.
TOP_K = 10000
MAX_DRAW = 20  # safety cap; assert below if filtered count ever exceeds this
N_COLS = 6     # grid is 6 wide on A4 portrait; rows scale with count
DPI_OUT = 200


def main():
    edge_prob = compute_edge_supports()
    G = build_graph(edge_prob)

    paths_per_term = all_paths_per_terminal(G)
    combos, terms = top_k_combinations(paths_per_term, TOP_K)

    print("=== Edge-condition check ===")
    print(f"  G has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"  All paths from nx.all_simple_paths — guaranteed valid edges.")
    clones_df = load_clone_regions()
    print(f"  Loaded {len(clones_df)} clones for size annotation.")

    by_sig = defaultdict(list)
    sig_to_seq = {}
    for idx, lp in combos:
        combo_paths = {t: paths_per_term[t][idx[k]][0]
                       for k, t in enumerate(terms)}
        seq = derive_restriction_sequence(combo_paths)
        sig = signature(seq)
        by_sig[sig].append({"idx": idx, "lp": lp})
        if sig not in sig_to_seq:
            sig_to_seq[sig] = seq

    sigs_sorted = sorted(by_sig.keys(),
                         key=lambda s: -max(m["lp"] for m in by_sig[s]))
    sigs_bif = [s for s in sigs_sorted
                if _is_strictly_bifurcating(_build_children(sig_to_seq[s]))]
    sigs_supp = [
        s for s in sigs_bif
        if _all_internal_nodes_supported(
            _build_children(sig_to_seq[s]),
            frozenset(sig_to_seq[s][0]["fates_before_split"]),
            clones_df,
        )
    ]
    sigs_filtered = [
        s for s in sigs_supp
        if _is_median_monotonic(
            _build_children(sig_to_seq[s]),
            frozenset(sig_to_seq[s][0]["fates_before_split"]),
            clones_df,
        )
    ]
    print(f"\n{len(sigs_sorted)} distinct signatures →")
    print(f"  {len(sigs_bif)} strictly bifurcating →")
    print(f"  {len(sigs_supp)} with every bifurcation supported (n>0) →")
    print(f"  {len(sigs_filtered)} with monotone median clone size.")

    assert len(sigs_filtered) <= MAX_DRAW, (
        f"{len(sigs_filtered)} signatures pass all filters but layout caps at "
        f"{MAX_DRAW}. Either widen the cap or tighten the filters."
    )
    n_show = len(sigs_filtered)
    top_sigs = sigs_filtered
    n_paths = {t: len(paths_per_term[t]) for t in terms}

    # Layout: pack the panels into a roughly-square grid, never wider than
    # N_COLS. For small counts (≤ N_COLS) we use a single row.
    if n_show <= N_COLS:
        n_cols, n_rows = n_show, 1
    else:
        n_cols, n_rows = N_COLS, math.ceil(n_show / N_COLS)
    panel_w = 9.5 / N_COLS  # keep per-panel width consistent with prior 12-panel layout
    fig_w = max(4.5, panel_w * n_cols)
    # Single-row figures need extra vertical room so the legend below doesn't
    # collide with the bottom-row terminal labels.
    fig_h = 1.8 * n_rows + (1.4 if n_rows == 1 else 0.7)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h), dpi=DPI_OUT)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, sig in enumerate(top_sigs):
        ax = axes_flat[i]
        members = sorted(by_sig[sig], key=lambda m: -m["lp"])
        rep = members[0]
        rep_idx = rep["idx"]

        margins = []
        for k, t in enumerate(terms):
            i_path = rep_idx[k]
            p_here = paths_per_term[t][i_path][1]
            p_next = (paths_per_term[t][i_path + 1][1]
                      if i_path + 1 < n_paths[t] else 0.0)
            if p_next > 0:
                margins.append(p_here / p_next)
        margin_med = float(np.median(margins)) if margins else 1.0

        sig_id = f"S{i + 1}"
        ranks_str = ", ".join(
            f"{terms[k]}:{rep_idx[k] + 1}" for k in range(len(terms)))
        title = f"{sig_id}   {ranks_str}"
        draw_clado(ax, sig_to_seq[sig], title=title,
                   n_combos=len(members), log_p=rep["lp"],
                   margin=margin_med, clones_df=clones_df)

    # Hide unused axes in the last row
    for ax in axes_flat[n_show:]:
        ax.axis("off")

    fig.suptitle(
        f"All {n_show} biologically-plausible restriction-sequence topologies\n"
        "(strictly bifurcating, every bifurcation clone-supported, "
        "monotone median clone size)",
        fontsize=10, fontweight="bold", y=1.0,
    )
    # Legend matching restriction_cladogram.py for visual continuity.
    from _topology_utils import (
        CLUSTER_PALETTE, ARC_COLOR,
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
               label="Significant co-occurrence  (FDR < 0.05)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#DDDDDD", markeredgecolor="#BBBBBB",
               markersize=7,
               label="Bifurcation node  (label = median clone size)"),
    ]
    legend_y = 0.04 if n_rows == 1 else 0.02
    rect_bottom = 0.20 if n_rows == 1 else 0.05
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, legend_y), ncol=3,
               fontsize=7, frameon=False, handletextpad=0.5,
               columnspacing=1.6, labelspacing=0.4)
    plt.tight_layout(pad=0.6, rect=(0, rect_bottom, 1, 0.92))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=DPI_OUT, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
