#!/usr/bin/env python3
"""
Table4.py
=========
Reproduces Table 4 of the accompanying manuscript: a per-topology summary
for the biologically-plausible restriction-sequence topologies drawn in
Figure 4a, ranked by joint score (product of path scores across the six
terminals; path score = product of edge supports along the path).

Built from the SAME exhaustive enumeration as Figure 4a
(_topology_utils.enumerate_full_space) so the two are always consistent —
all topologies meeting the three biological filters (strictly bifurcating,
every bifurcation clone-supported, monotone median clone size) over the
full ~21M-combination space, not a top-K shortlist.

One row per topology. Columns:
  - #                    topology rank (1 = best-supported)
  - log score            natural-log joint score of the topology
  - Rel. score           joint score as % of the best topology
  - support              marginal support weight (sum of exp(joint score)
                         over all combos realizing it / over all combos)
  - per-terminal margin  median across the six terminals of
                         score(chosen path) / score(next-ranked path)
  - Δ next topology      ratio of this topology's joint score to the
                         next-ranked topology's joint score
  - n combos             number of distinct path-combination tuples (over
                         the full space) collapsing to this topology

Inputs
------
../data/Supplementary_Table_S1.csv       clone-by-region matrix
../data/Supplementary_Table_S3.csv       potency graph edges
../data/clone_path_attachments.csv       clones propagated along paths

Output
------
../figures/Table4.png

Run
---
    python Table4.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import compute_edge_supports, build_graph
from _topology_utils import (
    all_paths_per_terminal, enumerate_full_space, load_clone_regions,
)


_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, "..", "figures", "Table4.png")


def main():
    G = build_graph(compute_edge_supports())
    clones_df = load_clone_regions()
    paths_per_term = all_paths_per_terminal(G)

    print("Enumerating full space (no top-K cutoff) ...")
    res = enumerate_full_space(G, clones_df, progress=True)
    terms = res["terms"]
    survivors = res["survivors"]          # sorted by best_lp desc
    f = res["funnel"]
    n_paths = {t: len(paths_per_term[t]) for t in terms}
    print(f"\nFunnel: {f['distinct']} distinct -> {f['bifurcating']} bifurcating "
          f"-> {f['supported']} supported -> {f['monotone']} monotone-median.")

    best_lp = survivors[0]["best_lp"]
    rows = []
    for i, e in enumerate(survivors):
        idx = e["best_idx"]
        # Per-terminal margin: median of score(chosen path) / score(next path)
        margins = []
        for k, t in enumerate(terms):
            ip = idx[k]
            p_here = paths_per_term[t][ip][1]
            p_next = (paths_per_term[t][ip + 1][1]
                      if ip + 1 < n_paths[t] else 0.0)
            if p_next > 0:
                margins.append(p_here / p_next)
        margin_med = float(np.median(margins)) if margins else float("nan")

        next_lp = survivors[i + 1]["best_lp"] if i + 1 < len(survivors) else None
        margin_to_next = (float(np.exp(e["best_lp"] - next_lp))
                          if next_lp is not None else float("inf"))

        rows.append({
            "topology":              i + 1,
            "joint_log_prob":        e["best_lp"],
            "relative_to_best":      float(np.exp(e["best_lp"] - best_lp)),
            "support_weight":        e["weight"],
            "margin_median_per_terminal": margin_med,
            "margin_to_next_topology":    margin_to_next,
            "n_path_combinations":   e["n_combos"],
        })

    # ── Render as a table figure ────────────────────────────────────────
    col_labels = [
        "#", "log score", "Rel. score", "support",
        "per-terminal margin", "Δ next topo", "n combos",
    ]
    col_widths = [0.05, 0.12, 0.11, 0.11, 0.23, 0.14, 0.12]
    cell_text = []
    for r in rows:
        cell_text.append([
            f"{r['topology']}",
            f"{r['joint_log_prob']:.3f}",
            f"{r['relative_to_best']:.0%}",
            f"{r['support_weight']:.3%}",
            f"{r['margin_median_per_terminal']:.2f}×",
            (f"{r['margin_to_next_topology']:.2f}×"
             if r["margin_to_next_topology"] != float("inf") else "—"),
            f"{r['n_path_combinations']:,}",
        ])

    fig_h = max(3.2, 0.30 * len(rows) + 1.3)
    fig, ax = plt.subplots(figsize=(7.6, fig_h), dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
        colColours=["#EEEEEE"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    for i in range(1, len(cell_text) + 1):
        if i % 2 == 0:
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor("#F8F8F8")
    for j in range(len(col_labels)):
        table[(0, j)].get_text().set_fontweight("bold")

    fig.suptitle(
        f"All {len(rows)} biologically-plausible restriction-sequence "
        f"topologies\n(full enumeration of {res['total']:,} path "
        f"combinations; cf. Figure 4a)",
        fontsize=10, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Saved: {OUT_PNG}")

    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
