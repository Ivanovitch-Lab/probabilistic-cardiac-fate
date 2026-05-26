#!/usr/bin/env python3
"""
Table3.py
=========
Reproduces Table 3 of the accompanying manuscript:
per-terminal backbone justification — for each of the six terminal fates
this reports the number of root→terminal paths, the backbone (rank-1)
score, the runner-up score, total score mass over all paths, dominance
(= backbone / total mass) and margin (= backbone / runner-up). Scores
are products of edge supports along the path; they are not
probabilities and need not sum to one across paths.

The rendered table is the numerical companion to the strip plot in
Figure 4b: same per-terminal numbers, in tabular form.

Inputs
------
../data/Supplementary_Table_S3.csv       potency graph edges
../data/clone_path_attachments.csv       clones propagated along paths

Output
------
../figures/Table3.png

Run
---
    python Table3.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import (
    compute_edge_supports, build_graph, _count_regions,
    _terminal_label, path_score, EDGES_CSV, TREES_CSV,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, "..", "figures", "Table3.png")

OBLIGATE_THRESHOLD = 0.99   # Edge is "obligate" if its edge support ≥ this


def _all_paths_to(G, root, term):
    return list(nx.all_simple_paths(G, root, term))


def _summarise(G, root, term):
    """Per-terminal summary numbers (matches Figure4b._summarise)."""
    paths = _all_paths_to(G, root, term)
    if not paths:
        return None
    probs_full = np.array([path_score(G, p) for p in paths])
    order = np.argsort(probs_full)[::-1]
    backbone = paths[order[0]]
    backbone_edges = list(zip(backbone[:-1], backbone[1:]))
    backbone_p_each = [G.edges[u, v]["p"] for u, v in backbone_edges]
    bb_full = probs_full[order[0]]
    return {
        "terminal":          _terminal_label(term),
        "n_paths":           len(paths),
        "backbone_p_full":   bb_full,
        "second_p_full":     probs_full[order[1]] if len(order) > 1 else 0.0,
        "total_mass_full":   probs_full.sum(),
        "dominance_full":    bb_full / probs_full.sum() if probs_full.sum() > 0 else 0.0,
        "margin_full":       (bb_full / probs_full[order[1]]
                              if len(order) > 1 and probs_full[order[1]] > 0
                              else float("inf")),
        "n_backbone_steps":  len(backbone_edges),
        "n_obligate_steps":  sum(1 for p in backbone_p_each
                                 if p >= OBLIGATE_THRESHOLD),
    }


def plot_table(summaries):
    summaries = sorted(summaries, key=lambda s: -s["backbone_p_full"])
    fig, ax = plt.subplots(figsize=(7.4, 0.55 + 0.32 * len(summaries)),
                           dpi=200)
    ax.axis("off")

    col_labels = [
        "Terminal",
        "# paths",
        "Backbone score",
        "Runner-up score",
        "Total mass",
        "Dominance",
        "Margin",
    ]
    col_widths = [0.10, 0.09, 0.16, 0.17, 0.12, 0.13, 0.12]

    cell_text = []
    for s in summaries:
        margin = s["margin_full"]
        margin_str = f"{margin:.3f}×" if np.isfinite(margin) else "—"
        cell_text.append([
            s["terminal"],
            f"{s['n_paths']}",
            f"{s['backbone_p_full']:.3f}",
            f"{s['second_p_full']:.3f}",
            f"{s['total_mass_full']:.2f}",
            f"{s['dominance_full']:.1%}",
            margin_str,
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
        colColours=["#EEEEEE"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for i in range(1, len(cell_text) + 1):
        if i % 2 == 0:
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor("#F8F8F8")
    for j in range(len(col_labels)):
        table[(0, j)].get_text().set_fontweight("bold")

    fig.suptitle(
        "Per-terminal backbone justification — "
        "scores are products of edge supports along the path (full path)",
        fontsize=9, fontweight="bold", y=0.97,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Saved: {OUT_PNG}")


def main():
    if not os.path.exists(EDGES_CSV) or not os.path.exists(TREES_CSV):
        print("ERROR: required CSVs not found")
        sys.exit(1)

    edge_prob = compute_edge_supports()
    G = build_graph(edge_prob)
    roots = [n for n in G.nodes if _count_regions(n) >= 6]
    terminals = [n for n in G.nodes if _count_regions(n) == 1]
    root = roots[0]

    summaries = [s for s in
                 (_summarise(G, root, t)
                  for t in sorted(terminals, key=_terminal_label))
                 if s]
    plot_table(summaries)


if __name__ == "__main__":
    main()
