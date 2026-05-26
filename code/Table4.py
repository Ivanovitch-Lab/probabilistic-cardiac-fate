#!/usr/bin/env python3
"""
Table4.py
=========
Reproduces Table 4 of Ivanovitch (BioEssays 2026):
per-topology summary for the five surviving restriction-sequence
topologies shown in Figure 4d, ranked by joint score (product of path
scores across the six terminals; path score = product of edge supports
along the path).

One row per topology. Columns:
  - #                    topology rank (1 = best-supported)
  - log score            natural-log joint score of the topology
  - Rel. score           joint score as % of the best topology
  - per-terminal margin  median across the six terminals of
                         score(rank-1 path) / score(rank-2 path)
  - Δ next topology      ratio of this topology's joint score to the
                         joint score of the next-ranked topology
  - n combos             number of distinct path-combination tuples
                         collapsing to this same topology signature

The five topologies span a small range of joint score — the table
gives the exact numbers behind the manuscript claim "all within X pp of
the best".

Inputs
------
../data/Supplementary_Table_S1.csv       Meilhac clone-by-region matrix
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
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import compute_edge_supports, build_graph
from _sequence_utils import derive_restriction_sequence
from _topology_utils import (
    signature,
    all_paths_per_terminal, top_k_combinations,
    _build_children, _is_strictly_bifurcating,
    _all_internal_nodes_supported, _is_median_monotonic,
    load_clone_regions,
)


_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, "..", "figures", "Table4.png")

TOP_K = 10000


def main():
    edge_prob = compute_edge_supports()
    G         = build_graph(edge_prob)
    paths_per_term  = all_paths_per_terminal(G)
    combos, terms   = top_k_combinations(paths_per_term, TOP_K)
    clones_df       = load_clone_regions()
    n_paths         = {t: len(paths_per_term[t]) for t in terms}

    # Group path combinations by topology signature
    by_sig     = defaultdict(list)
    sig_to_seq = {}
    for idx, lp in combos:
        cp  = {t: paths_per_term[t][idx[k]][0] for k, t in enumerate(terms)}
        seq = derive_restriction_sequence(cp)
        sg  = signature(seq)
        by_sig[sg].append({"idx": idx, "lp": lp})
        if sg not in sig_to_seq:
            sig_to_seq[sg] = seq

    # Sort topology signatures by best joint log-score
    sigs_sorted = sorted(
        by_sig.keys(),
        key=lambda s: -max(m["lp"] for m in by_sig[s]),
    )

    # Apply the three biological filters
    sigs_filtered = []
    for s in sigs_sorted:
        co = _build_children(sig_to_seq[s])
        rt = frozenset(sig_to_seq[s][0]["fates_before_split"])
        if not _is_strictly_bifurcating(co):
            continue
        if not _all_internal_nodes_supported(co, rt, clones_df):
            continue
        if not _is_median_monotonic(co, rt, clones_df):
            continue
        sigs_filtered.append(s)

    print(f"{len(sigs_filtered)} topologies passed all filters")

    # Build per-topology rows
    log_ps  = [max(m["lp"] for m in by_sig[s]) for s in sigs_filtered]
    best_lp = log_ps[0]
    rows = []
    for i, s in enumerate(sigs_filtered):
        members = sorted(by_sig[s], key=lambda m: -m["lp"])
        rep     = members[0]
        rep_idx = rep["idx"]

        # Per-terminal margin: median of (rank-1 path P) / (rank-2 path P)
        margins = []
        for k, t in enumerate(terms):
            ip = rep_idx[k]
            p_here = paths_per_term[t][ip][1]
            p_next = (paths_per_term[t][ip + 1][1]
                      if ip + 1 < n_paths[t] else 0.0)
            if p_next > 0:
                margins.append(p_here / p_next)
        margin_med = float(np.median(margins)) if margins else float("nan")

        # Inter-topology margin: joint P of this topology / next topology
        next_lp = log_ps[i + 1] if i + 1 < len(log_ps) else None
        margin_to_next = (float(np.exp(rep["lp"] - next_lp))
                          if next_lp is not None else float("inf"))

        rows.append({
            "topology":             i + 1,
            "joint_log_prob":       rep["lp"],
            "relative_to_best":     float(np.exp(rep["lp"] - best_lp)),
            "margin_median_per_terminal": margin_med,
            "margin_to_next_topology":    margin_to_next,
            "n_path_combinations":  len(members),
        })

    # ── Render as table figure
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=200)
    ax.axis("off")

    col_labels = [
        "#",
        "log score",
        "Rel. score",
        "per-terminal margin",
        "Δ next topology",
        "n combos",
    ]
    col_widths = [0.05, 0.13, 0.12, 0.26, 0.18, 0.12]
    cell_text = []
    for r in rows:
        cell_text.append([
            f"{r['topology']}",
            f"{r['joint_log_prob']:.3f}",
            f"{r['relative_to_best']:.0%}",
            f"{r['margin_median_per_terminal']:.2f}×",
            (f"{r['margin_to_next_topology']:.2f}×"
             if r["margin_to_next_topology"] != float("inf") else "—"),
            f"{r['n_path_combinations']}",
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
    table.set_fontsize(8.5)
    table.scale(1.0, 1.4)
    for i in range(1, len(cell_text) + 1):
        if i % 2 == 0:
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor("#F8F8F8")
    for j in range(len(col_labels)):
        table[(0, j)].get_text().set_fontweight("bold")

    rng_pp = (rows[0]["relative_to_best"] - rows[-1]["relative_to_best"]) * 100
    fig.suptitle(
        f"All {len(rows)} plausible restriction-sequence topologies — "
        f"all within {rng_pp:.0f} pp of the best",
        fontsize=10, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Saved: {OUT_PNG}")

    # Print the table to console for easy copy-paste
    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
