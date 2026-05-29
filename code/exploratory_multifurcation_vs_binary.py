#!/usr/bin/env python3
"""
exploratory_multifurcation_vs_binary.py
========================================
EXPLORATORY — not a manuscript figure / not part of the paper.

Question: among clone-supported restriction-sequence topologies, are
MULTIFURCATING shapes (a >=3-way split somewhere) more plausible than
strictly-BINARY ones?

This scans the full ~21M path-combination space tracking EVERY distinct
topology (not just the binary ones the manuscript keeps), keeps those where
every internal node has >=1 assigned clone (the SAME support filter applied
to binary and multifurcating alike, for a fair comparison), and plots each
surviving topology along ONE plausibility axis, Figure-3d style, coloured
by class:

  best-realization score, relative to overall best
    For each topology T:
        best_realization(T) = score(T) / score_overall_best
    where
        score(T)            = joint score of the SINGLE BEST 6-path
                              combination that produces topology T
        score_overall_best  = joint score of the rank-1 combination across
                              all ~21M (= every terminal's rank-1 path)

    Many combinations can collapse to the same topology T when the
       6 paths are overlaid (different per-terminal path choices, same
       fate-partition tree); score(T) keeps only the best-scoring one of
       those — hence "single best path-combination producing the tree."

  By construction the global-best combination has best_realization = 1.0;
  every other topology sits between 0 and 1. The global best is a binary
  tree (so binary reaches 1.0); the best multifurcating topology comes
  essentially to the same point — they're effectively tied at the top.

Output: figures/exploratory/EXPLORATORY_multifurcation_vs_binary.png
Independent of the manuscript pipeline; not wired into reproduce.sh.
"""

import math
import os
import sys
import time
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import compute_edge_supports, build_graph
from _sequence_utils import derive_restriction_sequence
from _topology_utils import (
    all_paths_per_terminal, fast_signature, load_clone_regions,
    _build_children, _all_internal_nodes_supported,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "figures", "exploratory")
OUT_PNG = os.path.join(OUT_DIR, "EXPLORATORY_multifurcation_vs_binary.png")

C_BINARY = "#4C72B0"   # blue
C_MULTI = "#DD8452"    # orange


def scan_all_topologies(G, progress=True):
    """Full 21M scan tracking EVERY distinct topology (binary + multi).
    Returns terms, node_paths_t, total, total_mass, global_best_lp, and
    per-signature best (lp, idx), mass (sum exp lp), count (n combos)."""
    ppt = all_paths_per_terminal(G)
    terms = sorted(ppt.keys())
    node_paths_t = [[tuple(p) for p, _ in ppt[t]] for t in terms]
    logsc_t = [np.array([np.log(max(s, 1e-30)) for _, s in ppt[t]],
                        dtype=np.float64) for t in terms]
    ranges = [len(x) for x in node_paths_t]
    total = int(np.prod(ranges))

    best, mass, count = {}, {}, {}
    total_mass, global_best = 0.0, -np.inf
    l0, l1, l2, l3, l4, l5 = logsc_t
    np0, np1, np2, np3, np4, np5 = node_paths_t

    t0 = time.time()
    c = 0
    for i0, i1, i2, i3, i4, i5 in product(
            range(ranges[0]), range(ranges[1]), range(ranges[2]),
            range(ranges[3]), range(ranges[4]), range(ranges[5])):
        lp = l0[i0] + l1[i1] + l2[i2] + l3[i3] + l4[i4] + l5[i5]
        e = math.exp(lp)
        total_mass += e
        if lp > global_best:
            global_best = lp
        sig = fast_signature(
            [np0[i0], np1[i1], np2[i2], np3[i3], np4[i4], np5[i5]])
        prev = best.get(sig)
        if prev is None or lp > prev[0]:
            best[sig] = (lp, (i0, i1, i2, i3, i4, i5))
        mass[sig] = mass.get(sig, 0.0) + e
        count[sig] = count.get(sig, 0) + 1
        c += 1
        if progress and c % 2_000_000 == 0:
            print(f"    {c:,}/{total:,} ({100*c/total:4.1f}%) "
                  f"{time.time()-t0:5.1f}s  distinct={len(mass):,}")

    return (terms, node_paths_t, total, total_mass, global_best,
            best, mass, count)


def main():
    G = build_graph(compute_edge_supports())
    clones_df = load_clone_regions()
    print("Scanning all topologies over the full space ...")
    (terms, node_paths_t, total, total_mass, global_best,
     best, mass, count) = scan_all_topologies(G)

    # Classify + apply the clone-support filter to ALL shapes alike.
    rows = []
    for sig, (lp, idx) in best.items():
        combo_paths = {terms[k]: list(node_paths_t[k][idx[k]])
                       for k in range(len(terms))}
        seq = derive_restriction_sequence(combo_paths)
        children_of = _build_children(seq)
        root = frozenset(seq[0]["fates_before_split"])
        if not _all_internal_nodes_supported(children_of, root, clones_df):
            continue
        arities = [len(kids) for kids in children_of.values()]
        rows.append({
            "weight": mass[sig] / total_mass,
            "peak_rel": math.exp(lp - global_best),
            "is_binary": all(a == 2 for a in arities),
            "max_arity": max(arities),
            "n_combos": count[sig],
        })

    binary = [r for r in rows if r["is_binary"]]
    multi = [r for r in rows if not r["is_binary"]]
    w_bin = sum(r["weight"] for r in binary)
    w_mul = sum(r["weight"] for r in multi)
    top = max(rows, key=lambda r: r["peak_rel"])
    print(f"\nClone-supported topologies: {len(rows)} "
          f"({len(binary)} binary, {len(multi)} multifurcating).")
    print(f"Marginal support: binary {w_bin:.2%}, multifurcating {w_mul:.2%} "
          f"(of all {total:,} combos).")
    print(f"Single most-probable (MAP) topology is "
          f"{'BINARY' if top['is_binary'] else 'MULTIFURCATING'} "
          f"(max arity {top['max_arity']}).")
    if multi:
        best_multi_peak = max(r["peak_rel"] for r in multi)
        best_bin_peak = max(r["peak_rel"] for r in binary) if binary else 0
        print(f"Best peak relative likelihood: binary {best_bin_peak:.3f}, "
              f"multifurcating {best_multi_peak:.3f}.")

    # ── Compact column-width strip plot (~1/3 A4 width, Figure-3d style) ──
    fig, ax = plt.subplots(figsize=(2.7, 2.3), dpi=300)
    rng = np.random.default_rng(0)
    Y = {"binary": 1.0, "multi": 0.0}
    for grp, yc, col in [(binary, Y["binary"], C_BINARY),
                         (multi, Y["multi"], C_MULTI)]:
        xs = np.array([r["peak_rel"] for r in grp])
        ys = yc + rng.uniform(-0.18, 0.18, size=len(xs))
        ax.scatter(xs, ys, s=5, c=col, alpha=0.75,
                   edgecolors="white", linewidths=0.2, zorder=3)

    ax.axvline(1.0, color="#999999", lw=0.5, ls="--", zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel(
        "best-realization score\n(relative to overall best = 1.0)",
        fontsize=6)
    ax.tick_params(axis="x", labelsize=5, pad=1.5, length=2.5, width=0.5)
    ax.tick_params(axis="y", length=0, pad=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    fig.text(0.01, 0.995, "exploratory", ha="left", va="top",
             fontsize=5, style="italic", color="#999999")

    leg_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C_BINARY, markeredgecolor="white",
                   markersize=3, label=f"binary (n={len(binary)})"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C_MULTI, markeredgecolor="white",
                   markersize=3,
                   label=f"multifurcating (n={len(multi)})"),
    ]
    ax.legend(handles=leg_handles, fontsize=5, frameon=False,
              loc="upper left", bbox_to_anchor=(0.02, 0.99),
              handletextpad=0.25, borderpad=0.0, labelspacing=0.2)

    plt.tight_layout(pad=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
