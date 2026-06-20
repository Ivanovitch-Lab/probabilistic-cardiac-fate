#!/usr/bin/env python3
"""
restriction_topology_helpers.py
================================

Helper module — no `__main__` entry point. Defines the shared analysis
machinery for restriction-sequence topology figures:

  - `signature`             : canonical hashable signature of a topology
                              (frozenset, order-invariant)
  - `top_k_combinations`    : best-first enumeration of joint-score
                              path combinations across the six terminals
  - `compute_lca_medians`   : per-node clone count + median size via the
                              hybrid root-rule + LCA-descent assignment
  - `_is_strictly_bifurcating`,
    `_all_internal_nodes_supported`,
    `_is_median_monotonic`  : the three biological filters
  - `draw_clado`            : the cladogram renderer used by the panels
  - `CLUSTER_PALETTE`, `ARC_COLOR`, `FATE_LABEL`, etc. : shared visual
                              constants matching restriction_cladogram.py

Imported by Figure4b.py (top-5 mini-grid), SuppFigure3.py (full 17-topology
overview), SuppFigure2.py
(rank-1 vs rank-3 backbone DAGs), and Table4.py (per-topology summary).
"""

import heapq
import math
import os
import sys
import time
from collections import defaultdict
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import (
    compute_edge_supports, build_graph, _count_regions,
    _terminal_label, path_score,
)
from _sequence_utils import derive_restriction_sequence


def signature(sequence):
    """Canonical hashable signature of a fate-partition tree, independent
    of which intermediate node names the backbones traversed *and*
    independent of the order in which `derive_restriction_sequence`
    happened to emit the splits (which depends on frozenset iteration
    order, i.e. hash randomisation, and therefore varies across Python
    invocations). Two sequences with the same signature produce the
    same hierarchical clustering of the six terminal fates."""
    return frozenset(
        (frozenset(step["fates_before_split"]),
         frozenset(frozenset(c["fates"]) for c in step["children"]))
        for step in sequence
    )

# ── Clone-region data (Supplementary Table S1.csv) for clone-size annotation
# Same source used by cooccurrence_analysis.py, kept consistent.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "data", "Supplementary_Table_S1.csv")
REGION_COLS   = ["R1_OFT", "R2_RV", "R3_LV", "R4_AVC", "R5_AB", "R6_Atria"]
REGION_LABELS = ["OFT",    "RV",   "LV",   "AVC",   "AB",   "Atria"]


def load_clone_regions():
    """Return DataFrame with columns clone_id, size, region_set (frozenset)."""
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    df = df[df["clone_id"].notna()].copy()
    for col in REGION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["region_set"] = df.apply(
        lambda r: frozenset(REGION_LABELS[i]
                            for i, c in enumerate(REGION_COLS) if r[c] > 0),
        axis=1,
    )
    return df[["clone_id", "size", "region_set"]].copy()


def compute_lca_medians(clones_df, children_of, root):
    """Assign each clone to a single node and return per-node medians.

    Hybrid rule (matches restriction_cladogram.py's canonical numbers):
      - Most-multipotent clones (|region_set| ≥ K−1, where K = root's
        fate count) → assigned to ROOT. This is canonical's "5+6-fate
        clones at root" rule.
      - Non-spanning clones (region_set ⊆ a single child's subtree)
        → assigned to their LCA via descent.
      - Spanning clones with fewer than K−1 fates → AMBIGUOUS, not
        assigned to any node (they sit between the root kids without
        being multipotent enough). Reported in the unassigned count.

    Each clone is counted at most once, so medians are monotone-or-equal
    with depth (parent ≥ child medians, modulo small-sample variability).

    Returns: ({node: (n_clones, median_size)}, n_unassigned)."""
    from collections import defaultdict
    K = len(root)
    node_sizes = defaultdict(list)
    n_unassigned = 0
    for _, clone in clones_df.iterrows():
        rs = clone["region_set"]
        if not rs:
            continue
        # Root rule: most-multipotent clones (|rs| >= K-1) → root
        if len(rs) >= K - 1:
            node_sizes[root].append(clone["size"])
            continue
        # Otherwise: LCA descent
        cur = root
        while True:
            kids = children_of.get(cur, [])
            descended = False
            for k in kids:
                if rs <= k:
                    cur = k
                    descended = True
                    break
            if not descended:
                break
        if cur == root:
            # Spanning clone with < K-1 fates — ambiguous, unassigned
            n_unassigned += 1
            continue
        node_sizes[cur].append(clone["size"])
    medians = {node: (len(sizes),
                      float(np.median(sizes)) if sizes else 0.0)
               for node, sizes in node_sizes.items()}
    return medians, n_unassigned


# ── Style constants — copied verbatim from restriction_cladogram.py ─────
CLUSTER_PALETTE = {
    1: "#CC79A7",   # Atria, OFT
    2: "#E69F00",   # AB, AVC
    3: "#0088FF",   # LV, RV
}
FATE_TO_CLUSTER = {
    "Atria": 1, "OFT":   1,
    "AB":    2, "AVC":   2,
    "LV":    3, "RV":    3,
}
# Single-letter abbreviations used inside leaf circles to keep panels small
# enough to pack many side-by-side.
FATE_LABEL = {
    "Atria": "A",
    "AB":    "B",
    "AVC":   "V",
    "LV":    "L",
    "RV":    "R",
    "OFT":   "O",
}
NODE_R   = 0.058
INT_R    = 0.036
LINE_COL = "#BBBBBB"
LINE_LW  = 1.4

# Co-occurrence pairs that pass FDR ≤ 0.05 (Sox2-Cre, small-clone bin),
# from cooccurrence_fdr_table.txt — drawn as arcs under the matching
# sister leaves in each cladogram (matches restriction_cladogram.py).
COOCCURRENCE_PAIRS = [
    ("AB", "AVC"),     # z=4.96, FDR=1.1×10⁻⁵
    ("OFT", "RV"),     # z=4.38, FDR=8.9×10⁻⁵
]
ARC_COLOR = "#333333"

def all_paths_per_terminal(G):
    roots = [n for n in G.nodes if _count_regions(n) >= 6]
    terms = [n for n in G.nodes if _count_regions(n) == 1]
    root = roots[0]
    out = {}
    for t in terms:
        paths = list(nx.all_simple_paths(G, root, t))
        ranked = sorted(
            ((p, path_score(G, p)) for p in paths),
            key=lambda x: (-x[1], tuple(sorted(map(str, x[0])))),
        )
        out[_terminal_label(t)] = ranked
    return out


def top_k_combinations(paths_per_term, K):
    terms = sorted(paths_per_term.keys())
    n_per_term = [len(paths_per_term[t]) for t in terms]

    def lp(idx):
        return sum(np.log(max(paths_per_term[t][i][1], 1e-30))
                   for t, i in zip(terms, idx))

    init = tuple([0] * len(terms))
    visited = {init}
    heap = [(-lp(init), init)]
    out = []
    while heap and len(out) < K:
        neg_lp, idx = heapq.heappop(heap)
        out.append((idx, -neg_lp))
        for k in range(len(terms)):
            new = list(idx)
            new[k] += 1
            if new[k] >= n_per_term[k]:
                continue
            new_t = tuple(new)
            if new_t in visited:
                continue
            visited.add(new_t)
            heapq.heappush(heap, (-lp(new_t), new_t))
    return out, terms


def fast_signature(paths):
    """Canonical topology signature from a list of node-paths (one per
    fate, indexed by fate-id 0..n-1). Equivalent to
    signature(derive_restriction_sequence(...)) up to the fixed
    fate-id<->fate-name relabelling, which preserves set equality, so it is
    a faithful key for de-duplicating topologies across the full
    combination space. Returns a frozenset of (frozenset(parent fate-ids),
    frozenset(child fate-id sets)) split events.

    This is the cheap inner loop used by enumerate_full_space; it avoids
    building the verbose restriction-sequence records for all 21M combos."""
    steps = []

    def recurse(fates, depth):
        groups = {}
        for f in fates:
            p = paths[f]
            key = p[depth] if depth < len(p) else ("LEAF", f)
            g = groups.get(key)
            if g is None:
                groups[key] = [f]
            else:
                g.append(f)
        if len(groups) == 1:
            only = next(iter(groups.values()))
            if len(only) > 1:
                recurse(only, depth + 1)
            return
        steps.append((frozenset(fates),
                      frozenset(frozenset(g) for g in groups.values())))
        for g in groups.values():
            if len(g) > 1:
                recurse(g, depth + 1)

    recurse(list(range(len(paths))), 0)
    return frozenset(steps)


def enumerate_full_space(G, clones_df, progress=False):
    """Exhaustively score EVERY root->terminal path combination across the
    six terminals (no top-K cutoff) and return the topologies that pass the
    biological filters, together with the marginal-support weight each one
    carries.

    Rationale: the three biological filters (strict bifurcation, every
    bifurcation clone-supported, monotone median clone size) depend only on
    the TOPOLOGY, not on the joint score. A topology can therefore pass the
    filters yet first appear far below any top-K search cutoff, so a top-K
    scan under-counts the surviving topologies. This function removes that
    artefact by visiting the entire space.

    The 'support weight' of a topology is sum(exp(joint log-score)) over all
    combinations realising it, divided by that sum over the whole space —
    i.e. its marginal posterior weight under the per-terminal-independent
    path model. (Path scores are edge-support products, not strict
    probabilities; see _graph_utils — so this is a relative support measure,
    not a calibrated probability.)

    Returns a dict:
      terms              : sorted terminal-fate labels (fate-id order)
      total              : total number of combinations scanned
      funnel             : {distinct, bifurcating, supported, monotone}
      survivors          : list (sorted by best_lp desc) of topologies
                           passing ALL filters; each is a dict with
                           sig, combo_paths (best path, for drawing),
                           best_idx, best_lp, weight, n_combos, combo_rank
      supported          : same, for the filter set with median DROPPED
                           (bifurcating + every node supported)
      weight_binary      : collective support weight of all bifurcating topos
      weight_supported   : ... of the clone-supported set
      weight_monotone    : ... of the full-filter (drawn) set
      weight_top5        : ... of the 5 highest-best_lp survivors
      map_is_binary      : whether the single highest-scoring combination
                           (the MAP reconstruction) is itself a binary tree
    """
    ppt = all_paths_per_terminal(G)
    terms = sorted(ppt.keys())
    node_paths_t = [[tuple(p) for p, _ in ppt[t]] for t in terms]
    logsc_t = [np.array([np.log(max(s, 1e-30)) for _, s in ppt[t]],
                        dtype=np.float64) for t in terms]
    ranges = [len(x) for x in node_paths_t]
    total = int(np.prod(ranges))

    all_lp = np.empty(total, dtype=np.float64)
    best = {}   # sig -> (best_lp, best_idx)
    mass = {}   # sig -> sum exp(lp) over its combos (bifurcating only)
    count = {}  # sig -> number of combos collapsing to it (bifurcating only)
    distinct = set()
    map_lp = -np.inf
    map_sig = None
    l0, l1, l2, l3, l4, l5 = logsc_t
    np0, np1, np2, np3, np4, np5 = node_paths_t

    t0 = time.time()
    c = 0
    for i0, i1, i2, i3, i4, i5 in product(
            range(ranges[0]), range(ranges[1]), range(ranges[2]),
            range(ranges[3]), range(ranges[4]), range(ranges[5])):
        lp = l0[i0] + l1[i1] + l2[i2] + l3[i3] + l4[i4] + l5[i5]
        all_lp[c] = lp
        sig = fast_signature(
            [np0[i0], np1[i1], np2[i2], np3[i3], np4[i4], np5[i5]])
        distinct.add(hash(sig))
        if lp > map_lp:
            map_lp, map_sig = lp, sig
        if all(len(child) == 2 for _, child in sig):
            prev = best.get(sig)
            if prev is None or lp > prev[0]:
                best[sig] = (lp, (i0, i1, i2, i3, i4, i5))
            mass[sig] = mass.get(sig, 0.0) + math.exp(lp)
            count[sig] = count.get(sig, 0) + 1
        c += 1
        if progress and c % 2_000_000 == 0:
            print(f"    {c:,}/{total:,} ({100*c/total:4.1f}%) "
                  f"{time.time()-t0:5.1f}s")

    total_mass = float(np.exp(all_lp).sum())
    asc = np.sort(all_lp)

    def make_entry(sig, lp, idx):
        combo_paths = {terms[k]: list(node_paths_t[k][idx[k]])
                       for k in range(len(terms))}
        return {
            "sig": sig,
            "combo_paths": combo_paths,
            "best_idx": idx,
            "best_lp": lp,
            "weight": mass[sig] / total_mass,
            "n_combos": count[sig],
            "combo_rank": total - int(np.searchsorted(asc, lp, "right")) + 1,
        }

    supported, survivors = [], []
    for sig, (lp, idx) in best.items():
        entry = make_entry(sig, lp, idx)
        seq = derive_restriction_sequence(entry["combo_paths"])
        children_of = _build_children(seq)
        root = frozenset(seq[0]["fates_before_split"])
        if not _is_strictly_bifurcating(children_of):
            continue  # should already hold; structural guard
        if not _all_internal_nodes_supported(children_of, root, clones_df):
            continue
        supported.append(entry)
        if _is_median_monotonic(children_of, root, clones_df):
            survivors.append(entry)

    supported.sort(key=lambda e: -e["best_lp"])
    survivors.sort(key=lambda e: -e["best_lp"])

    return {
        "terms": terms,
        "total": total,
        "funnel": {
            "distinct": len(distinct),
            "bifurcating": len(best),
            "supported": len(supported),
            "monotone": len(survivors),
        },
        "survivors": survivors,
        "supported": supported,
        "weight_binary": sum(mass.values()) / total_mass,
        "weight_supported": sum(e["weight"] for e in supported),
        "weight_monotone": sum(e["weight"] for e in survivors),
        "weight_top5": sum(e["weight"] for e in survivors[:5]),
        "map_is_binary": all(len(child) == 2 for _, child in map_sig),
    }


def _build_children(sequence):
    children_of = {}
    for step in sequence:
        parent = frozenset(step["fates_before_split"])
        children_of[parent] = [
            frozenset(c["fates"]) for c in step["children"]
        ]
    return children_of


def _is_strictly_bifurcating(children_of):
    """All non-leaf nodes have exactly 2 children — required for a clean
    binary cladogram with no multifurcations."""
    return all(len(kids) == 2 for kids in children_of.values())


def _all_internal_nodes_supported(children_of, root, clones_df):
    """Every internal node must have at least one clone assigned to it
    (n_clones > 0). A node with n=0 means no clone in the dataset
    supports that bifurcation — biologically unjustified."""
    medians, _ = compute_lca_medians(clones_df, children_of, root)
    for node in children_of.keys():
        n_cl, _ = medians.get(node, (0, 0.0))
        if n_cl == 0:
            return False
    return True


def _is_median_monotonic(children_of, root, clones_df):
    """Every parent's median clone size must be ≥ each of its (internal
    or leaf) children's medians. Otherwise the cladogram has a more
    multipotent ancestor with smaller clones than its restricted
    descendant, which is biologically implausible — usually a small-
    sample artefact."""
    medians, _ = compute_lca_medians(clones_df, children_of, root)
    for parent, kids in children_of.items():
        n_p, med_p = medians.get(parent, (0, 0.0))
        if n_p == 0:
            continue
        for k in kids:
            n_k, med_k = medians.get(k, (0, 0.0))
            if n_k > 0 and med_p < med_k:
                return False
    return True


def _layout(children_of, root):
    """Recursive cladogram layout. Returns leaf_order, leaf_x, int_x, int_y.

    Internal-node Y position is determined by SUBTREE SIZE (number of
    fates that remain at that node), NOT by bifurcation depth. This
    matches restriction_cladogram.py's hand-coded y-values: a 2-fate
    node sits at the same height as another 2-fate node, regardless of
    where in the tree it appears, so the median-clone-size annotations
    are visually consistent with restriction level."""
    leaves = []

    # Sort kids so visually-related fates (same cluster) end up adjacent.
    def sort_key(s):
        clusters = sorted({FATE_TO_CLUSTER[f] for f in s})
        return (clusters[0], len(s), sorted(s)[0])

    def in_order(node):
        """Recurse so the two FDR-significant pairs (OFT–RV, AB–AVC) are
        pulled toward each other across any subclade boundary, while
        non-anchor branches keep the default cluster-based ordering."""
        if len(node) == 1:
            leaves.append(next(iter(node)))
            return
        kids = children_of.get(node, [])

        has_oft = "OFT" in node
        has_rv  = "RV"  in node

        if has_oft and has_rv:
            oft_kid = next((k for k in kids if "OFT" in k), None)
            rv_kid  = next((k for k in kids if "RV"  in k), None)
            if oft_kid is rv_kid:
                # The OFT–RV LCA sits below this node — defer.
                kids = sorted(kids, key=sort_key)
            else:
                # This node IS the OFT–RV LCA: OFT-kid on the left, RV-kid
                # on the right, so the two leaves meet at the boundary.
                others = sorted(
                    (k for k in kids if k is not oft_kid and k is not rv_kid),
                    key=sort_key,
                )
                kids = [oft_kid] + others + [rv_kid]
        elif has_oft:
            # Push the OFT-containing child to the right edge of this
            # subclade so OFT ends up adjacent to the boundary with RV.
            oft_kid = next((k for k in kids if "OFT" in k), None)
            others = sorted(
                (k for k in kids if k is not oft_kid), key=sort_key)
            kids = others + [oft_kid]
        elif has_rv:
            # Push the RV-containing child to the left edge.
            rv_kid = next((k for k in kids if "RV" in k), None)
            others = sorted(
                (k for k in kids if k is not rv_kid), key=sort_key)
            kids = [rv_kid] + others
        else:
            kids = sorted(kids, key=sort_key)

        for k in kids:
            in_order(k)

    in_order(root)

    # Leaf x positions in [0.08, 0.92], evenly spaced
    n = len(leaves)
    if n > 1:
        leaf_x = {f: 0.08 + (0.92 - 0.08) * i / (n - 1)
                  for i, f in enumerate(leaves)}
    else:
        leaf_x = {leaves[0]: 0.5}

    # Y positions by subtree size — matches restriction_cladogram.py:
    #   1 fate (leaf): y = 0.18
    #   2 fates:        y = 0.38
    #   3 fates:        y = 0.56
    #   4 fates:        y = 0.74
    #   6 fates (root): y = 0.90
    Y_FOR_SIZE = {1: 0.18, 2: 0.38, 3: 0.56, 4: 0.74, 5: 0.82, 6: 0.90}
    int_y = {n_int: Y_FOR_SIZE.get(len(n_int), 0.5) for n_int in children_of}

    # Internal node x = midpoint of children's x
    int_x = {}

    def get_x(node):
        if len(node) == 1:
            return leaf_x[next(iter(node))]
        if node in int_x:
            return int_x[node]
        kids = children_of[node]
        xs = [get_x(k) for k in kids]
        int_x[node] = (min(xs) + max(xs)) / 2
        return int_x[node]

    for n_int in children_of:
        get_x(n_int)

    return leaves, leaf_x, int_x, int_y


def draw_clado(ax, sequence, title, n_combos, log_p, margin,
               clones_df=None, cooccurrence_sisters_only=False):
    """Draw one cladogram in restriction_cladogram.py style. If clones_df
    is provided, annotate each internal node with its median clone size
    (clones spanning that bifurcation) and each leaf with its single-fate
    median clone size.

    cooccurrence_sisters_only: if True, an FDR co-occurrence arc is drawn
    only when its two fates are direct sister leaves in THIS topology
    (i.e. frozenset({f1, f2}) is a bifurcation node). Default False keeps
    the original behaviour (arc drawn whenever both leaves are present),
    so manuscript figures are unaffected."""
    children_of = _build_children(sequence)
    if not children_of:
        ax.axis("off")
        return
    root = frozenset(sequence[0]["fates_before_split"])

    leaves, leaf_x, int_x, int_y = _layout(children_of, root)
    leaf_y = 0.18

    # Tree edges: horizontal crossbar at parent y, vertical drops to children
    for parent, kids in children_of.items():
        xP, yP = int_x[parent], int_y[parent]
        kid_xs = []
        for k in kids:
            if len(k) == 1:
                xK, yK = leaf_x[next(iter(k))], leaf_y
            else:
                xK, yK = int_x[k], int_y[k]
            kid_xs.append(xK)
            ax.plot([xK, xK], [yP, yK], color=LINE_COL, lw=LINE_LW,
                    solid_capstyle="round", zorder=1)
        ax.plot([min(kid_xs), max(kid_xs)], [yP, yP],
                color=LINE_COL, lw=LINE_LW,
                solid_capstyle="round", zorder=1)

    # LCA-based assignment: compute median per node ONCE for the whole tree
    if clones_df is not None:
        lca_medians, _n_unassigned = compute_lca_medians(
            clones_df, children_of, root)
    else:
        lca_medians = {}

    # Internal nodes — small grey circles, minimal label: just median or "—".
    # Detailed per-node n + median go to CSV (see export_node_csv).
    for n_int, kids in children_of.items():
        ax.add_patch(plt.Circle(
            (int_x[n_int], int_y[n_int]), INT_R,
            facecolor="#DDDDDD", edgecolor="white",
            linewidth=0.8, zorder=4))
        n_cl, med = lca_medians.get(n_int, (0, 0.0))
        label = f"{med:.0f}" if n_cl > 0 else "—"
        ax.text(int_x[n_int], int_y[n_int] - INT_R - 0.018,
                label,
                ha="center", va="top",
                fontsize=6, fontweight="bold",
                color="#777777" if n_cl > 0 else "#BBBBBB", zorder=5)

    # Leaves — coloured circles by cluster, white labels inside
    for f in leaves:
        c_id = FATE_TO_CLUSTER.get(f, 1)
        col = CLUSTER_PALETTE[c_id]
        ax.add_patch(plt.Circle(
            (leaf_x[f], leaf_y), NODE_R,
            facecolor=col, edgecolor="white", linewidth=1.4,
            alpha=0.92, zorder=5))
        ax.text(leaf_x[f], leaf_y, FATE_LABEL.get(f, f),
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", zorder=6)
        # Single-fate median clone size below the leaf
        leaf_node = frozenset({f})
        n_cl, med = lca_medians.get(leaf_node, (0, 0.0))
        label = f"{med:.0f}" if n_cl > 0 else "—"
        ax.text(leaf_x[f], leaf_y - NODE_R - 0.018,
                label,
                ha="center", va="top",
                fontsize=5.5, fontweight="bold",
                color="#777777" if n_cl > 0 else "#BBBBBB", zorder=6)

    # FDR-significant clonal pair arcs, drawn UNDER the leaf row.
    # Same neutral dark-grey arc styling as restriction_cladogram.py.
    for f1, f2 in COOCCURRENCE_PAIRS:
        if f1 in leaf_x and f2 in leaf_x:
            if cooccurrence_sisters_only and \
                    frozenset({f1, f2}) not in children_of:
                continue
            ax.add_patch(FancyArrowPatch(
                posA=(leaf_x[f1], leaf_y),
                posB=(leaf_x[f2], leaf_y),
                connectionstyle="arc3,rad=-0.55",
                arrowstyle="-",
                color=ARC_COLOR, linewidth=1.0, alpha=0.85, zorder=3,
            ))
    # No per-panel title — keeping the figure as compact as possible.

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.set_aspect("auto")
    ax.axis("off")

