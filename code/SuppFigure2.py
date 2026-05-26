#!/usr/bin/env python3
"""
SuppFigure2.py
===========
Reproduces Supplementary Figure 2 of Ivanovitch (BioEssays 2026):
the per-terminal backbone paths through the potency graph implied by the
rank-1 (left) and rank-3 (right) plausible restriction-sequence topologies
shown in Figure 4b and Figure 4c. Two side-by-side panels share the same
k=3 cluster-band layout as Figure 3c, and within each panel only the
backbone nodes and edges of the chosen topology are rendered.

The two topologies differ at exactly one branching decision: where LV
splits off — in rank 1 LV joins the junctional pool (left panel), in
rank 3 LV joins the OFT/RV pool (right panel). The contrast makes the
graph-level path difference visible at a glance.

Inputs
------
../data/Supplementary_Table_S1.csv     raw Meilhac clones (for edge supports)
../data/Supplementary_Table_S3.csv     potency graph edges (from Figure3a)
../data/Supplementary_Table_S4.csv     per-state cluster IDs   (from Figure3a)

Output
------
../figures/SuppFigure2.png

Run
---
    python SuppFigure2.py

Requires pydot + graphviz `dot`. On macOS:
    brew install graphviz && pip install pydot
"""

import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.colors as mcolors
from networkx.drawing.nx_pydot import graphviz_layout


REGIONS  = ["OFT", "RV", "LV", "AVC", "AB", "Atria"]
MAX_SIZE = 92

# Layout parameters (ported from 9.k3_clustered_hierarchy_viz.py to match the
# published Figure 3c band geometry exactly)
CLUSTER_PALETTE = {1: "#CC79A7", 2: "#E69F00", 3: "#0088FF"}
CLUSTER_LABELS  = {1: "Atria/OFT-biased", 2: "AVC/AB-biased", 3: "LV/RV-biased"}
FATE_TO_CLUSTER = {
    "Atria": 1, "OFT": 1,
    "AVC":   2, "AB":  2,
    "LV":    3, "RV":  3,
}
TERMINAL_FATE_ORDER = {"LV": 1, "RV": 2, "OFT": 3, "AVC": 4, "AB": 5, "Atria": 6}

Y_SPACING = 700.0     # vertical gap between rows of states by region count

# Band x-ranges (left → right): LV/RV, AVC/AB, Atria/OFT.
# Spread further apart from script 9 defaults so the LV/RV band fits 8
# intermediates (including {Atria, LV}) and the whole figure reads
# comfortably at full A4 print width without feeling crowded.
BAND_RANGES = {
    3: (-4500, -2000),    # LV/RV (left)
    2: ( -700,   700),    # AVC/AB (centre)
    1: ( 2000,  5800),    # Atria/OFT (right) — widest, hosts 10 states
    "NA": (0, 0),
}
ROOT_X = -100              # multipotent root x position (top centre)
TERMINAL_SPACING = 1500    # spacing between terminal nodes in a band

_HERE        = os.path.dirname(os.path.abspath(__file__))
CLONES_CSV   = os.path.join(_HERE, "..", "data", "Supplementary_Table_S1.csv")
EDGES_CSV    = os.path.join(_HERE, "..", "data", "Supplementary_Table_S3.csv")
CLUSTERS_CSV = os.path.join(_HERE, "..", "data", "Supplementary_Table_S4.csv")
OUT_PNG      = os.path.join(_HERE, "..", "figures", "SuppFigure2.png")


# ─── Helpers ────────────────────────────────────────────────────────────────
def canon(regions):
    if isinstance(regions, str):
        s = regions.strip("{}'").replace("'", "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = list(regions)
    return "{" + ", ".join(sorted(parts)) + "}"


def n_regions(canon_str):
    s = canon_str.strip("{}'").replace("'", "")
    return len([p for p in s.split(",") if p.strip()])


def to_region_set(canon_str):
    s = canon_str.strip("{}'").replace("'", "")
    return frozenset(p.strip() for p in s.split(",") if p.strip())


# ─── Visual helpers (ported from script 9) ──────────────────────────────────
def format_node_label(canon_str):
    """Format the region set as 'A+B+C' rather than '{A, B, C}'. The fully
    multipotent 6-region root keeps a fixed display order."""
    parts = [p.strip() for p in canon_str.strip("{}'").split(",") if p.strip()]
    if len(parts) == 6:
        return "OFT+LV+RV+AVC+AB+Atria"
    return "+".join(parts)


def intensity_color(base_color, score):
    """Scale a base color by intensity (0–1) — high score → saturated,
    low score → pastel. Below 0.30 is clamped to 0.20 intensity to keep
    weakly-biased states visibly tinted rather than white."""
    min_score, min_intensity = 0.30, 0.2
    score = float(score or 0.0)
    if score < min_score:
        intensity = min_intensity
    else:
        norm = max(0.0, min((score - min_score) / (1.0 - min_score), 1.0))
        intensity = min_intensity + norm * (1.0 - min_intensity)
    white = 1.0 - intensity
    c = mcolors.to_rgb(base_color)
    return tuple(x * (1 - white) + 1.0 * white for x in c)


def node_color(d):
    """Multipotent root → lightgray. Terminals → full-saturation cluster
    color. Intermediates → intensity-scaled by dominant_frac."""
    nr = d["n_regions"]
    if nr >= 6:
        return "lightgray"
    band = d.get("cluster_band")
    if band not in CLUSTER_PALETTE:
        return "lightgray"
    base = CLUSTER_PALETTE[band]
    if nr == 1:
        return base
    return intensity_color(base, d.get("dominant_frac", 0.0))


# ─── Graph construction (from S4 + S5) ──────────────────────────────────────
def load_graph(edges_df, clusters_df):
    G = nx.DiGraph()
    for _, r in clusters_df.iterrows():
        node = canon(r["node_regions"])
        cluster = (int(r["cluster_k3"]) if not pd.isna(r["cluster_k3"]) else None)
        G.add_node(node,
                   n_regions=int(r["n_regions"]),
                   n=int(r["n_clones"]),
                   mean_size=float(r["mean_size"]),
                   cluster_k3=cluster,
                   dominant_fate=str(r["dominant_fate"]),
                   dominant_frac=float(r["dominant_frac"]))
    for _, r in edges_df.iterrows():
        u = canon(r["parent_regions"])
        v = canon(r["child_regions"])
        G.add_edge(u, v)
    return G


# ─── Cluster-band propagation (port of script 9) ────────────────────────────
def propagate_cluster_bands(G, pos):
    """Assign per-node cluster_band. Intermediates get their cluster_k3;
    terminals inherit via FATE_TO_CLUSTER; remaining unassigned nodes (only
    the multipotent root in practice) stay as None."""
    # Step 1: direct from cluster_k3
    for n in G.nodes:
        raw = G.nodes[n].get("cluster_k3")
        G.nodes[n]["cluster_band"] = (
            int(raw) if raw in CLUSTER_PALETTE else None)

    # Step 2: terminals via dominant fate
    for n in G.nodes:
        if G.nodes[n]["cluster_band"] in CLUSTER_PALETTE:
            continue
        fate = G.nodes[n].get("dominant_fate", "")
        if fate in FATE_TO_CLUSTER:
            G.nodes[n]["cluster_band"] = FATE_TO_CLUSTER[fate]

    # Step 3: top → bottom propagation if any remain unassigned
    nodes_sorted = sorted(G.nodes, key=lambda n: pos[n][1], reverse=True)
    for n in nodes_sorted:
        for child in G.successors(n):
            if G.nodes[child]["cluster_band"] in CLUSTER_PALETTE:
                continue
            parent_bands = {
                G.nodes[p]["cluster_band"] for p in G.predecessors(child)
                if G.nodes[p]["cluster_band"] in CLUSTER_PALETTE
            }
            if len(parent_bands) == 1:
                G.nodes[child]["cluster_band"] = next(iter(parent_bands))


# ─── Band-based layout (port of script 9 main) ──────────────────────────────
def layout(G):
    pos = graphviz_layout(G, prog="dot")

    # Force y by region count so layers line up cleanly
    for n in G.nodes:
        pos[n] = (pos[n][0], G.nodes[n]["n_regions"] * Y_SPACING)

    propagate_cluster_bands(G, pos)

    root_nodes = []
    nodes_by_band = {1: [], 2: [], 3: [], "NA": []}
    for n in G.nodes:
        if G.nodes[n]["n_regions"] >= 6:
            root_nodes.append(n)
            continue
        band = G.nodes[n]["cluster_band"]
        nodes_by_band[int(band) if band in CLUSTER_PALETTE else "NA"].append(n)

    for band, nodes in nodes_by_band.items():
        if not nodes:
            continue
        x_min, x_max = BAND_RANGES[band]

        # group by y-level (= region count)
        y_groups = defaultdict(list)
        for n in nodes:
            y_groups[pos[n][1]].append(n)

        for y, level_nodes in y_groups.items():
            num = len(level_nodes)
            is_terminal = (G.nodes[level_nodes[0]]["n_regions"] == 1)

            if num == 1:
                pos[level_nodes[0]] = ((x_min + x_max) / 2, y)
                continue

            if is_terminal:
                center_x = (x_min + x_max) / 2
                start_x  = center_x - (num - 1) * TERMINAL_SPACING / 2
                ordered  = sorted(
                    level_nodes,
                    key=lambda n: (
                        TERMINAL_FATE_ORDER.get(
                            G.nodes[n]["dominant_fate"], 99),
                        n,
                    ),
                )
                for i, n in enumerate(ordered):
                    pos[n] = (start_x + i * TERMINAL_SPACING, y)
            else:
                spacing = (x_max - x_min) / (num - 1)
                ordered = sorted(level_nodes, key=lambda n: n)
                for i, n in enumerate(ordered):
                    pos[n] = (x_min + i * spacing, y)

    # Multipotent root(s) at centre x
    if root_nodes:
        root_y = pos[root_nodes[0]][1]
        for rn in root_nodes:
            pos[rn] = (ROOT_X, root_y)

    return pos


# ─── Edge supports (clone propagation along paths) ──────────────────────────
def compute_edge_supports(G, clones):
    clones = clones.copy()
    clones["pattern_canon"] = clones["leaf_regions"].apply(canon)
    pattern_to_clones = clones.groupby(
        "pattern_canon")["clone_id"].apply(set).to_dict()

    sources = [n for n, d in G.in_degree()  if d == 0]
    sinks   = [n for n, d in G.out_degree() if d == 0]
    paths = []
    for s in sources:
        for t in sinks:
            paths.extend(list(nx.all_simple_paths(G, s, t)))

    node_clones = defaultdict(set)
    edge_clones = defaultdict(set)
    for path in paths:
        clones_on_path = set()
        for node in path:
            if node in pattern_to_clones:
                clones_on_path |= pattern_to_clones[node]
        if not clones_on_path:
            continue
        for node in path:
            node_clones[node] |= clones_on_path
        for u, v in zip(path[:-1], path[1:]):
            edge_clones[(u, v)] |= clones_on_path

    edge_p = {}
    for (u, v), cset in edge_clones.items():
        denom = len(node_clones[u])
        if denom > 0:
            edge_p[(u, v)] = len(cset) / denom
    return edge_p


# ─── Plausible backbones per terminal, for the rank-N plausible topology ────
def find_backbones(G, edge_p, rank=1):
    """For each terminal, return the path used by the rank-N PLAUSIBLE
    topology — the N-th highest-joint-score topology whose underlying
    paths pass the three biological filters (strictly bifurcating, every
    bifurcation clone-supported, monotone median clone size).

    rank=1 gives the topology shown in Figure 4b / panel 1 of Figure 4a.
    rank=3 gives the topology shown in Figure 4c / panel 3 of Figure 4a.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _topology_utils import (
        signature, all_paths_per_terminal, top_k_combinations,
        _build_children, _is_strictly_bifurcating,
        _all_internal_nodes_supported, _is_median_monotonic,
        load_clone_regions,
    )
    from _sequence_utils import derive_restriction_sequence

    root = next((n for n in G.nodes if G.nodes[n]["n_regions"] >= 6), None)
    if root is None:
        return {}

    # _topology_utils calls path_score(G, p) which reads G.edges[u, v]["p"]
    for (u, v), p in edge_p.items():
        if (u, v) in G.edges:
            G.edges[u, v]["p"] = p

    paths_per_term  = all_paths_per_terminal(G)
    combos, terms   = top_k_combinations(paths_per_term, 10000)
    clones_df       = load_clone_regions()

    # Group combos by topology signature
    by_sig = defaultdict(list)
    sig_to_seq = {}
    for idx, lp in combos:
        cp = {t: paths_per_term[t][idx[k]][0] for k, t in enumerate(terms)}
        seq = derive_restriction_sequence(cp)
        sg  = signature(seq)
        by_sig[sg].append({"idx": idx, "lp": lp, "cp": cp})
        if sg not in sig_to_seq:
            sig_to_seq[sg] = seq

    # Sort signatures by best joint log-score, then apply the three filters
    sigs_sorted = sorted(by_sig.keys(),
                         key=lambda s: -max(m["lp"] for m in by_sig[s]))
    sigs_plausible = []
    for s in sigs_sorted:
        co = _build_children(sig_to_seq[s])
        rt = frozenset(sig_to_seq[s][0]["fates_before_split"])
        if (_is_strictly_bifurcating(co)
            and _all_internal_nodes_supported(co, rt, clones_df)
            and _is_median_monotonic(co, rt, clones_df)):
            sigs_plausible.append(s)

    if rank > len(sigs_plausible):
        return {}

    target_sig = sigs_plausible[rank - 1]
    best_combo = max(by_sig[target_sig], key=lambda m: m["lp"])
    cp = best_combo["cp"]

    # Re-key by terminal graph-node ("{AVC}") to match the rest of this
    # script's pipeline.
    terminals = [n for n, d in G.out_degree() if d == 0]
    label_to_node = {next(iter(set(n.strip("{}'").replace("'", "")
                                    .split(", ")))).strip(): n
                     for n in terminals}
    return {label_to_node[t]: cp[t] for t in terms}


# ─── Per-panel drawing helper ───────────────────────────────────────────────
def draw_backbone_panel(ax, G, pos, edge_p, backbones,
                        node_colors, node_alphas, node_sizes,
                        title=None):
    backbone_edges = set()
    backbone_nodes = set()
    for path in backbones.values():
        for u, v in zip(path[:-1], path[1:]):
            backbone_edges.add((u, v))
            backbone_nodes.add(u)
            backbone_nodes.add(v)
    # Always keep all six terminals so the bottom row reads consistently
    for n in G.nodes:
        if G.nodes[n]["n_regions"] == 1:
            backbone_nodes.add(n)

    def _w(p): return 0.15 + 1.6 * p
    def _a(p): return 0.15 + 0.80 * p

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) * 1.1, max(all_x) * 1.1)
    ax.set_ylim(min(all_y) - 500, max(all_y) + 500)

    sizes_for_arrow = [node_sizes[n] for n in G.nodes]
    for u, v in backbone_edges:
        p = edge_p.get((u, v), 0.0)
        u_band = G.nodes[u]["cluster_band"]
        v_band = G.nodes[v]["cluster_band"]
        same   = (u_band in CLUSTER_PALETTE) and (u_band == v_band)
        if G.nodes[u]["n_regions"] >= 6:
            ec = "lightgray"
        elif same:
            ec = CLUSTER_PALETTE[u_band]
        else:
            ec = "#B7B7B7"
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=ec, width=_w(p), alpha=_a(p),
            arrows=True, arrowsize=4, arrowstyle="-|>",
            node_size=sizes_for_arrow,
            connectionstyle="arc3,rad=0.05", ax=ax,
        )

    for n in backbone_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[n],
            node_color=[node_colors[n]], node_size=[node_sizes[n]],
            edgecolors="white", linewidths=0.7, alpha=node_alphas[n], ax=ax,
        )

    for node in backbone_nodes:
        if G.nodes[node]["n_regions"] != 1:
            continue
        x, y = pos[node]
        label = format_node_label(node)
        ax.text(x, y - 220, label,
                ha="center", va="top",
                fontsize=7, fontweight="bold", color="black")

    if title is not None:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    ax.set_axis_off()
    ax.margins(0.05)


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SUPP FIGURE 2 — backbones of rank-1 and rank-3 plausible topologies")
    print("=" * 70)
    for f in (CLONES_CSV, EDGES_CSV, CLUSTERS_CSV):
        if not os.path.exists(f):
            sys.exit(f"ERROR: cannot find {f}. Run Figure3a.py first.")

    clones = pd.read_csv(CLONES_CSV, encoding="utf-8-sig")
    clones = clones[clones["clone_id"].notna()].copy()
    clones = clones[clones["size"] <= MAX_SIZE].copy()
    edges_df    = pd.read_csv(EDGES_CSV)
    clusters_df = pd.read_csv(CLUSTERS_CSV)

    G = load_graph(edges_df, clusters_df)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    edge_p = compute_edge_supports(G, clones)
    print(f"Edge supports computed for {len(edge_p)} / {G.number_of_edges()} edges")

    pos = layout(G)

    # Node visual encoding (shared across both panels)
    node_colors, node_alphas, node_sizes = {}, {}, {}
    for n in G.nodes:
        d = G.nodes[n]
        node_colors[n] = node_color(d)
        nr = d["n_regions"]
        if nr == 1:
            node_alphas[n] = 1.0
        elif nr >= 6:
            node_alphas[n] = 0.4
        else:
            node_alphas[n] = 1.0 - (nr - 1) * (1.0 - 0.4) / 5
        node_sizes[n] = min(80.0 + 2.5 * d["mean_size"], 240.0)

    # Find backbones for the two topologies
    bb_rank1 = find_backbones(G, edge_p, rank=1)
    bb_rank3 = find_backbones(G, edge_p, rank=3)
    print(f"Rank-1 backbones: {len(bb_rank1)} terminals")
    print(f"Rank-3 backbones: {len(bb_rank3)} terminals")

    # Two side-by-side panels — same width as the original ~2.75 in panel
    # for each subplot → total figure width ~5.6 in.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5.6, 3.2), dpi=300)
    draw_backbone_panel(ax_l, G, pos, edge_p, bb_rank1,
                        node_colors, node_alphas, node_sizes,
                        title="Rank-1 plausible")
    draw_backbone_panel(ax_r, G, pos, edge_p, bb_rank3,
                        node_colors, node_alphas, node_sizes,
                        title="Rank-3 plausible")

    plt.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
