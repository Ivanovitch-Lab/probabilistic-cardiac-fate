#!/usr/bin/env python3
"""
Figure3c.py
===========
Reproduces Figure 3c of Ivanovitch (BioEssays 2026):
the full lineage-potency hierarchy as a directed acyclic graph, with the
29 potency states arranged into three k=3 cluster bands (LV/RV-biased on
the left, AVC/AB-biased in the centre, Atria/OFT-biased on the right) and
the fully multipotent root at top centre. Edge width and opacity scale
with the edge support s(A → B) (= fraction of clones compatible with
state A that are also compatible with the step A → B; not a probability,
since supports leaving a state can sum to more than 1).

Inputs
------
../data/Supplementary_Table_S1.csv     raw Meilhac clones (for edge supports)
../data/Supplementary_Table_S3.csv     potency graph edges (from Figure3a)
../data/Supplementary_Table_S4.csv     per-state cluster IDs   (from Figure3a)

Output
------
../figures/Figure3c.png

Run
---
    python Figure3c.py

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
TERMINAL_SPACING = 1000    # spacing between terminal nodes in a band

_HERE        = os.path.dirname(os.path.abspath(__file__))
CLONES_CSV   = os.path.join(_HERE, "..", "data", "Supplementary_Table_S1.csv")
EDGES_CSV    = os.path.join(_HERE, "..", "data", "Supplementary_Table_S3.csv")
CLUSTERS_CSV = os.path.join(_HERE, "..", "data", "Supplementary_Table_S4.csv")
OUT_PNG      = os.path.join(_HERE, "..", "figures", "Figure3c.png")


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


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FIGURE 3c — lineage-potency hierarchy DAG (k=3 cluster bands)")
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
    p_min = min(edge_p.values()) if edge_p else 0.0
    p_max = max(edge_p.values()) if edge_p else 1.0
    print(f"  p_min = {p_min:.3f},  p_max = {p_max:.3f}")

    pos = layout(G)

    # Node visual encoding (matches script 9)
    node_colors, node_alphas, node_sizes = [], [], []
    for n in G.nodes:
        d = G.nodes[n]
        node_colors.append(node_color(d))
        nr = d["n_regions"]
        if nr == 1:
            node_alphas.append(1.0)
        elif nr >= 6:
            node_alphas.append(0.4)
        else:
            node_alphas.append(1.0 - (nr - 1) * (1.0 - 0.4) / 5)
        node_sizes.append(min(200.0 + 5.0 * d["mean_size"], 600.0))

    def _w(p): return 0.2 + 3.3 * p
    def _a(p): return 0.10 + 0.85 * p

    # Full A4 width (8.27 in); height bumped from script 9's 4.6 in to give
    # the now-wider bands more vertical breathing room
    fig, ax = plt.subplots(figsize=(8.27, 6.0), dpi=300)

    # Edges
    for u, v in G.edges():
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
            arrows=True, arrowsize=8, arrowstyle="-|>",
            node_size=node_sizes,
            connectionstyle="arc3,rad=0.05", ax=ax,
        )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors="white", linewidths=1.5, alpha=node_alphas, ax=ax,
    )

    # Labels below each node (white-boxed) — formula and "+" separator
    # ported from script 9
    for node, (x, y) in pos.items():
        label = format_node_label(node)
        sz_lbl = 300.0 + 8.0 * G.nodes[node]["mean_size"]
        offset = np.sqrt(sz_lbl / np.pi) * 11 + 40
        ax.text(x, y - offset, label,
                ha="center", va="top",
                fontsize=5.5, fontweight="bold", color="black",
                bbox=dict(facecolor="white", alpha=0.6,
                          edgecolor="none", pad=0.3))

    # Legend (same encoding as the published version)
    handles = [
        Patch(facecolor="#CCCCCC", edgecolor="white", lw=1, label="Common progenitor"),
        Patch(facecolor=CLUSTER_PALETTE[3], edgecolor="white", lw=1, label="LV/RV-biased"),
        Patch(facecolor=CLUSTER_PALETTE[2], edgecolor="white", lw=1, label="AVC/AB-biased"),
        Patch(facecolor=CLUSTER_PALETTE[1], edgecolor="white", lw=1, label="Atria/OFT-biased"),
        Line2D([0], [0], color="#B7B7B7", lw=1.5, alpha=0.65, label="Between bias classes"),
        Line2D([0], [0], color="gray", lw=_w(p_min), alpha=_a(p_min),
               label=f"Least supported edge (s = {p_min:.2f})"),
        Line2D([0], [0], color="gray", lw=_w(p_max), alpha=_a(p_max),
               label=f"Most supported edge (s = {p_max:.2f})"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0),
              fontsize=6, frameon=False, borderaxespad=0.0)

    ax.set_axis_off()
    ax.margins(0.05)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
