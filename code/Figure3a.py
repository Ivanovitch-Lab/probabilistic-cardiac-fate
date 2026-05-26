#!/usr/bin/env python3
"""
Figure3a.py
===========
Reproduces Figure 3a:
hierarchical clustering of intermediate potency-state fate spectra
(dendrogram + heatmap). The 22 multi-region states with 2–5 regions are
clustered with Ward linkage on Euclidean distances; the k = 3 cut is
shown as a coloured strip alongside the heatmap.

This script also writes the supplementary tables used by Figure3b
and Figure3c:
  Supplementary_Table_S3.csv — potency graph edges
  Supplementary_Table_S4.csv — per-state fate bias + k=3 cluster

Input
-----
../data/Supplementary_Table_S1.csv

Outputs
-------
../figures/Figure3a.png
../data/Supplementary_Table_S3.csv
../data/Supplementary_Table_S4.csv

Run
---
    python Figure3a.py
"""

import os
import sys
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist


REGIONS     = ["OFT", "RV", "LV", "AVC", "AB", "Atria"]
REGION_COLS = ["R1_OFT", "R2_RV", "R3_LV", "R4_AVC", "R5_AB", "R6_Atria"]
MAX_SIZE    = 92

CLUSTER_PALETTE = {1: "#CC79A7", 2: "#E69F00", 3: "#0088FF"}
CLUSTER_LABELS  = {1: "Atria/OFT", 2: "AVC/AB", 3: "LV/RV"}
FATE_TO_CLUSTER = {
    "Atria": 1, "OFT": 1,
    "AVC":   2, "AB":  2,
    "LV":    3, "RV":  3,
}

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_HERE, "..", "data", "Supplementary_Table_S1.csv")
OUT_PNG    = os.path.join(_HERE, "..", "figures", "Figure3a.png")
OUT_S4_CSV = os.path.join(_HERE, "..", "data", "Supplementary_Table_S3.csv")
OUT_S5_CSV = os.path.join(_HERE, "..", "data", "Supplementary_Table_S4.csv")


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


# ─── Graph construction ─────────────────────────────────────────────────────
def build_potency_graph(clones):
    """Build the potency graph from the clone-by-region table.

    Edge rule: regions(B) ⊊ regions(A), |regions(A)|−|regions(B)| ∈ {1, 2},
    and mean_size(A) > mean_size(B).
    """
    clones = clones.copy()
    clones["pattern_canon"] = clones["leaf_regions"].apply(canon)
    grouped = clones.groupby("pattern_canon").agg(
        n=("size", "size"), mean_size=("size", "mean")
    ).reset_index()

    nodes = {}
    for _, r in grouped.iterrows():
        nodes[r["pattern_canon"]] = {
            "regions":   to_region_set(r["pattern_canon"]),
            "n":         int(r["n"]),
            "mean_size": float(r["mean_size"]),
            "n_regions": n_regions(r["pattern_canon"]),
        }

    G = nx.DiGraph()
    for name, attrs in nodes.items():
        G.add_node(name, **attrs)

    rows = []
    states = list(nodes.keys())
    for a, b in combinations(states, 2):
        A, B = nodes[a], nodes[b]
        for parent, child in [(A, B), (B, A)]:
            if child["regions"] < parent["regions"]:
                lost = len(parent["regions"]) - len(child["regions"])
                if lost in (1, 2) and parent["mean_size"] > child["mean_size"]:
                    p_name = canon(parent["regions"])
                    c_name = canon(child["regions"])
                    G.add_edge(p_name, c_name)
                    rows.append({
                        "parent_regions":   p_name,
                        "parent_n":         parent["n"],
                        "parent_mean_size": parent["mean_size"],
                        "child_regions":    c_name,
                        "child_n":          child["n"],
                        "child_mean_size":  child["mean_size"],
                        "delta_size":       parent["mean_size"] - child["mean_size"],
                        "regions_lost":     lost,
                    })

    edges_df = pd.DataFrame(rows).sort_values(
        ["parent_regions", "child_regions"]).reset_index(drop=True)
    return G, edges_df


# ─── Path enumeration & fate spectra ────────────────────────────────────────
def enumerate_paths(G):
    sources = [n for n, d in G.in_degree()  if d == 0]
    sinks   = [n for n, d in G.out_degree() if d == 0]
    paths = []
    for s in sources:
        for t in sinks:
            paths.extend(list(nx.all_simple_paths(G, s, t)))
    return paths


def compute_fate_spectra(G, paths):
    state_terminal_counts = defaultdict(lambda: {f: 0 for f in REGIONS})
    for path in paths:
        terminal_fate = list(to_region_set(path[-1]))[0]
        for state in path:
            state_terminal_counts[state][terminal_fate] += 1

    rows = []
    for state, counts in state_terminal_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        rec = {
            "node_regions": state,
            "n_regions":    G.nodes[state]["n_regions"],
            "n_clones":     G.nodes[state]["n"],
            "mean_size":    G.nodes[state]["mean_size"],
            "n_paths_through": total,
        }
        for f in REGIONS:
            rec[f"frac_{f}"] = counts[f] / total
        rows.append(rec)
    spectra = pd.DataFrame(rows)
    fcols = [f"frac_{f}" for f in REGIONS]
    spectra["dominant_fate"] = spectra[fcols].idxmax(axis=1).str.replace("frac_", "")
    spectra["dominant_frac"] = spectra[fcols].max(axis=1)
    return spectra


# ─── Clustering (Ward k=3, with canonical relabelling) ──────────────────────
def cluster_intermediates(spectra):
    inter = spectra[(spectra["n_regions"] >= 2) &
                    (spectra["n_regions"] <= 5)].copy().reset_index(drop=True)
    fcols = [f"frac_{f}" for f in REGIONS]
    X = inter[fcols].values
    Z = linkage(pdist(X, metric="euclidean"), method="ward")
    raw = fcluster(Z, t=3, criterion="maxclust")
    inter["raw_cluster"] = raw

    # Relabel raw clusters to {1: Atria/OFT, 2: AVC/AB, 3: LV/RV} by
    # majority-vote of dominant-fate → cluster mapping
    score = defaultdict(lambda: {1: 0, 2: 0, 3: 0})
    for _, r in inter.iterrows():
        score[int(r["raw_cluster"])][FATE_TO_CLUSTER[r["dominant_fate"]]] += 1
    raw_sizes = inter.groupby("raw_cluster").size().sort_values(ascending=False).index
    raw_to_canon = {}
    used = set()
    for raw_id in raw_sizes:
        for canon_id in sorted(score[int(raw_id)], key=lambda k: (-score[int(raw_id)][k], k)):
            if canon_id not in used:
                raw_to_canon[int(raw_id)] = canon_id
                used.add(canon_id)
                break

    inter["cluster_k3"] = inter["raw_cluster"].map(raw_to_canon)
    inter = inter.drop(columns=["raw_cluster"])
    return inter, Z


# ─── Plotting ───────────────────────────────────────────────────────────────
def plot(inter, Z, out_png):
    fcols = [f"frac_{f}" for f in REGIONS]
    n = len(inter)

    # Sized to sit beside Figure 3b in a full-A4-width composite.
    # Both panels are at their natural cell aspect — they will be aligned
    # by vertical centre in the layout step (Illustrator).
    fig = plt.figure(figsize=(5.5, 5.0), dpi=300)
    gs = fig.add_gridspec(1, 4,
        width_ratios=[1.6, 0.08, 3.5, 0.15], wspace=0.05)
    ax_dendro  = fig.add_subplot(gs[0, 0])
    ax_strip   = fig.add_subplot(gs[0, 1])
    ax_heat    = fig.add_subplot(gs[0, 2])
    ax_cb_room = fig.add_subplot(gs[0, 3])
    ax_cb_room.axis("off")

    labels = [
        f"{r.dominant_fate} ({r.dominant_frac:.2f}) | {r.node_regions}"
        for _, r in inter.iterrows()
    ]
    dendro = dendrogram(
        Z, labels=labels, leaf_font_size=7.0,
        color_threshold=0, above_threshold_color="black",
        orientation="right", ax=ax_dendro,
    )
    leaf_order = dendro["leaves"]
    ax_dendro.set_xlabel("Distance", fontsize=9)
    ax_dendro.tick_params(axis="y", labelsize=7.0, pad=1, length=0)
    ax_dendro.tick_params(axis="x", labelsize=8)
    ymin, ymax = ax_dendro.get_ylim(); ax_dendro.set_ylim(ymax, ymin)
    xmin, xmax = ax_dendro.get_xlim(); ax_dendro.set_xlim(xmax, xmin)

    inter_ord = inter.iloc[leaf_order].reset_index(drop=True)

    # Cluster strip
    strip = np.array(
        [list(mcolors.to_rgba(CLUSTER_PALETTE[int(c)]))
         for c in inter_ord["cluster_k3"]]).reshape(-1, 1, 4)
    ax_strip.imshow(strip, aspect="auto")
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    # Heatmap
    mat = inter_ord[fcols].values
    im = ax_heat.imshow(mat, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    ax_heat.set_yticks(range(len(inter_ord)))
    ax_heat.set_yticklabels([""] * len(inter_ord))
    ax_heat.set_xticks(range(len(REGIONS)))
    ax_heat.set_xticklabels(REGIONS, fontsize=9, rotation=35, ha="right")

    # Colorbar
    cb = fig.colorbar(im, ax=ax_cb_room, fraction=0.85, pad=0.02)
    cb.set_label("Fraction of descendant paths", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Cluster legend in a single column, placed outside the right edge of
    # the figure so it does not overlap the colorbar/heatmap.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CLUSTER_PALETTE[c], edgecolor="white",
                     label=CLUSTER_LABELS[c]) for c in [3, 2, 1]]
    fig.legend(handles=handles, loc="center left",
               bbox_to_anchor=(1.02, 0.5),
               ncol=1, fontsize=8.5, frameon=False,
               title="k = 3 cluster", title_fontsize=9)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FIGURE 3a — Ward k=3 clustering of intermediate fate spectra")
    print("=" * 70)
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: cannot find {INPUT_FILE}")
    clones = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    clones = clones[clones["clone_id"].notna()].copy()
    clones = clones[clones["size"] <= MAX_SIZE].copy()
    print(f"Loaded {len(clones)} clones (excluding the 142-cell clone)")

    G, edges_df = build_potency_graph(clones)
    print(f"Potency graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    edges_df.to_csv(OUT_S4_CSV, index=False)
    print(f"  ✓ Wrote: {OUT_S4_CSV}  ({len(edges_df)} edges)")

    paths = enumerate_paths(G)
    full_root = next(
        (n for n in G.nodes if G.nodes[n]["n_regions"] == 6), None)
    n_from_root = sum(1 for p in paths if p[0] == full_root) if full_root else 0
    print(f"Paths: {len(paths)} total  "
          f"({n_from_root} from the fully multipotent root, "
          f"{len(paths) - n_from_root} from partial-restriction sources)")

    spectra = compute_fate_spectra(G, paths)
    inter, Z = cluster_intermediates(spectra)
    print(f"Clustered intermediates: {len(inter)}")
    for c in sorted(inter["cluster_k3"].unique()):
        sub = inter[inter["cluster_k3"] == c]
        print(f"  cluster_k3={c} ({CLUSTER_LABELS[c]:>10}):  {len(sub)} states")

    # Supp Table S5 = full per-state summary (all 29 nodes) with cluster IDs
    # filled only for the 22 clustered intermediates
    fcols = [f"frac_{f}" for f in REGIONS]
    s5 = spectra[["node_regions", "n_regions", "n_clones", "mean_size",
                  "n_paths_through", "dominant_fate", "dominant_frac"] + fcols].copy()
    s5 = s5.merge(inter[["node_regions", "cluster_k3"]],
                  on="node_regions", how="left")
    s5["cluster_k3"] = s5["cluster_k3"].astype("Int64")
    s5.to_csv(OUT_S5_CSV, index=False)
    print(f"  ✓ Wrote: {OUT_S5_CSV}  ({len(s5)} states)")

    plot(inter, Z, OUT_PNG)
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
