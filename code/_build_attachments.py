#!/usr/bin/env python3
"""
_build_attachments.py
=====================
Helper script — regenerates `clone_path_attachments.csv` (S7 in the
original numbering) from the canonical inputs S1 (Meilhac clones) and S3
(potency graph edges).

Each clone is attached to its home node — the graph node whose regional
pattern equals the clone's pattern — and then propagated along every
simple root-to-terminal path passing through that node. The resulting
table has one row per (path, node, clone) triple and is the source of
node_clones(N) and edge_clones(A → B) computed in `_graph_utils.py`.

This is `_attach_clones_to_lineage_treesV2.py` with the dead
`unmatched_patterns` branch removed: because the graph is built from the
observed clone patterns, every clone matches a node by construction.

Run
---
    python _build_attachments.py
"""

import os
import sys
from collections import defaultdict

import pandas as pd
import networkx as nx


REGIONS = ["OFT", "RV", "LV", "AVC", "AB", "Atria"]

_HERE       = os.path.dirname(os.path.abspath(__file__))
CLONES_CSV  = os.path.join(_HERE, "..", "data", "Supplementary_Table_S1.csv")
EDGES_CSV   = os.path.join(_HERE, "..", "data", "Supplementary_Table_S3.csv")
OUT_CSV     = os.path.join(_HERE, "..", "data", "clone_path_attachments.csv")


def canon(regions):
    if isinstance(regions, str):
        s = regions.strip("{}'").replace("'", "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = list(regions)
    return "{" + ", ".join(sorted(parts)) + "}"


def bits_of(canon_str):
    """6-bit pattern string over OFT, RV, LV, AVC, AB, Atria."""
    s = canon_str.strip("{}'").replace("'", "")
    in_set = {p.strip() for p in s.split(",") if p.strip()}
    return "".join("1" if r in in_set else "0" for r in REGIONS)


def main():
    if not os.path.exists(CLONES_CSV):
        sys.exit(f"ERROR: cannot find {CLONES_CSV}")
    if not os.path.exists(EDGES_CSV):
        sys.exit(f"ERROR: cannot find {EDGES_CSV}. Run Figure3a.py first.")

    # ── Load clones (all 94 — the 142-cell clone IS used here, since it
    # is the unique observed instance of the multipotent root pattern).
    clones = pd.read_csv(CLONES_CSV, encoding="utf-8-sig")
    clones = clones[clones["clone_id"].notna()].copy()
    clones["pattern_canon"] = clones["leaf_regions"].apply(canon)
    pattern_to_clones = clones.groupby("pattern_canon").apply(
        lambda g: g[["clone_id", "size", "bin"]].to_dict("records")
    ).to_dict()
    print(f"Loaded {len(clones)} clones across "
          f"{len(pattern_to_clones)} distinct regional patterns")

    # ── Build the graph from S3
    edges_df = pd.read_csv(EDGES_CSV)
    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        u = canon(r["parent_regions"])
        v = canon(r["child_regions"])
        G.add_edge(u, v)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Enumerate every simple root → terminal path
    sources = [n for n, d in G.in_degree()  if d == 0]
    sinks   = [n for n, d in G.out_degree() if d == 0]
    paths = []
    for s in sources:
        for t in sinks:
            paths.extend(list(nx.all_simple_paths(G, s, t)))
    print(f"Paths: {len(paths)} total")

    # ── For each path, attach every clone whose home is on the path, and
    # write one row per (path, node, clone) triple
    rows = []
    path_counter = defaultdict(int)
    for path in paths:
        root, leaf = path[0], path[-1]
        path_counter[(root, leaf)] += 1
        path_idx = path_counter[(root, leaf)]
        # Clones recorded on this path: any clone whose home node is on this path
        path_clones = []
        for node in path:
            if node in pattern_to_clones:
                path_clones.extend(pattern_to_clones[node])
        for depth, node in enumerate(path):
            for clone in path_clones:
                rows.append({
                    "group":          "All",
                    "component_index": 1,
                    "root_bits":      bits_of(root),
                    "root_regions":   root,
                    "leaf_bits":      bits_of(leaf),
                    "leaf_regions":   leaf,
                    "path_index":     path_idx,
                    "path_length":    len(path) - 1,
                    "node_depth":     depth,
                    "node_bits":      bits_of(node),
                    "node_regions":   node,
                    "clone_id":       clone["clone_id"],
                    "size":           clone["size"],
                    "bin":            clone["bin"],
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n✓ Wrote: {OUT_CSV}")
    print(f"  rows: {len(df)}")
    print(f"  distinct paths: {df[['root_regions','leaf_regions','path_index']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
