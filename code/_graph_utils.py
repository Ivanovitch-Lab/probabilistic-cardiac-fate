#!/usr/bin/env python3
"""
_graph_utils.py
===============
Build the potency graph from Supplementary_Table_S3.csv (edges) and
clone_path_attachments.csv (clones propagated along paths), then provide
helpers for path-level scoring.

Terminology (matches the manuscript Methods section):

  - Edge support s(A → B)
        = (clones compatible with the step A → B) / (clones compatible
          with state A).
        Not a probability: a clone whose pattern is still compatible
        with several of A's next steps is counted toward each of them,
        so the supports leaving one state can sum to more than 1.

  - Path score
        = product of edge supports along a root → terminal path.
        Also not a probability — it inherits the property that scores
        across alternative paths need not sum to one.

  - Dominance (per terminal)
        = backbone path score / sum of all path scores to that terminal.
        This IS a normalized fraction (0, 1], used in Figure 4b / Table 3.

  - Margin (per terminal)
        = backbone path score / runner-up path score.

The edge-attribute key in the NetworkX graph is `"p"` for backwards
compatibility — the value stored is the edge SUPPORT, not a probability.

Standalone main(): prints per-terminal backbone-dominance numbers as a
sanity check. Figure / table scripts import the helpers directly.
"""

import os
import sys
from collections import defaultdict

import networkx as nx
import pandas as pd


_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "..", "data")
EDGES_CSV  = os.path.join(DATA_DIR, "Supplementary_Table_S3.csv")
TREES_CSV  = os.path.join(DATA_DIR, "clone_path_attachments.csv")
OUT_CSV    = os.path.join(_HERE, "backbone_dominance.csv")


def _count_regions(s: str) -> int:
    s = str(s or "").strip("{}'").replace("'", "")
    return len(s.split(",")) if s else 0


def _terminal_label(node_regions: str) -> str:
    s = str(node_regions or "").strip("{}'").replace("'", "")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts[0] if len(parts) == 1 else node_regions


def compute_edge_supports():
    """Edge support s(A → B) = |clones compatible with the step A → B|
    divided by |clones compatible with state A|. See module docstring."""
    trees = pd.read_csv(TREES_CSV)
    trees_sorted = trees.sort_values(
        ["group", "component_index", "path_index",
         "root_bits", "leaf_bits", "node_depth"]
    )

    edge_clones = defaultdict(set)
    node_clones = defaultdict(set)
    for _, path_df in trees_sorted.groupby(
        ["group", "component_index", "path_index", "root_bits", "leaf_bits"]
    ):
        nodes  = path_df.sort_values("node_depth")["node_regions"].tolist()
        clones = set(path_df["clone_id"].dropna().unique())
        for node_reg in path_df["node_regions"].astype(str).unique():
            node_clones[node_reg].update(clones)
        for i in range(len(nodes) - 1):
            if nodes[i] != nodes[i + 1]:
                edge_clones[(nodes[i], nodes[i + 1])].update(clones)

    edge_support = {}
    for (u, v), c in edge_clones.items():
        n_parent = len(node_clones.get(u, set()))
        if n_parent > 0:
            edge_support[(u, v)] = len(c) / n_parent
    return edge_support


def build_graph(edge_support):
    """Build NetworkX DiGraph from S3 edges, attaching the precomputed
    edge supports as the `"p"` edge attribute (key retained for
    backwards compatibility — the stored value is the edge support)."""
    edges = pd.read_csv(EDGES_CSV)
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        u = str(row["parent_regions"]).strip()
        v = str(row["child_regions"]).strip()
        if (u, v) in edge_support:
            G.add_edge(u, v, p=edge_support[(u, v)])
    return G


def path_score(G, path, *, exclude_root_and_terminal=False):
    """Path score = product of edge supports along a root → terminal path.
    Not a probability (see module docstring).

    If exclude_root_and_terminal is True, drop the first edge (out of the
    multipotent root) and the last edge (into the terminal fate) — these
    encode root prevalence and terminal-population frequencies rather
    than fate-restriction decisions proper.
    """
    s = 1.0
    edges = list(zip(path[:-1], path[1:]))
    if exclude_root_and_terminal and len(edges) > 2:
        edges = edges[1:-1]
    for u, v in edges:
        s *= G.edges[u, v]["p"]
    return s


def main():
    if not os.path.exists(EDGES_CSV) or not os.path.exists(TREES_CSV):
        print(f"ERROR: required CSVs not found in {DATA_DIR}")
        sys.exit(1)

    print("Computing edge supports ...")
    edge_support = compute_edge_supports()
    G = build_graph(edge_support)

    roots     = [n for n in G.nodes if _count_regions(n) >= 6]
    terminals = [n for n in G.nodes if _count_regions(n) == 1]
    if len(roots) != 1:
        raise SystemExit(f"Expected 1 multipotent root, found {len(roots)}")
    root = roots[0]
    print(f"Root: {root}")
    print(f"Terminals: {len(terminals)} | Edges: {len(G.edges)}")

    rows = []
    for variant_label, exclude_rt in [
        ("FULL paths (5 edges)", False),
        ("INTERNAL edges only (3 edges, root+terminal removed)", True),
    ]:
        print(f"\n=== {variant_label} ===\n")
        print(f"{'Terminal':<8} {'#paths':>7} {'backbone score':>15} "
              f"{'total mass':>12} {'dominance':>10} {'2nd best':>10} "
              f"{'margin':>8}")
        print("-" * 80)

        for term in sorted(terminals, key=_terminal_label):
            all_paths = list(nx.all_simple_paths(G, root, term))
            if not all_paths:
                continue
            scores = [path_score(G, p, exclude_root_and_terminal=exclude_rt)
                      for p in all_paths]
            order = sorted(range(len(scores)),
                           key=lambda i: scores[i], reverse=True)
            sorted_scores = [scores[i] for i in order]
            sorted_paths  = [all_paths[i] for i in order]

            backbone_score = sorted_scores[0]
            backbone_path  = sorted_paths[0]
            total_mass     = sum(sorted_scores)
            dominance      = (backbone_score / total_mass
                              if total_mass > 0 else 0.0)
            second_score   = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            margin         = (backbone_score / second_score
                              if second_score > 0 else float("inf"))

            label = _terminal_label(term)
            print(f"{label:<8} {len(all_paths):>7d} {backbone_score:>15.4f} "
                  f"{total_mass:>12.4f} {dominance:>10.2%} "
                  f"{second_score:>10.4f} {margin:>8.2f}")

            if not exclude_rt:
                rows.append({
                    "terminal": label,
                    "n_paths_total": len(all_paths),
                    "backbone_score": backbone_score,
                    "total_mass": total_mass,
                    "dominance": dominance,
                    "second_score": second_score,
                    "margin_ratio": margin,
                    "backbone_path": " → ".join(backbone_path),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Saved: {OUT_CSV}")
    print("\nSummary across terminals:")
    print(f"  median dominance : {df['dominance'].median():.2%}")
    print(f"  range dominance  : {df['dominance'].min():.2%} – "
          f"{df['dominance'].max():.2%}")
    print(f"  median margin    : "
          f"{df['margin_ratio'].replace(float('inf'), pd.NA).median():.2f}×")


if __name__ == "__main__":
    main()
