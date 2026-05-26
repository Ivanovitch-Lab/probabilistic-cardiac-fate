#!/usr/bin/env python3
"""
fate_restriction_sequence.py
============================
From the ranked backbones (most-probable, runner-up, etc., obtained by
brute-force enumeration of all simple root→terminal paths) derive a
consensus binary tree of fate restrictions:

  Step 1 : multipotent root holds all 6 terminal fates
  Step 2 : first split — two daughter groups
  Step 3 : each daughter splits further
  …
  Step n : six singleton terminal fates

The split at each level is decided by where each terminal's backbone passes:
two terminals that share an intermediate node at level k stay together until
that level. The function `mrca_depth` quantifies this; the binary tree is
built from those depths via single-linkage clustering.

This is a helper module (no `__main__` entry point) — `kth_best_paths` and
`derive_restriction_sequence` are imported by the figure-producing scripts.
"""

import os
import sys
from collections import defaultdict

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import (
    compute_edge_supports, build_graph, _count_regions,
    _terminal_label, path_score,
)


def kth_best_paths(G: nx.DiGraph, rank: int = 1) -> dict[str, list[str]]:
    """For each terminal fate, return its k-th most probable root→terminal
    simple path as a list of node ids."""
    roots     = [n for n in G.nodes if _count_regions(n) >= 6]
    terminals = [n for n in G.nodes if _count_regions(n) == 1]
    if not roots:
        return {}
    root = roots[0]

    out: dict[str, list[str]] = {}
    for t in terminals:
        all_paths = list(nx.all_simple_paths(G, root, t))
        if not all_paths:
            continue
        all_paths.sort(key=lambda p: path_score(G, p), reverse=True)
        if rank > len(all_paths):
            continue
        out[_terminal_label(t)] = all_paths[rank - 1]
    return out


def derive_restriction_sequence(paths: dict[str, list[str]]) -> list[dict]:
    """Walk the backbones from root downward and emit one row per restriction
    step. Each row records the parent node, the daughter groups that emerge,
    and which fates each group contains.
    """
    fates = list(paths.keys())
    if not fates:
        return []

    # depth-of-each-node-in-each-backbone, indexed by fate
    # path[fate][i] = node visited at i-th step (i=0 → root)
    sequence = []
    # current_partition: list of dicts {node: parent_node_visited, fates_in_group}
    # Start with all fates sitting at the root (depth 0)
    root = paths[fates[0]][0]
    assert all(p[0] == root for p in paths.values()), "Backbones must share root"

    # The "active group" structure: tuple(parent_node, frozenset_of_fates).
    # At each step, for each active group, look one step deeper in each member's
    # backbone and re-partition by the next node visited.
    groups = [(root, frozenset(fates))]
    step_idx = 0
    while groups:
        # Move every group one step forward (if possible)
        next_groups: list[tuple[str, frozenset[str]]] = []
        emitted_split = False
        for parent_node, fates_here in groups:
            # Children-by-node: which next node does each fate go to?
            by_next: dict[str, list[str]] = defaultdict(list)
            for f in fates_here:
                p = paths[f]
                # Find the index of parent_node in this backbone
                try:
                    idx = p.index(parent_node)
                except ValueError:
                    # This fate's backbone never visited the parent — keep it
                    # in the same group for now.
                    by_next[parent_node].append(f)
                    continue
                if idx + 1 < len(p):
                    by_next[p[idx + 1]].append(f)
                else:
                    # Already at the terminal; group becomes a singleton
                    by_next[parent_node].append(f)

            if len(by_next) == 1 and parent_node in by_next:
                # No restriction at this level — group is at its terminal
                continue
            elif len(by_next) == 1:
                # All fates went to the same next node — no split, just
                # advance the group.
                only_next = next(iter(by_next))
                next_groups.append((only_next, frozenset(by_next[only_next])))
            else:
                # Genuine split — record it
                emitted_split = True
                children_records = []
                for next_node, fates_next in by_next.items():
                    if next_node == parent_node:
                        # Group already at its terminal — emit as singleton
                        for f in fates_next:
                            children_records.append({
                                "next_node": f,
                                "fates": frozenset([f]),
                            })
                    else:
                        children_records.append({
                            "next_node": next_node,
                            "fates": frozenset(fates_next),
                        })
                        next_groups.append((next_node, frozenset(fates_next)))

                sequence.append({
                    "step": step_idx,
                    "parent_node": parent_node,
                    "fates_before_split": sorted(fates_here),
                    "n_regions_before": _count_regions(parent_node),
                    "children": [
                        {
                            "node": c["next_node"],
                            "n_regions": _count_regions(c["next_node"])
                                          if c["next_node"] not in fates_here else 1,
                            "fates": sorted(c["fates"]),
                        }
                        for c in children_records
                    ],
                })

        groups = next_groups
        step_idx += 1
        if not emitted_split and step_idx > 10:
            # Safety net
            break
    return sequence


