#!/usr/bin/env python3
"""
enumerate_all_topologies.py
===========================
Detailed report for the exhaustive (not top-K) enumeration of the full
~21M-combination space of root->terminal path choices across the six
terminal fates. Answers:

    "If we score every possible combination instead of only the top
     10,000, how many distinct restriction-sequence topologies survive
     all three biological filters, and how plausible are the ones beyond
     the five originally shown in Figure 4a?"

The heavy lifting lives in _topology_utils.enumerate_full_space (shared
with Figure4a.py). This script just validates the fast signature routine
against the canonical signature(derive_restriction_sequence(...)) and then
prints a detailed per-topology table (best joint lp, relative likelihood,
marginal support weight, cumulative weight, and the rank of each
topology's best combination among all ~21M combos).
"""

import math

import numpy as np

from _graph_utils import compute_edge_supports, build_graph
from _topology_utils import (
    all_paths_per_terminal, signature, fast_signature,
    enumerate_full_space, load_clone_regions,
)
from _sequence_utils import derive_restriction_sequence


def validate_fast_sig(node_paths_t, terms, n_random=20000, seed=0):
    """Assert fast_signature groups combos identically to the canonical
    signature(derive_restriction_sequence(...)) on a random sample."""
    rng = np.random.default_rng(seed)
    ranges = [len(x) for x in node_paths_t]
    for _ in range(n_random):
        idx = [int(rng.integers(r)) for r in ranges]
        combo_paths = {terms[k]: list(node_paths_t[k][idx[k]])
                       for k in range(len(terms))}
        canon = signature(derive_restriction_sequence(combo_paths))
        fs = fast_signature([node_paths_t[k][idx[k]] for k in range(len(terms))])
        fs_named = frozenset(
            (frozenset(terms[i] for i in parent),
             frozenset(frozenset(terms[i] for i in child) for child in childset))
            for parent, childset in fs
        )
        assert fs_named == canon, (
            f"fast_signature mismatch at idx={idx}\n fast={fs_named}\n canon={canon}")
    print(f"  fast_signature validated against canonical on {n_random} combos.")


def _tree_str(combo_paths):
    seq = derive_restriction_sequence(combo_paths)
    return " | ".join(
        "{" + "+".join(sorted(s["fates_before_split"])) + "}->"
        + "/".join("".join(sorted(c["fates"])) for c in s["children"])
        for s in seq
    )


def main():
    G = build_graph(compute_edge_supports())
    ppt = all_paths_per_terminal(G)
    terms = sorted(ppt.keys())
    node_paths_t = [[tuple(p) for p, _ in ppt[t]] for t in terms]

    print("Validating fast signature routine ...")
    validate_fast_sig(node_paths_t, terms)

    print("\nEnumerating full space ...")
    clones_df = load_clone_regions()
    res = enumerate_full_space(G, clones_df, progress=True)
    f = res["funnel"]
    total = res["total"]

    print("\n" + "=" * 72)
    print(f"FUNNEL over ALL {total:,} combinations")
    print(f"  distinct topologies (any shape)      : {f['distinct']:,}")
    print(f"  strictly bifurcating                 : {f['bifurcating']:,}"
          f"   (support {res['weight_binary']:.2%})")
    print(f"  + every bifurcation clone-supported  : {f['supported']:,}"
          f"   (support {res['weight_supported']:.2%})   <- median DROPPED")
    print(f"  + monotone median clone size         : {f['monotone']}"
          f"   (support {res['weight_monotone']:.2%})   <- drawn in Figure 4a")
    print("=" * 72)
    print(f"MAP (single most-probable) reconstruction is "
          f"{'a binary tree.' if res['map_is_binary'] else 'MULTIFURCATING.'}")

    print(f"\n{f['monotone']} topologies survive ALL three filters:")
    print(f"  {'id':>4} {'best_lp':>9} {'rel.L':>7} {'post.wt':>9} "
          f"{'cum.wt':>8} {'combo_rank':>12}")
    lp_top = res["survivors"][0]["best_lp"]
    cum = 0.0
    for r, e in enumerate(res["survivors"], 1):
        cum += e["weight"]
        relL = math.exp(e["best_lp"] - lp_top)
        print(f"  S{r:<3d} {e['best_lp']:9.3f} {relL:7.3f} {e['weight']:8.2%} "
              f"{cum:8.2%} {e['combo_rank']:>11,}")

    print("\nTrees (S1..) :")
    for r, e in enumerate(res["survivors"], 1):
        print(f"  S{r:<2d} {_tree_str(e['combo_paths'])}")

    print(f"\n(Figure4a.py originally scanned only the top TOP_K=10,000 combos,"
          f" which surfaced just 5 of these {f['monotone']}.)")


if __name__ == "__main__":
    main()
