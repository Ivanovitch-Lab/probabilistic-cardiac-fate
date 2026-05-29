#!/usr/bin/env python3
"""
test_fast_signature.py
======================
Correctness check for the optimisation that makes the full ~21M-combination
scan tractable. _topology_utils.fast_signature is a fast topology-deduper
used inside enumerate_full_space; this test asserts it groups path
combinations IDENTICALLY to the slow, canonical
signature(derive_restriction_sequence(...)) on a large random sample.

If this passes, the de-duplication driving Figure 4a / Table 4 is trustworthy.

Run
---
    python test_fast_signature.py        # or: pytest test_fast_signature.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_utils import compute_edge_supports, build_graph
from _sequence_utils import derive_restriction_sequence
from _topology_utils import (
    all_paths_per_terminal, signature, fast_signature,
)


def test_fast_signature_matches_canonical(n_random=20000, seed=0):
    G = build_graph(compute_edge_supports())
    ppt = all_paths_per_terminal(G)
    terms = sorted(ppt.keys())
    node_paths_t = [[tuple(p) for p, _ in ppt[t]] for t in terms]
    ranges = [len(x) for x in node_paths_t]

    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        idx = [int(rng.integers(r)) for r in ranges]
        combo_paths = {terms[k]: list(node_paths_t[k][idx[k]])
                       for k in range(len(terms))}
        canon = signature(derive_restriction_sequence(combo_paths))
        fs = fast_signature([node_paths_t[k][idx[k]]
                             for k in range(len(terms))])
        # relabel fast_signature's integer fate-ids back to fate names
        fs_named = frozenset(
            (frozenset(terms[i] for i in parent),
             frozenset(frozenset(terms[i] for i in child)
                       for child in childset))
            for parent, childset in fs
        )
        assert fs_named == canon, (
            f"fast_signature mismatch at idx={idx}\n"
            f"  fast = {fs_named}\n  canon= {canon}")


if __name__ == "__main__":
    test_fast_signature_matches_canonical()
    print("OK: fast_signature matches the canonical signature on 20,000 combos.")
