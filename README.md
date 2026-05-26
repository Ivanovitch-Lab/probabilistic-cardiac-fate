# Beyond binary: cardiac patterning as probabilistic fate restriction

> Ivanovitch, K. **Beyond binary: cardiac patterning as probabilistic
> fate restriction.** *Manuscript in preparation.*

This repository contains the analysis code, processed data, and
figure-generating scripts behind every quantitative figure and table in
the manuscript. Each script reproduces a single panel.

## Reproduce all figures and tables (one command)

```bash
bash reproduce.sh
```

This single command regenerates from scratch:

- the **Curveball permutation null model** for clonal-coupling z-scores
  (100,000 permutations; writes `Supplementary_Table_S2.csv`),
- the **Ward k = 3 hierarchical clustering** of the 22 multi-region
  intermediate states (writes `Supplementary_Table_S3.csv` and
  `Supplementary_Table_S4.csv`),
- the **potency-graph construction** and clone-to-path attachments
  (writes `clone_path_attachments.csv`),
- every figure (PNG, 300 dpi) and every table (PNG) in the main text
  and supplement.

All randomness uses a fixed seed (`RANDOM_SEED = 0`) set inside each
script, so numerical outputs are bit-identical across runs and
platforms. Expected runtime: ~5 minutes on a 2024 MacBook.

## Requirements

- **Python 3.10+**
- Python packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `networkx`,
  `pydot`
- **`graphviz`** as a system dependency (`dot` binary on `PATH`) for the
  DAG layout used by Figure 3c, Figure 4a, and SuppFigure 2:
  - macOS: `brew install graphviz`
  - Debian/Ubuntu: `apt install graphviz`

Install Python dependencies:
```bash
pip install numpy pandas scipy matplotlib networkx pydot
```

## Repository layout

```
├── code/                  Python scripts (one per figure/table panel)
│   ├── Figure1b.py        Live-imaging clone-size strip plot
│   ├── Figure2a.py        Clone size vs regional breadth
│   ├── Figure2b.py        Co-occurrence z-score heatmaps (cutoff = 30)
│   ├── Figure3a.py        Fate-spectrum dendrogram + heatmap
│   ├── Figure3b.py        Cluster log2 fold-enrichment
│   ├── Figure3c.py        k=3 clustered potency hierarchy DAG
│   ├── Figure3d.py        Per-terminal path-score strip plot
│   ├── Figure4a.py        Five plausible restriction-sequence topologies (overview)
│   ├── Figure4b.py        Rank-1 restriction cladogram
│   ├── Figure4c.py        Rank-3 restriction cladogram
│   ├── SuppFigure1.py     Sliding-window sensitivity heatmaps
│   ├── SuppFigure2.py     Rank-1 vs rank-3 backbone DAGs (side-by-side)
│   ├── Table1.py          Small-bin sensitivity table
│   ├── Table2.py          Large-bin sensitivity table
│   ├── Table3.py          Per-terminal backbone justification table
│   ├── Table4.py          Per-topology summary for the 5 plausible topologies
│   ├── _build_attachments.py   regenerates clone_path_attachments.csv from S1 + S3
│   └── _graph_utils.py, _sequence_utils.py, _topology_utils.py
│                          shared library modules (not stand-alone scripts)
├── data/
│   ├── Supplementary_Table_S1.csv   Meilhac clone-by-region matrix
│   ├── Supplementary_Table_S2.csv   Sensitivity-sweep z-scores and FDR
│   ├── Supplementary_Table_S3.csv   Potency graph edges
│   ├── Supplementary_Table_S4.csv   Per-state fate spectrum + k=3 cluster
│   ├── Abukar_clone_data.csv        Live-imaging per-clone counts (Abukar et al. 2025)
│   └── clone_path_attachments.csv   Clone-to-path attachments with propagated support
├── figures/               PNG outputs at 300 dpi
├── reproduce.sh           one-command cold-start reproduction
└── README.md              this file
```

The four `Supplementary_Table_S{1–4}.csv` files are cited as
Supplementary Tables in the manuscript. `Abukar_clone_data.csv`
re-publishes the live-imaging per-clone counts used by Figure 1b
(Abukar et al. 2025). `clone_path_attachments.csv` is a cached
intermediate (clones propagated along every root-to-terminal path)
that `_build_attachments.py` regenerates from S1 + S3 — it is shipped
so that individual figure / table scripts run standalone, but it is
not cited in the manuscript.

## Running a single figure

Each script writes its PNG to `../figures/` and is self-contained:

```bash
cd code/
python Figure2a.py
```

Outputs `../figures/Figure2a.png`.

## Manual step-by-step reproduction

A few scripts produce Supplementary Tables and intermediate data files
that other scripts consume, so order matters for a cold rerun. `reproduce.sh`
runs these in the correct order; equivalently:

```bash
cd code/

# Step 1 — Curveball permutation null (writes Supplementary_Table_S2.csv) ~3 min
python SuppFigure1.py

# Step 2 — potency graph + Ward k=3 clustering (writes S3 and S4)
python Figure3a.py

# Step 3 — propagate clones along every root-to-terminal path
#          (writes clone_path_attachments.csv from S1 + S3)
python _build_attachments.py

# Step 4 — every remaining figure and table (any order)
python Figure1b.py
python Figure2a.py
python Figure2b.py
python Figure3b.py
python Figure3c.py
python Figure3d.py
python Figure4a.py
python Figure4b.py
python Figure4c.py
python SuppFigure2.py
python Table1.py
python Table2.py
python Table3.py
python Table4.py
```

`Figure1b`, `Figure2a`, `Figure2b`, `Figure4b`, `Figure4c` are
independent and can run without any other script. `Figure3d`,
`Figure4a`, `SuppFigure2`, `Table3` and `Table4` all consume
`clone_path_attachments.csv`, so they require Step 3 to have run first.

## Citation

Ivanovitch, K. Beyond binary: cardiac patterning as probabilistic
fate restriction. *Manuscript in preparation.* Citation will be
updated on acceptance.

## Contact

k.ivanovitch@ucl.ac.uk
