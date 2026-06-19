#!/usr/bin/env bash
# reproduce.sh
# Regenerates every figure and table in the accompanying manuscript from
# scratch — including the Curveball permutation null model
# (Supplementary_Table_S2.csv), the Ward k=3 hierarchical clustering
# (Supplementary_Table_S3.csv + S4.csv), and the potency-graph clone
# attachments (clone_path_attachments.csv).
#
# All randomness uses a fixed seed (RANDOM_SEED = 0) set inside each
# script, so outputs are bit-identical across runs and platforms.
# Expected runtime: ~5 minutes on a 2024 MacBook.

set -euo pipefail
cd "$(dirname "$0")/code"

# Step 1 — permutation null model (writes Supplementary_Table_S2.csv) ~3 min
python SuppFigure1.py

# Step 2 — potency graph + Ward k=3 clustering (writes S3 and S4)
python Figure3a.py

# Step 3 — propagate clones along every root-to-terminal path
#          (writes clone_path_attachments.csv from S1 + S3)
python _build_attachments.py

# Step 4 — every remaining figure and table (no inter-dependencies)
for f in Figure1b.py Figure2a.py Figure2b.py Figure3b.py Figure3c.py \
         Figure3d.py Figure4a.py Figure4b.py Figure4c.py Figure4d.py \
         SuppFigure2.py SuppFigure3.py SuppFigure4.py \
         Table1.py Table2.py Table3.py Table4.py; do
    python "$f"
done

echo
echo "✓ All figures and tables reproduced — see figures/"
