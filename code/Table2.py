#!/usr/bin/env python3
"""
Table2.py
=========
Reproduces Table 2 of the accompanying manuscript:
robustness of Large-clone co-occurrence z-scores to the Small/Large size
threshold. 15 region-pairs (rows) × 5 size cutoffs (columns). Cells with
FDR ≤ 0.05 are bolded and shaded pale orange.

Input
-----
../data/Supplementary_Table_S2.csv   sensitivity-sweep stats (from SuppFigure1.py)

Output
------
../figures/Table2.png

Run
---
    python Table2.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BIN_LABEL = "Large"

PAIR_ORDER = [
    "OFT-RV", "AVC-AB", "OFT-LV", "AB-Atria", "OFT-AVC",
    "OFT-AB", "OFT-Atria", "RV-LV", "RV-AVC", "RV-AB",
    "RV-Atria", "LV-AVC", "LV-AB", "LV-Atria", "AVC-Atria",
]

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_HERE, "..", "data", "Supplementary_Table_S2.csv")
OUT_PNG    = os.path.join(_HERE, "..", "figures", "Table2.png")
OUT_XLSX   = os.path.join(_HERE, "..", "figures", "Table2.xlsx")


def main():
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: cannot find {INPUT_FILE}. Run SuppFigure1.py first.")
    df = pd.read_csv(INPUT_FILE)
    df_bin = df[df["bin"] == BIN_LABEL].copy()
    cutoffs = sorted(df_bin["cutoff"].unique())
    print(f"Loaded {len(df_bin)} rows for the {BIN_LABEL}-clone bin "
          f"({len(PAIR_ORDER)} pairs × {len(cutoffs)} cutoffs)")

    cell_text, sig_mask = [], []
    for pair in PAIR_ORDER:
        row_text, row_sig = [], []
        for c in cutoffs:
            sub = df_bin[(df_bin["pair"] == pair) & (df_bin["cutoff"] == c)]
            if sub.empty:
                row_text.append("—"); row_sig.append(False); continue
            r = sub.iloc[0]
            mark = "*" if r["sig"] else ""
            row_text.append(f"{r['z']:.2f}{mark}")
            row_sig.append(bool(r["sig"]))
        cell_text.append(row_text)
        sig_mask.append(row_sig)

    col_labels = [f">{c} cells" for c in cutoffs]

    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=PAIR_ORDER,
        colLabels=col_labels,
        cellLoc="center", rowLoc="right", loc="center",
        colColours=["#EEEEEE"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.4)

    for j in range(len(col_labels)):
        table[(0, j)].get_text().set_fontweight("bold")
    for i in range(1, len(PAIR_ORDER) + 1):
        table[(i, -1)].get_text().set_fontweight("bold")
        if i % 2 == 0:
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor("#F8F8F8")
        for j, sig in enumerate(sig_mask[i - 1]):
            if sig:
                table[(i, j)].set_facecolor("#FFE8B0")
                table[(i, j)].get_text().set_fontweight("bold")

    fig.suptitle(f"Clonal-coupling z-scores — {BIN_LABEL}-clone bin",
                 fontsize=10.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.04,
             "Asterisk (*) and shading mark cells with FDR ≤ 0.05  ·  "
             "Curveball null, 100 000 permutations  ·  "
             "BH FDR correction within each bin × cutoff",
             ha="center", va="bottom", fontsize=7.5, color="#444")

    plt.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Saved: {OUT_PNG}")

    # Editable Excel companion for the typesetter (same cell content as the PNG).
    df_xlsx = pd.DataFrame(cell_text, index=PAIR_ORDER, columns=col_labels)
    df_xlsx.index.name = "Region pair"
    df_xlsx.to_excel(OUT_XLSX)
    print(f"✓ Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
