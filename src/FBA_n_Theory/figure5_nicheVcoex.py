#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 01:59:59 2025
@author: brendonmcguinness
"""
import re
import cometspy as c
import cobra
from cobra.io import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helpers import analyze_biomass_growth, analyze_biomass_growth_mult
from matplotlib.colors import TwoSlopeNorm

def clean_source(source: str) -> str:
    return re.sub(r"^EX_", "", re.sub(r"_e$", "", source))

def clean_pair(label: str) -> str:
    a, b = label.split(", ")
    return clean_source(a) + " + " + clean_source(b)

def clean_label(label: str) -> str:
    a, b = label.split(" + ")
    return clean_source(a) + ", " + clean_source(b)

# … your data-loading and merging as before …
df_merged = pd.read_csv('figure5plotnichevcoex.csv')
# Rename columns
df_merged = df_merged.rename(columns={
    "Value_niche":   "Niche Differences",
    "Value_fitness":"Fitness Differences",
    "Value":         "Coexistence Strength"
})

# … compute Growth Rate Diff, Carrying Capacity Diff, Carbon Pair, Carbon Pair Label …

# ─── NEW: Min–max normalize Niche Differences to [0,1] ─────────────────
df_merged['ND norm'] = df_merged["Niche Differences"]**0.5 / (1.0 + df_merged["Niche Differences"]**0.5)
nd_min = df_merged['ND norm'].min()
nd_max = df_merged['ND norm'].max()
df_merged["Niche Differences (norm)"] = (
    df_merged['ND norm'] - nd_min
) / (nd_max - nd_min)
"""
nd_min = df_merged["Niche Differences"].min()
nd_max = df_merged["Niche Differences"].max()
df_merged["Niche Differences (norm)"] = (
    df_merged["Niche Differences"] - nd_min
) / (nd_max - nd_min)
"""
# ───────────────────────────────────────────────────────────────────────



df_merged["Carbon Pair Label"] = df_merged["Carbon Pair"].map(clean_label)

# 2) Recompute the sort order on the cleaned labels
growth_diff = (
    df_merged[["Carbon Pair Label", "Growth Rate Diff"]]
    .drop_duplicates()
    .sort_values("Growth Rate Diff")
)

sorted_labels = growth_diff["Carbon Pair Label"].tolist()

# Build reversed-Viridis palette
palette = sns.color_palette("viridis_r", n_colors=len(sorted_labels))
pair_color_map = dict(zip(sorted_labels, palette))

# Plot (using the normalized niche differences)
plt.figure(figsize=(8, 4))
sns.lineplot(
    data=df_merged,
    x="Niche Differences",        # <— use the normalized column
    y="Coexistence Strength",
    hue="Carbon Pair Label",
    hue_order=sorted_labels,
    palette=pair_color_map,
    marker="o",
    linestyle="-",
    legend="full"
)

plt.axhline(0, linestyle="--", color="k", linewidth=2)
plt.xlabel("Raw Niche Difference (ND)")
plt.ylabel("Coexistence Strength (Coex)")

plt.legend(
    title="Carbon Pair (ordered by |Δ max growth rate|)",
    title_fontsize="small",
    fontsize="small",
    ncol=2,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.savefig('fig5_niche_v_coex_labelpairs_raw.pdf')
plt.show()

# Plot (using the normalized niche differences)
plt.figure(figsize=(8, 4))
sns.lineplot(
    data=df_merged,
    x="KO Bound Source 1",        # <— use the normalized column
    y="Niche Differences (norm)",
    hue="Carbon Pair Label",
    hue_order=sorted_labels,
    palette=pair_color_map,
    marker="o",
    linestyle="-",
    legend="full"
)

plt.axhline(0, linestyle="--", color="k", linewidth=2)
plt.xlabel("Niche Difference (ND)")
plt.ylabel("Coexistence Strength")

plt.legend(
    title="Carbon Pair (ordered by |Δ max growth rate|)",
    title_fontsize="small",
    fontsize="small",
    ncol=2,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
#plt.savefig('fig5_niche_v_coex_labelpairs_unprime.pdf')
plt.show()