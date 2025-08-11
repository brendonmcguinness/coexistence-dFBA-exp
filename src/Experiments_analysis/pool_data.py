#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:55:38 2025

@author: brendonmcguinness
"""
import glob
import os
import pandas as pd

# 1) List your two CSV paths
file_paths = [
    "/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/per_replicate_SC_metrics_imputed_extraKanpipeline.csv",
    "/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/per_replicated_SC_metrics_imputed.csv",
]

df2 = pd.read_csv(file_paths[1])
df2 = df2[~((df2.strain1 == "MG") & (df2.strain2 == "MGK"))].copy()

df1 = pd.read_csv(file_paths[0])
# 2) Read & tag each one
# 3) Renumber df2’s replicates so they follow on from df1’s
max_rep = df1["replicate"].max()
df2["replicate"] = df2["replicate"] + max_rep

# 4) Concatenate
merged = pd.concat([df1, df2], ignore_index=True)

# 5) (Optionally) sort or reset index
merged_data = merged.sort_values(["strain1","strain2","replicate"]).reset_index(drop=True)


merged_data.to_csv('per_replicate_SC_metrics_pooled.csv')
