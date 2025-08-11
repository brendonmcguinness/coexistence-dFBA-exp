
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 13:36:01 2025

@author: brendonmcguinness

– Loops over initial mean populations AND a range of init_ratio_diff values
– Plots ND & FD vs. initial_mean_pop for each init_ratio_diff
"""

import cometspy as c
from cobra.io import load_model
import numpy as np
import matplotlib.pyplot as plt
from helpers import nicheOverlapSaveTimeSeriesMultipleChooseTime, returnSCMatrix

# --- SETTINGS ---
exchange_rxns = ["EX_glc__D_e", "EX_xyl__D_e"]
M = 2
s1_conc = np.array([0.1])
s2_conc = np.array([0.1])
ko1 = ko2 = -5
source1, source2 = exchange_rxns

# Prepare model objects
mut1 = c.model(load_model("iJO1366"))
mut2 = c.model(load_model("iJO1366"))
for rxn in exchange_rxns:
    mut1.change_bounds(rxn, -10, 1000)
    mut2.change_bounds(rxn, -10, 1000)
mut1.change_bounds(source1, ko1, 1000)
mut2.change_bounds(source2, ko2, 1000)
mut1.id = f"{source1}_KO_{ko1}"
mut2.id = f"{source2}_KO_{ko2}"

# Vary initial populations and ratio differences
initial_mean_pops = np.logspace(-5, -2, 5)     # 10 values from 1e-6 to 1e-2
init_ratio_diffs  = np.logspace(-4, -1, 5)     # 10 values from 1e-6 to 1e-1

# Storage for results
ND_results = {}  # key = ratio_diff, value = list of ND over initial_mean_pops
FD_results = {}  # key = ratio_diff, value = list of FD over initial_mean_pops

for ratio in init_ratio_diffs:
    ND_list = []
    FD_list = []
    for imp in initial_mean_pops:
        s_wt, s_mut, wt_freq, media_list = returnSCMatrix(
            mut1, mut2, source1, source2,
            s1_conc, s2_conc,
            N=M,
            init_ratio_diff=ratio,
            init_mean_pop=imp
        )
        # niche overlap → ND
        NO = (s_wt[0] - s_wt[-1]) / (wt_freq[0] - wt_freq[-1])
        ND_list.append(abs(NO))
        # fitness difference FD
        FD_list.append(s_wt[0] - s_mut[1])

    ND_results[ratio] = ND_list
    FD_results[ratio] = FD_list

# --- PLOTTING ---
plt.figure(figsize=(12, 5))

# ND subplot
plt.subplot(1, 2, 1)
for ratio, nd_vals in ND_results.items():
    plt.plot(initial_mean_pops, nd_vals, marker='o', label=f"ratio={ratio:.0e}")
plt.xscale('log')
plt.xlabel('initial_mean_pop')
plt.ylabel('Niche difference (ND)')
plt.title('ND vs init_mean_pop')
plt.legend(fontsize='small', ncol=2)

# FD subplot
plt.subplot(1, 2, 2)
for ratio, fd_vals in FD_results.items():
    plt.plot(initial_mean_pops, fd_vals, marker='s', label=f"ratio={ratio:.0e}")
plt.xscale('log')
plt.xlabel('initial_mean_pop')
plt.ylabel('Fitness difference (FD)')
plt.title('FD vs init_mean_pop')
plt.legend(fontsize='small', ncol=2)

plt.tight_layout()
plt.show()
