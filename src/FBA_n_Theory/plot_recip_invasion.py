#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 23:47:12 2025

@author: brendonmcguinness
"""

import os
import shutil
import tempfile
from copy import deepcopy
from itertools import combinations
import numpy as np
import pandas as pd
import cometspy as c
import cobra
from cobra.io import load_model
import matplotlib.pyplot as plt

from helpers import (
    nicheOverlapSaveTimeSeriesMultipleChooseTime,
    fitnessDifferenceSaveTimeSeriesMultipleRandChooseTime,
)

def setup_mutants(base, s1, s2, ko1, ko2, cross_bound, rxns):
    mut1, mut2 = deepcopy(base), deepcopy(base)
    for r in rxns:
        mut1.change_bounds(r, cross_bound, 1000)
        mut2.change_bounds(r, cross_bound, 1000)
    mut1.id, mut2.id = f"{s1}_KO_{ko1}", f"{s2}_KO_{ko2}"
    mut1.change_bounds(s1, ko1, 1000)
    mut1.change_bounds(s2, -20, 1000)
    mut2.change_bounds(s2, ko2, 1000)
    mut2.change_bounds(s1, -20, 1000)
    return mut1, mut2


# --- Configuration ---
os.environ['COMETS_HOME'] = '/Applications/COMETS'
os.environ['GUROBI_COMETS_HOME'] = '/Library/gurobi1003/macos_universal2'
os.environ['GRB_LICENSE_FILE'] = '/Library/gurobi1003/macos_universal2/gurobi.lic'
"""
carbs = [
    "EX_glc__D_e", "EX_fru_e", "EX_gal_e", "EX_xyl__D_e", "EX_succ_e", 
    "EX_mal__L_e", "EX_fum_e", "EX_glyc_e", "EX_cit_e"
]
"""
carbs = [
    "EX_glc__D_e", "EX_cit_e"
]

carbon_pairs = list(combinations(carbs, 2))
ko_bounds   = np.arange(-10, 1)
M           = 1
s1_conc     = np.linspace(0.005, 0.05, M)
s1_conc = np.ones(2) *1e-2 #np.array([0.0275,0.0275])
s2_conc     = s1_conc[::-1]
exchange_rxns = [
    "EX_glc__D_e", "EX_fru_e", "EX_gal_e", "EX_man_e", "EX_xyl__D_e", 
    "EX_arab__L_e", "EX_ac_e", "EX_lac__D_e", "EX_pyr_e", "EX_succ_e", 
    "EX_mal__L_e", "EX_fum_e", "EX_glyc_e", "EX_cit_e", "EX_akg_e",
    "EX_eth_e", "EX_for_e"
]

base_model = c.model(load_model("iJO1366"))
mut1, mut2 = setup_mutants(base_model, carbs[0], carbs[1], -3, -3, 0, exchange_rxns)

no, co_val, b1, m1, winner, sc_val = nicheOverlapSaveTimeSeriesMultipleChooseTime(mut1, mut2, carbs[0], carbs[1],s1_conc[0], s1_conc[0], max_cyc=240,init_mean_pop=1e-4)


# b1 should be your list of DataFrames, each with columns ['cycle', 'EX_glc__D_e_KO…', 'EX_succ_e_KO…', 't']

for idx, df in enumerate(b1, start=1):
    fig, ax = plt.subplots(figsize=(4, 3))
    
    t = df['t']
    # Identify the two biomass columns (everything except 'cycle' and 't')
    biomass_cols = [c for c in df.columns if c not in ('cycle', 't')]
    
    # Plot strain 1
    ax.semilogy(t, df[biomass_cols[0]],
                label='strain 1',
                color='tab:blue',
                linewidth=2)
    # Plot strain 2
    ax.semilogy(t, df[biomass_cols[1]],
                label='strain 2',
                color='tab:orange',
                linewidth=2)
    
    # Annotate initial (0) and final (f) points
    for i, col in enumerate(biomass_cols, start=1):
        y0 = df[col].iloc[0]
        yf = df[col].iloc[-1]
        # initial
        """
        ax.annotate(
            rf'$N_{i}^0$',
            xy=(t.iloc[0], y0),
            xytext=(t.iloc[0] + 0.1, y0 *5), fontsize=12#, 
            #arrowprops=dict(arrowstyle='->', lw=0.8)
        )
        # final
        ax.annotate(
            rf'$N_{i}^f$',
            xy=(t.iloc[-1]-0.9, yf),
            xytext=(t.iloc[-1] - 0.9, yf / 3), fontsize=12#, 
            #arrowprops=dict(arrowstyle='->', lw=0.8)
        )
        """
    ax.set_xlabel('time',fontsize=12)
    ax.set_ylabel('biomass (gr.)',fontsize=12)
    #ax.set_title(f'Simulation {idx}')
    ax.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(f'recip_inv_{idx}_newfig_coex.pdf')
    plt.show()
