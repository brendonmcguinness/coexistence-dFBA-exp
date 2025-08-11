#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:09:44 2025
Refactored to isolate COMETS runs via chdir, add error handling,
avoid Decimal keys, load base model once, and clean up workspaces.
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

from helpers import (
    nicheOverlapSaveTimeSeriesMultipleChooseTime,
    fitnessDifferenceSaveTimeSeriesMultipleRandChooseTime,
)

# --- Configuration ---
os.environ['COMETS_HOME'] = '/Applications/COMETS'
os.environ['GUROBI_COMETS_HOME'] = '/Library/gurobi1003/macos_universal2'
os.environ['GRB_LICENSE_FILE'] = '/Library/gurobi1003/macos_universal2/gurobi.lic'


carbs = [
    "EX_glc__D_e", "EX_fru_e", "EX_gal_e", "EX_xyl__D_e", "EX_succ_e", 
    "EX_mal__L_e", "EX_fum_e", "EX_glyc_e", "EX_cit_e"]

carbon_pairs = list(combinations(carbs, 2))  # Generate all pairs

"""
carbon_pairs = [
    #('EX_glc__D_e', 'EX_succ_e'),
    #('EX_glc__D_e', 'EX_fru_e'),
    ('EX_glc__D_e', 'EX_cit_e')#,
    #('EX_glc__D_e', 'EX_xyl__D_e'),

]
"""
ko_bounds   = np.arange(-10, 1)  # -10, -9, …, 0
#ko_bounds   = np.array([-10,-3,-2,-1,0])
M           = 3 #10
s1_conc     = np.linspace(0.005, 0.05, M)
s2_conc     = s1_conc[::-1]
#exchange_rxns = ["EX_glc__D_e", "EX_fru_e", "EX_succ_e", "EX_cit_e", "EX_xyl__D_e"]
exchange_rxns = [
    "EX_glc__D_e", "EX_fru_e", "EX_gal_e", "EX_man_e", "EX_xyl__D_e", 
    "EX_arab__L_e", "EX_ac_e", "EX_lac__D_e", "EX_pyr_e", "EX_succ_e", 
    "EX_mal__L_e", "EX_fum_e", "EX_glyc_e", "EX_cit_e", "EX_akg_e",
    "EX_eth_e", "EX_for_e"
]
output_dir = 'coex_data_mut1_vs_mut2_refactored_init'
os.makedirs(output_dir, exist_ok=True)

# --- Helpers ---

def make_key(s1_name, s2_name, s1, ko1, ko2):
    return (
        (s1_name, s2_name),
        f"{s1:.4f}",
        ko1,
        ko2
    )

def save_results_to_csv(results, fname):
    rows = []
    for key, vals in results.items():
        (pair, conc_str, ko1, ko2) = key
        src1, src2 = pair
        for idx, v in enumerate(vals):
            rows.append({
                "Carbon Source 1": src1,
                "Carbon Source 2": src2,
                "Concentration": float(conc_str),
                "KO Bound Source 1": ko1,
                "KO Bound Source 2": ko2,
                "KO Strain Index": (idx % 2) + 1,
                "Value": v
            })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, fname), index=False)
    print(f"✅ Saved: {fname}")

def setup_mutants(base, s1, s2, ko1, ko2, cross_bound, rxns):
    mut1 = deepcopy(base)
    mut2 = deepcopy(base)
    for r in rxns:
        mut1.change_bounds(r, cross_bound, 1000)
        mut2.change_bounds(r, cross_bound, 1000)
    mut1.id = f"{s1}_KO_{ko1}"
    mut1.change_bounds(s1, ko1, 1000)
    mut1.change_bounds(s2, -20, 1000)
    mut2.id = f"{s2}_KO_{ko2}"
    mut2.change_bounds(s2, ko2, 1000)
    mut2.change_bounds(s1, -20, 1000)
    return mut1, mut2

def run_in_temp(fn, *args, **kwargs):
    """chdir into a temp dir, run `fn`, then clean up."""
    ws = tempfile.mkdtemp(prefix="comets_run_")
    cwd = os.getcwd()
    try:
        os.chdir(ws)
        return fn(*args, **kwargs)
    finally:
        os.chdir(cwd)
        shutil.rmtree(ws, ignore_errors=True)

# --- Load base model once ---
base_model = c.model(load_model("iJO1366"))

# --- Storage dicts ---
nd, fd, coex, win, mgg, mgl, sc = (
    {}, {}, {}, {}, {}, {}, {}
)

# --- Main loops ---
for s1_name, s2_name in carbon_pairs:
    for ko in ko_bounds:
        #trying with cross feeding reactions any that may come up
        mut1, mut2 = setup_mutants(
            base_model, s1_name, s2_name, ko, ko, 0, exchange_rxns
        )
        for conc1, conc2 in zip(s1_conc, s2_conc):
            key = make_key(s1_name, s2_name, conc1, ko, ko)

            # Niche
            try:
                no, co_val, b1, m1, winner, sc_val = run_in_temp(
                    nicheOverlapSaveTimeSeriesMultipleChooseTime,
                    mut1, mut2, s1_name, s2_name, conc1, conc2, max_cyc=320
                )
            except Exception as e:
                print(f"⚠️  Niche failed {key}: {e}")
                continue

            # Fitness
            """
            try:
                fd_val, b3, m3, mg_grad, mg_slope = run_in_temp(
                    fitnessDifferenceSaveTimeSeriesMultipleRandChooseTime,
                    mut1, mut2, s1_name, s2_name, conc1, conc2, max_cyc=320
                )
            except Exception as e:
                print(f"⚠️  Fitness failed {key}: {e}")
                continue
            """
            nd.setdefault(key, []).append(no)
            coex.setdefault(key, []).append(co_val)
            win.setdefault(key, []).append(winner)
            sc.setdefault(key, []).append(sc_val)
            #fd.setdefault(key, []).append(fd_val)
            #mgg.setdefault(key, []).append(mg_grad)
            #mgl.setdefault(key, []).append(mg_slope)
            #print(key)

# --- Convert & save ---
for dct, fname in [
    (nd,   "niche_differences.csv"),
    #(fd,   "fitness_differences.csv"),
    (coex, "coexistence_results.csv"),
    (win,  "winner_results.csv"),
    #(mgg,  "mgg_diff_results.csv"),
    #(mgl,  "mgl_diff_results.csv"),
    (sc,   "sc_diff_results.csv"),
]:
    for k in dct:
        dct[k] = np.array(dct[k])
    save_results_to_csv(dct, fname)
