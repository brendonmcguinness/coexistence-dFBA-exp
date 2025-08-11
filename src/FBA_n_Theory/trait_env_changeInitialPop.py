
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
"""
carbs = [
    "EX_glc__D_e", "EX_fru_e", "EX_gal_e", "EX_xyl__D_e", "EX_succ_e", 
    "EX_mal__L_e", "EX_fum_e", "EX_glyc_e", "EX_cit_e"
]
"""
carbs = [
    "EX_glc__D_e", "EX_xyl__D_e", "EX_succ_e"
]

carbon_pairs = list(combinations(carbs, 2))
ko_bounds   = np.arange(-10, 1)
M           = 1
s1_conc     = np.linspace(0.005, 0.05, M)
s1_conc = np.array([0.0275,0.0275])
s2_conc     = s1_conc[::-1]
exchange_rxns = [
    "EX_glc__D_e", "EX_fru_e", "EX_gal_e", "EX_man_e", "EX_xyl__D_e", 
    "EX_arab__L_e", "EX_ac_e", "EX_lac__D_e", "EX_pyr_e", "EX_succ_e", 
    "EX_mal__L_e", "EX_fum_e", "EX_glyc_e", "EX_cit_e", "EX_akg_e",
    "EX_eth_e", "EX_for_e"
]
output_dir = 'coex_data_mut1_vs_mut2_refactored_initpop'
os.makedirs(output_dir, exist_ok=True)

init_mean_pops = [5e-6, 5e-5, 5e-4, 5e-3]

# --- Helpers ---
def make_key(s1_name, s2_name, s1, ko1, ko2, pop):
    return ((s1_name, s2_name), f"{s1:.4f}", ko1, ko2, pop)

def save_results_to_csv(results, fname):
    rows = []
    for key, vals in results.items():
        (pair, conc_str, ko1, ko2, pop) = key
        src1, src2 = pair
        for idx, v in enumerate(vals):
            rows.append({
                "Carbon Source 1": src1,
                "Carbon Source 2": src2,
                "Concentration": float(conc_str),
                "KO Bound Source 1": ko1,
                "KO Bound Source 2": ko2,
                "Initial Mean Pop": pop,
                "KO Strain Index": (idx % 2) + 1,
                "Value": v
            })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, fname), index=False)
    print(f"✅ Saved: {fname}")

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

def run_in_temp(fn, *args, **kwargs):
    ws = tempfile.mkdtemp(prefix="comets_run_")
    cwd = os.getcwd()
    try:
        os.chdir(ws)
        return fn(*args, **kwargs)
    finally:
        os.chdir(cwd)
        shutil.rmtree(ws, ignore_errors=True)

def get_log_phase_end_time(df, threshold=1e-5):
    times = {}
    for strain in df.columns[1:-1]:  # skip 'cycle' and 't' columns
        biomass = df[strain].values
        growth_rates = np.diff(biomass) / biomass[:-1]
        below_thresh_indices = np.where(growth_rates < threshold)[0]
        if len(below_thresh_indices) > 0:
            end_cycle = below_thresh_indices[0] + 1
            end_time = df.loc[end_cycle, 't']
        else:
            end_time = df['t'].iloc[-1]
        times[strain] = end_time
    return times

base_model = c.model(load_model("iJO1366"))
nd, coex, win, sc, log_times = {}, {}, {}, {}, {}

for pop in init_mean_pops:
    for s1_name, s2_name in carbon_pairs:
        for ko in ko_bounds:
            mut1, mut2 = setup_mutants(base_model, s1_name, s2_name, ko, ko, 0, exchange_rxns)
            for conc1, conc2 in zip(s1_conc, s2_conc):
                key = make_key(s1_name, s2_name, conc1, ko, ko, pop)
                try:
                    no, co_val, b1, m1, winner, sc_val = run_in_temp(
                        nicheOverlapSaveTimeSeriesMultipleChooseTime,
                        mut1, mut2, s1_name, s2_name, conc1, conc2, max_cyc=320,
                        init_mean_pop=pop
                    )
                    log_times[key] = [get_log_phase_end_time(df) for df in b1]
                except Exception as e:
                    print(f"⚠️  Niche failed {key}: {e}")
                    continue

                nd.setdefault(key, []).append(no)
                coex.setdefault(key, []).append(co_val)
                win.setdefault(key, []).append(winner)
                sc.setdefault(key, []).append(sc_val)

for dct, fname in [
    (nd,   "niche_differences.csv"),
    (coex, "coexistence_results.csv"),
    (win,  "winner_results.csv"),
    (sc,   "sc_diff_results.csv"),
    (log_times, "log_phase_end_times.csv")
]:
    for k in dct:
        dct[k] = np.array(dct[k])
    save_results_to_csv(dct, fname)
