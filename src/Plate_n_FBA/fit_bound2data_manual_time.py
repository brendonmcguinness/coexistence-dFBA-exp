#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid‐search fitting of COMETS flux bounds to OD600 data,
starting analysis at a tunable OD threshold and dynamically
limiting simulation cycles based on that threshold.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fit_KObound_2_data import run_comets_with_glc_bound as orig_run_glc, \
                                    run_comets_with_succ_bound as orig_run_suc

# === Environment Setup ===
os.environ['COMETS_HOME']        = '/Applications/COMETS'
os.environ['GUROBI_COMETS_HOME'] = '/Library/gurobi1003/macos_universal2'
os.environ['GRB_LICENSE_FILE']   = '/Library/gurobi1003/macos_universal2/gurobi.lic'

# === Tunable parameters ===
od_threshold      = 0.01    # OD600 at which the linear range begins
a_glc             = 0.005   # gDW per OD unit for glucose
a_suc             = 0.0020  # gDW per OD unit for succinate
default_max_cycles = 300    # full cycles covering 30 h at time_step
time_step          = 0.1    # hours per COMETS cycle

# === load data ===
#gluc = pd.read_csv('processed_data_glucose_ODb_2.csv')
#succ = pd.read_csv('processed_data_succinate_ODb_2.csv')

#doing with unprocessed data
gluc = pd.read_csv('unprocessed_data_glc_32h.csv')
succ = pd.read_csv('unprocessed_data_suc_32h.csv')

"""
succ = ( succ
            .groupby(['strain','carbon','time'], as_index=False)['ODb']
            .mean() )

gluc = ( gluc
            .groupby(['strain','carbon','time'], as_index=False)['ODb']
            .mean() )
"""
# --- after reading gluc and succ (before converting to mean-only) ---

# Compute mean and SEM by strain, carbon, time
gluc_stats = (
    pd.read_csv('unprocessed_data_glc_32h.csv')
      .groupby(['strain','carbon','time'])
      .ODb
      .agg(['mean','sem'])
      .reset_index()
      .rename(columns={'mean':'ODb_mean','sem':'ODb_sem'})
)

succ_stats = (
    pd.read_csv('unprocessed_data_suc_32h.csv')
      .groupby(['strain','carbon','time'])
      .ODb
      .agg(['mean','sem'])
      .reset_index()
      .rename(columns={'mean':'ODb_mean','sem':'ODb_sem'})
)

# === per‑strain search bounds ===
#glc_bounds = {'MG':(10,10),'dauA':(10,10),'dctA':(10,10),'manX':(4,7),'manXptsG':(0,5),'ptsG':(0.1,5)}
#suc_bounds = {'MG':(10,10),'dauA':(6,8),'dctA':(0,5),'manX':(10,10),'manXptsG':(2,7),'ptsG':(10,10)}

glc_bounds = {'MG':(10,10),'dauA':(10,10),'dctA':(10,10),'manX':(5,6),'manXptsG':(1.0,1.3),'ptsG':(2.5,2.8)}
suc_bounds = {'MG':(10,10),'dauA':(7,7),'dctA':(3,3),'manX':(10,10),'manXptsG':(5,9),'ptsG':(10,10)}

glc_bounds = {'MG':(10,10),'dauA':(10,10),'dctA':(10,10),'manX':(5,5),'ptsG':(2.78,2.78)}
suc_bounds = {'MG':(10,10),'dauA':(7,7),'dctA':(3,3),'manX':(10,10),'ptsG':(10,10)}
#change manXptsG and other bounds 
default_glc = (0,10)
default_suc = (0,10)

# === Helpers ===
# simulation threshold (uses 't' and 'wt')
def get_sim_threshold_time(df, a_conv, od_thresh):
    df = df.copy()
    df['OD'] = df['wt'] / a_conv
    mask = df['OD'] >= od_thresh
    return float(df.loc[mask, 't'].iloc[0]) if mask.any() else np.nan

# experimental threshold (uses 'time' and 'ODb')
def get_exp_threshold_time(df, od_thresh):
    mask = df['ODb'] >= od_thresh
    return float(df.loc[mask, 'time'].iloc[0]) if mask.any() else np.nan

# dynamic-cycle wrappers

def run_comets_glc_dynamic(glc_bound, initial_biomass, t_threshold):
    """
    Run the original COMETS glucose-bound simulation, then truncate
    the result to simulate only up to (default_max_cycles - cycles_offset).
    """
    # run full simulation
    df = orig_run_glc(glc_bound, initial_biomass)
    # compute time cutoff after threshold
    cycles_offset = int(t_threshold / time_step)
    # remaining cycles
    rem_cycles = max(1, default_max_cycles - cycles_offset)
    max_time = rem_cycles * time_step
    # truncate
    return df[df['t'] <= max_time]


def run_comets_suc_dynamic(suc_bound, initial_biomass, t_threshold):
    """
    Run the original COMETS succinate-bound simulation, then truncate
    the result to simulate only up to (default_max_cycles - cycles_offset).
    """
    df = orig_run_suc(suc_bound, initial_biomass)
    cycles_offset = int(t_threshold / time_step)
    rem_cycles = max(1, default_max_cycles - cycles_offset)
    max_time = rem_cycles * time_step
    return df[df['t'] <= max_time]

# hierarchical grid search
def hierarchical_search(run_func, od_obs, a_conv, Y1, Y2,
                        refine_rounds=2, coarse_steps=5, fine_steps=5):
    if Y1 == Y2:
        return Y1, (run_func(Y1).iloc[-1].wt / a_conv - od_obs)**2
    best, bmse = None, np.inf
    low, high = Y1, Y2
    for stage in range(refine_rounds):
        steps = coarse_steps if stage == 0 else fine_steps
        for B in np.linspace(low, high, steps):
            od_sim = run_func(B).iloc[-1].wt / a_conv
            mse = (od_sim - od_obs)**2
            if mse < bmse:
                best, bmse = B, mse
        delta = (high - low) / (steps - 1)
        low, high = max(Y1, best - delta), min(Y2, best + delta)
    return best, bmse

def get_exp_OD_at_time(df, t_manual):
    """Return the first observed ODb at or after t_manual."""
    m = df['time'] >= t_manual
    if not m.any():
        raise ValueError(f"No data ≥ time {t_manual}")
    return float(df.loc[m, 'ODb'].iloc[0])
# --- manually specified threshold times (hours) ---
manual_t0 = {
    'MG':    {'glc':0.48, 'suc':0.48},
    'dauA':  {'glc':0.48, 'suc': 0.48},
    'dctA':  {'glc': 0.48, 'suc': 0.48},
    'manX':  {'glc': 0.48, 'suc': 0.48},
    'manXptsG': {'glc': 0.48, 'suc':0.48},
    'ptsG':  {'glc': 0.48, 'suc':0.48}
    # …and so on for each strain…
}


# === MAIN: perform fit and record threshold times ===
target_strains = sorted(set(gluc['strain']) & set(succ['strain']))
results = []
for strain in target_strains:
    g = gluc[gluc['strain'] == strain]
    s = succ[succ['strain'] == strain]
    if g.empty or s.empty:
        continue

    # experimental threshold time and starting biomass
    #t0_glc = get_exp_threshold_time(g, od_threshold)
    #t0_suc = get_exp_threshold_time(s, od_threshold)
    
    # use manual thresholds instead of computing them
    t0_glc = manual_t0[strain]['glc']
    t0_suc  = manual_t0[strain]['suc']

    # compute initial biomass from the observed ODb at that manual time
    init_bio_glc = get_exp_OD_at_time(g, t0_glc) * a_glc
    init_bio_suc = get_exp_OD_at_time(s, t0_suc) * a_suc

    # search bounds
    Y1_g, Y2_g = glc_bounds.get(strain, default_glc)
    Y1_s, Y2_s = suc_bounds.get(strain, default_suc)

    # wrap COMETS calls with dynamic cycles
    run_glc = lambda B: run_comets_glc_dynamic(B, init_bio_glc, t0_glc)
    run_suc = lambda B: run_comets_suc_dynamic(B, init_bio_suc, t0_suc)

    # observed final OD
    od_final_glc = g['ODb'].iloc[-1]
    od_final_suc = s['ODb'].iloc[-1]

    # fit bounds
    b_g, mse_g = hierarchical_search(run_glc, od_final_glc, a_glc, Y1_g, Y2_g)
    b_s, mse_s = hierarchical_search(run_suc, od_final_suc, a_suc, Y1_s, Y2_s)

    # simulation threshold times for record (if needed)
    df_g = run_glc(b_g)
    df_s = run_suc(b_s)


    results.append({
        'strain': strain,
        'glc_bound': b_g,    'glc_mse': mse_g,
        'suc_bound': b_s,    'suc_mse': mse_s,
        'glc_t_threshold': t0_glc, 'suc_t_threshold': t0_suc
    
    })

# save results
(df_res := pd.DataFrame(results)).to_csv('KO_bound_grid_threshold_maunal_2pointsin_newdctA.csv', index=False)
print("Done with dynamic-cycle fits.")

#df_res = pd.read_csv('KO_bound_grid_threshold_maunal_2pointsin.csv')
# === PLOTTING: overlay data and simulation starting at threshold ===
for strain in df_res['strain']:
    row = df_res[df_res['strain'] == strain].iloc[0]
    g = gluc[gluc['strain'] == strain]
    s = succ[succ['strain'] == strain]

    # experimental crossing times
    t0_glc = row['glc_t_threshold']
    t0_suc = row['suc_t_threshold']

    # starting biomass at threshold
    init_bio_glc = get_exp_OD_at_time(g, t0_glc) * a_glc
    init_bio_suc  = get_exp_OD_at_time(s, t0_suc) * a_suc

    # re-simulate with dynamic cycles
    df_g = run_comets_glc_dynamic(row['glc_bound'], init_bio_glc, t0_glc)
    df_s = run_comets_suc_dynamic(row['suc_bound'], init_bio_suc, t0_suc)

    # compute OD and shifted time
    df_g['OD'] = df_g['wt'] / a_glc
    df_s['OD'] = df_s['wt'] / a_suc
    df_g['t_shift'] = df_g['t'] + t0_glc
    df_s['t_shift'] = df_s['t'] + t0_suc

    # keep only sim-time >= 0
    df_g_plot = df_g[df_g['t'] >= 0]
    df_s_plot = df_s[df_s['t'] >= 0]

    # plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(df_g_plot['t_shift'], df_g_plot['OD'], label='sim Glc')
    plt.scatter(g['time'], g['ODb'], color='k', label='data')
    plt.axvline(t0_glc, linestyle='--', color='gray', label='exp threshold')
    plt.xlabel('Time (h)')
    plt.ylabel('OD600')
    plt.yscale('log')
    plt.ylim((2e-3,1))
    plt.title(f'{strain} – Glc Bound {row["glc_bound"]:.1f}')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(df_s_plot['t_shift'], df_s_plot['OD'], label='sim Suc')
    plt.scatter(s['time'], s['ODb'], color='k', label='data')
    plt.axvline(t0_suc, linestyle='--', color='gray')
    plt.xlabel('Time (h)')
    plt.ylabel('OD600')
    plt.yscale('log')
    plt.ylim((2e-3,1))

    plt.title(f'{strain} – Suc Bound {row["suc_bound"]:.1f}')
    plt.legend()

    plt.tight_layout()
    plt.show()
