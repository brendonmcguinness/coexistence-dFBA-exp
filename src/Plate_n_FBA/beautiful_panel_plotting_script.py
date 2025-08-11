#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_KO_bounds_results.py

…same header…
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# ─── ENV & CONSTANTS ──────────────────────────────────────────────────
# …your COMETS env and a_glc/a_suc/time_step/default_max_cycles…

# ─── USER‐DEFINED BOUNDS ──────────────────────────────────────────────
glc_bounds = {
    'MG':    (10, 10),
    'dauA':  (10, 10),
    'dctA':  (10, 10),
    'manX':  ( 5,  5),
    'ptsG':  (2.78,2.78),
    'manXptsG':  (2.78,2.78)
}
suc_bounds = {
    'MG':    (10, 10),
    'dauA':  ( 7,  7),
    'dctA':  ( 3.5,  3.5),
    'manX':  (10, 10),
    'ptsG':  (10, 10),
    'manXptsG':  (2.78,2.78)
}

# ─── WRAPPERS & HELPERS ───────────────────────────────────────────────
# ─── ENVIRONMENT ──────────────────────────────────────────────────────
os.environ['COMETS_HOME']        = '/Applications/COMETS'
os.environ['GUROBI_COMETS_HOME'] = '/Library/gurobi1003/macos_universal2'
os.environ['GRB_LICENSE_FILE']   = '/Library/gurobi1003/macos_universal2/gurobi.lic'

# ─── CONSTANTS ────────────────────────────────────────────────────────
a_glc, a_suc      = 0.005, 0.0020   # gDW per OD unit
default_max_cycles = 300
time_step          = 0.1            # hours per COMETS cycle

# ─── IMPORT ORIGINAL SIM FUNCTIONS ────────────────────────────────────
from fit_KObound_2_data import (
    run_comets_with_glc_bound as orig_run_glc,
    run_comets_with_succ_bound as orig_run_suc
)
def _odd_window(n, min_allowed=3):
    """Largest odd integer <= n, but at least min_allowed (and odd)."""
    if n < min_allowed:
        return max(3, n if n % 2 == 1 else n-1)
    w = n if n % 2 == 1 else n - 1
    return max(min_allowed, w)

def find_growth_phase(time, od, *, min_od=1e-2):
    """
    Return indices (i_start, i_end) delimiting the growth phase:
    start = first index where OD >= min_od;
    end   = index of global OD maximum after start.
    """
    t = np.asarray(time)
    y = np.asarray(od)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        return None

    t = t[mask]
    y = y[mask]

    # start: first OD >= min_od
    valid = np.where(y >= min_od)[0]
    if valid.size == 0:
        return None
    i_start = int(valid[0])

    # end: global max OD after start
    i_end_rel = int(np.argmax(y[i_start:]))
    i_end = i_start + i_end_rel

    # safety: need at least 3 points in phase
    if (i_end - i_start + 1) < 3:
        return None

    # Return indices in the original (masked) arrays' coordinates
    # (we will use sliced arrays built from t,y returned by this function)
    return i_start, i_end, t, y

def mu_over_growth_phase(time, od, *, min_od=1e-2, poly=2, return_bulk=True):
    """
    Compute μ over the detected growth phase:
      - Smooth ln(OD) over the *entire* phase (window = phase length, odd).
      - μ_max: max gradient of smoothed ln(OD) vs time within the phase.
      - (Optional) μ_bulk: single regression slope of ln(OD) over the phase.

    Returns
    -------
    result : dict with keys
        'mu_max', 't_at_mu', 'mu_bulk', 't_start', 't_end'
        (mu_bulk will be None if return_bulk=False)
    """
    found = find_growth_phase(time, od, min_od=min_od)
    if found is None:
        return {'mu_max': np.nan, 't_at_mu': np.nan,
                'mu_bulk': (np.nan if return_bulk else None),
                't_start': np.nan, 't_end': np.nan}

    i_start, i_end, t_all, y_all = found
    t_phase = t_all[i_start:i_end+1]
    y_phase = y_all[i_start:i_end+1]

    # Logs
    ln_od = np.log(y_phase)

    # Smooth ln(OD) with a window that spans the entire phase
    from scipy.signal import savgol_filter
    w = _odd_window(len(ln_od), min_allowed=3)
    # poly must be < window
    p = min(poly, max(1, w-1))
    ln_od_s = savgol_filter(ln_od, window_length=w, polyorder=p, mode='interp')

    # μ(t): gradient of smoothed ln(OD)
    mu = np.gradient(ln_od_s, t_phase)
    j = int(np.argmax(mu))
    mu_max = float(mu[j])
    t_at_mu = float(t_phase[j])

    # Single-slope bulk μ over phase (optional)
    mu_bulk = None
    if return_bulk:
        # Linear regression ln(OD) = a + (μ_bulk) * t
        # Using polyfit degree 1:
        coeff = np.polyfit(t_phase, ln_od, 1)
        mu_bulk = float(coeff[0])

    return {
        'mu_max': mu_max,
        't_at_mu': t_at_mu,
        'mu_bulk': mu_bulk,
        't_start': float(t_phase[0]),
        't_end': float(t_phase[-1]),
    }


def max_growth_rate(time, od, *, min_od=1e-2, smooth=False, window=11, poly=2):
    """
    Compute max specific growth rate mu_max [h^-1] and the time it occurs.
    Uses numerical gradient of ln(OD) w.r.t. time.
    
    Parameters
    ----------
    time : array-like
        Time vector in hours.
    od : array-like
        OD values (must be > 0 to take logs).
    min_od : float
        Ignore OD values <= min_od to avoid log underflow / noise.
    smooth : bool
        If True, apply Savitzky–Golay smoothing to ln(OD) before gradient.
    window : int
        Window length for Savitzky–Golay (must be odd and <= len(valid data)).
    poly : int
        Polynomial order for Savitzky–Golay.
        
    Returns
    -------
    mu_max : float
        Maximum specific growth rate [h^-1]. np.nan if insufficient data.
    t_at_mu : float
        Time [h] at which mu_max occurs. np.nan if insufficient data.
    """
    t = np.asarray(time)
    y = np.asarray(od)

    mask = np.isfinite(t) & np.isfinite(y) & (y > min_od)
    if mask.sum() < 3:
        return np.nan, np.nan

    t = t[mask]
    ln_od = np.log(y[mask])

    if smooth and len(ln_od) >= max(window, 5):
        # enforce odd window
        if window % 2 == 0:
            window += 1
        from scipy.signal import savgol_filter
        ln_od = savgol_filter(ln_od, window_length=min(window, len(ln_od) - (1 - len(ln_od) % 2)), 
                              polyorder=min(poly, max(1, len(ln_od)-1)), mode='interp')

    mu = np.gradient(ln_od, t)  # [h^-1]
    idx = int(np.argmax(mu))
    return float(mu[idx]), float(t[idx])


# …run_comets_glc_dynamic, run_comets_suc_dynamic, get_exp_OD_at_time, compute_exp_summary…
def run_comets_glc_dynamic(glc_bound, init_bio, t_thresh):
    df = orig_run_glc(glc_bound, init_bio)
    offset = int(t_thresh / time_step)
    rem    = max(1, default_max_cycles - offset)
    return df[df['t'] <= rem * time_step]

def run_comets_suc_dynamic(suc_bound, init_bio, t_thresh):
    df = orig_run_suc(suc_bound, init_bio)
    offset = int(t_thresh / time_step)
    rem    = max(1, default_max_cycles - offset)
    return df[df['t'] <= rem * time_step]

def get_exp_OD_at_time(df, t_manual):
    mask = df['time'] >= t_manual
    if not mask.any():
        raise ValueError(f"No experimental data ≥ time {t_manual}")
    return float(df.loc[mask, 'ODb'].iloc[0])
def compute_exp_summary(df_raw):
    """
    Returns DataFrame with columns ['strain','time','mean_OD','sem_OD']
    computed across all replicates.
    """
    summary = (
        df_raw
        .groupby(['strain','time'], as_index=False)['ODb']
        .agg(['mean','count','std'])
        .rename(columns={'mean':'mean_OD','std':'std_OD'})
        .reset_index()
    )
    summary['sem_OD'] = summary['std_OD'] / np.sqrt(summary['count']) * 1.96
    return summary[['strain','time','mean_OD','sem_OD']]
def generate_simulations(df_res, gluc, succ):
    sim_data = {}
    for _, row in df_res.iterrows():
        strain = row['strain']
        # thresholds from your df_res
        t0g = row['glc_t_threshold']
        t0s = row['suc_t_threshold']

        # look up your user‐specified bounds
        glc_b = glc_bounds[strain][0]
        suc_b = suc_bounds[strain][0]

        # initial biomass as before
        init_g = get_exp_OD_at_time(gluc[gluc['strain']==strain], t0g) * a_glc
        init_s = get_exp_OD_at_time(succ[succ['strain']==strain], t0s) * a_suc

        # run glucose sim
        df_g = run_comets_glc_dynamic(glc_b, init_g, t0g)
        df_g = df_g[df_g['t']>=0].copy()
        df_g['t_shift'] = df_g['t'] + t0g
        df_g['OD']      = df_g['wt'] / a_glc
        sim_data[(strain,'glc')] = df_g[['t_shift','OD']]

        # run succ sim
        df_s = run_comets_suc_dynamic(suc_b, init_s, t0s)
        df_s = df_s[df_s['t']>=0].copy()
        df_s['t_shift'] = df_s['t'] + t0s
        df_s['OD']      = df_s['wt'] / a_suc
        sim_data[(strain,'suc')] = df_s[['t_shift','OD']]

    return sim_data

# ─── LOAD & PREP ──────────────────────────────────────────────────────
df_res    = pd.read_csv('KO_bound_grid_threshold_maunal_2pointsin.csv')
gluc_raw  = pd.read_csv('unprocessed_data_glc_32h.csv')
succ_raw  = pd.read_csv('unprocessed_data_suc_32h.csv')
glc_sum   = compute_exp_summary(gluc_raw)
suc_sum   = compute_exp_summary(succ_raw)
sim_data  = generate_simulations(df_res, gluc_raw, succ_raw)
mu_rows = []  # put this OUTSIDE the loop before the figure is created
labels = {
    'MG':   [r'$manX^{+}\,ptsG^{+}$',
             r'$dauA^{+}\,dctA^{+}$'],
    'dauA': [r'$manX^{+}\,ptsG^{+}$',
             r'$\Delta\,dauA\,dctA^{+}$'],
    'dctA': [r'$manX^{+}\,ptsG^{+}$',
             r'$dauA^{+}\,\Delta\,dctA$'],
    'manX':[r'$\Delta\,manX\,ptsG^{+}$',
            r'$dauA^{+}\,dctA^{+}$'],
    'ptsG':[r'$manX^{+}\,\Delta\,ptsG$',
            r'$dauA^{+}\,dctA^{+}$'],
}
# ─── PLOTTING 5×2 GRID ────────────────────────────────────────────────
plot_strains = [s for s in df_res['strain'] if s!='manXptsG']
fig, axes = plt.subplots(5,2,figsize=(7,12),sharex=True,sharey=True)

for i, strain in enumerate(plot_strains):

    row = df_res.loc[df_res['strain']==strain].iloc[0]
    gsum, ssum = glc_sum[glc_sum['strain']==strain], suc_sum[suc_sum['strain']==strain]
    sim_g, sim_s = sim_data[(strain,'glc')], sim_data[(strain,'suc')]
    """
    mu_sim_g, t_sim_g = max_growth_rate(sim_g['t_shift'].values, sim_g['OD'].values, smooth=True, window=20)  # choose a small odd window
    mu_exp_g, t_exp_g = max_growth_rate(gsum['time'].values,   gsum['mean_OD'].values, smooth=True, window = 20)
    mu_rows.append({
        'strain': strain, 'substrate': 'glc',
        'mu_max_sim_h-1': mu_sim_g, 't_at_mu_sim_h': t_sim_g,
        'mu_max_exp_h-1': mu_exp_g, 't_at_mu_exp_h': t_exp_g
    })
        # --- Succinate panel ---
    mu_sim_s, t_sim_s = max_growth_rate(sim_s['t_shift'].values, sim_s['OD'].values, smooth=True, window= 19)
    mu_exp_s, t_exp_s = max_growth_rate(ssum['time'].values,   ssum['mean_OD'].values, smooth=True, window = 19)
    mu_rows.append({
        'strain': strain, 'substrate': 'suc',
        'mu_max_sim_h-1': mu_sim_s, 't_at_mu_sim_h': t_sim_s,
        'mu_max_exp_h-1': mu_exp_s, 't_at_mu_exp_h': t_exp_s
    })
    """
        # --- μ over growth phase (simulation) ---
    gp_g_sim = mu_over_growth_phase(sim_g['t_shift'].values, sim_g['OD'].values,
                                    min_od=1e-2, poly=1, return_bulk=True)
    gp_s_sim = mu_over_growth_phase(sim_s['t_shift'].values, sim_s['OD'].values,
                                    min_od=1e-2, poly=1, return_bulk=True)

    # --- μ over growth phase (experimental means) ---
    gp_g_exp = mu_over_growth_phase(gsum['time'].values, gsum['mean_OD'].values,
                                    min_od=1e-2, poly=1, return_bulk=True)
    gp_s_exp = mu_over_growth_phase(ssum['time'].values, ssum['mean_OD'].values,
                                    min_od=1e-2, poly=1, return_bulk=True)

    # Collect rows
    mu_rows.append({
        'strain': strain, 'substrate': 'glc',
        'mu_max_sim_h-1': gp_g_sim['mu_max'],
        't_at_mu_sim_h': gp_g_sim['t_at_mu'],
        'mu_bulk_sim_h-1': gp_g_sim['mu_bulk'],
        'phase_start_sim_h': gp_g_sim['t_start'],
        'phase_end_sim_h': gp_g_sim['t_end'],

        'mu_max_exp_h-1': gp_g_exp['mu_max'],
        't_at_mu_exp_h': gp_g_exp['t_at_mu'],
        'mu_bulk_exp_h-1': gp_g_exp['mu_bulk'],
        'phase_start_exp_h': gp_g_exp['t_start'],
        'phase_end_exp_h': gp_g_exp['t_end'],
    })
    mu_rows.append({
        'strain': strain, 'substrate': 'suc',
        'mu_max_sim_h-1': gp_s_sim['mu_max'],
        't_at_mu_sim_h': gp_s_sim['t_at_mu'],
        'mu_bulk_sim_h-1': gp_s_sim['mu_bulk'],
        'phase_start_sim_h': gp_s_sim['t_start'],
        'phase_end_sim_h': gp_s_sim['t_end'],

        'mu_max_exp_h-1': gp_s_exp['mu_max'],
        't_at_mu_exp_h': gp_s_exp['t_at_mu'],
        'mu_bulk_exp_h-1': gp_s_exp['mu_bulk'],
        'phase_start_exp_h': gp_s_exp['t_start'],
        'phase_end_exp_h': gp_s_exp['t_end'],
    })


    glc_label, suc_label = labels[strain]
    # LEFT: glucose
    ax = axes[i,0]
    ax.plot(sim_g['t_shift'], sim_g['OD'], label='sim Glc', linewidth=2)
    ax.errorbar(gsum['time'], gsum['mean_OD'], yerr=gsum['sem_OD'],
                fmt='o', capsize=3, color='k', label='exp ± CI')
    ax.set_yscale('log'); ax.set_ylim(1e-3,1)
    ax.set_title(f'{glc_label} — Glc bound {glc_bounds[strain][0]:.2f}')
    if i==4: ax.set_xlabel('Time (h)')
    ax.set_ylabel('OD600')
    ax.legend(fontsize='small')

    # RIGHT: succinate
    ax = axes[i,1]
    ax.plot(sim_s['t_shift'], sim_s['OD'], label='sim Suc', linewidth=2)
    ax.errorbar(ssum['time'], ssum['mean_OD'], yerr=ssum['sem_OD'],
                fmt='s', capsize=3, color='k', label='exp ± CI')
    ax.set_yscale('log'); ax.set_ylim(1e-3,1)
    ax.set_title(f'{suc_label} — Suc bound {suc_bounds[strain][0]:.2f}')
    if i==4: ax.set_xlabel('Time (h)')
    ax.legend(fontsize='small')

# turn off any unused rows (if <10)
for j in range(len(plot_strains), 5):
    axes[j,0].axis('off'); axes[j,1].axis('off')

plt.tight_layout()
#plt.savefig('FBA_fit_2_data_panelv2.pdf')
plt.show()

# After plt.tight_layout() (or before plt.show())
mu_df = pd.DataFrame(mu_rows)
print(mu_df.sort_values(['strain','substrate']))
