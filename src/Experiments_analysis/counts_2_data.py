#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:25:00 2025
@author: brendonmcguinness

Full pipeline: read plate‐summary CSVs, pivot counts so LB+kan never collapse,
compute CFU/mL, apply detection‐limit imputation, assemble Chesson metrics,
and plot selection coefficients with error bars.
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── DILUTION / VOLUME LOGIC ────────────────────────────────────────────────
def choose_dil_and_vol(time_point, plate, strain1=None, strain2=None, rel1=None, rel2=None):
    """
    Returns (dilution_factor, plated_volume):
      - time_point: 0 or 1
      - plate: 'LB' or 'kan'
      - optional strains & rels for special cases
    """
    if time_point == 0:
        return 1e-5, 0.10

    # time_point == 1
    if plate == 'kan':
        # MG→MGK or manX→dauA at 5:95 rare→common
        if rel1 == 95 and rel2 == 5 and (
           (strain1=='MG'   and strain2=='MGK') or
           (strain1=='manX' and strain2=='dauA')):
            return 1e-5, 0.10
        # ptsG→dctA at 5:95
        if rel1 == 95 and rel2 == 5 and (strain1=='ptsG' and strain2=='dctA'):
            return 1e-6, 0.10
        return 1e-6, 0.05  # all other kan at T1

    # LB at T1
    return 1e-6, 0.05

# ── RAW COUNTS LOADING ────────────────────────────────────────────────────
# replace these with your actual T0/T1 directories
folders = {
    '/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/colony_counting/exp_june28_2025/kan_extra_day/T0/tiff/counts_T0_kan': 0,
    '/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/colony_counting/exp_june28_2025/kan_extra_day/T1/tiff/counts_T1_kan': 1,
}
"""
folders = {
    '/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/colony_counting/exp_june28_2025/T0/counts_T0': 0,
    '/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/colony_counting/exp_june28_2025/T1/counts_T1': 1,
}
"""
TOTAL_CONC = 55.0  # mM total glucose+succinate

records = []
for folder, time_pt in folders.items():
    for fp in glob.glob(os.path.join(folder, '*_plate_summary.csv')):
        name = os.path.basename(fp).replace('_plate_summary.csv','')
        plate = 'kan' if name.endswith('_kan') else 'LB'
        if plate=='kan': name = name[:-4]

        parts = name.split('_')
        if len(parts)==5:
            s1,s2,rel1,rel2,rep = parts
            glc = suc = TOTAL_CONC/2
        elif len(parts)==6:
            s1,s2,ratio,rel1,rel2,rep = parts
            m = re.match(r'(\d+)G(\d+)S', ratio)
            gr,sr = map(int, m.groups())
            glc = TOTAL_CONC*gr/(gr+sr)
            suc = TOTAL_CONC*sr/(gr+sr)
        else:
            raise ValueError(f"Bad name parts: {parts}")

        cnt = int(pd.read_csv(fp)['Count'].iloc[0])
        records.append({
            'strain1':    s1,
            'strain2':    s2,
            'rel_init_1': int(rel1),
            'rel_init_2': int(rel2),
            'replicate':  int(rep),
            'time':       time_pt,
            'glucose':    glc,
            'succinate':  suc,
            'plate':      plate,
            'count':      cnt
        })

df = pd.DataFrame(records)

# ── PIVOT TO ONE ROW PER (replicate, time) ───────────────────────────────
summary = (
    df
    .pivot_table(
        index=[
            'strain1','strain2','rel_init_1','rel_init_2',
            'replicate','time','glucose','succinate'
        ],
        columns='plate',
        values='count',
        fill_value=0
    )
    .reset_index()
    .rename(columns={'LB':'count_LB','kan':'count_kan'})
)

# ── DETECTION LIMIT & CFU FUNCTIONS ───────────────────────────────────────
def get_detection_limit(row, which):
    dil, vol = choose_dil_and_vol(
        row['time'], which,
        row['strain1'], row['strain2'],
        row['rel_init_1'], row['rel_init_2']
    )
    return (1.0 / dil) / vol

def get_cfu_per_mL(row, which):
    cnt = row[f'count_{which}'] if row[f'count_{which}']>0 else 1
    dil, vol = choose_dil_and_vol(
        row['time'], which,
        row['strain1'], row['strain2'],
        row['rel_init_1'], row['rel_init_2']
    )
    return cnt * (1.0 / dil) / vol

# ── COMPUTE CFU, IMPUTATION ────────────────────────────────────────────────
summary['cfu_LB'] = summary.apply(get_cfu_per_mL, axis=1, which='LB')
summary['cfu_kan'] = summary.apply(get_cfu_per_mL, axis=1, which='kan')
summary['det_lim_LB'] = summary.apply(get_detection_limit, axis=1, which='LB')
summary['half_lim_LB'] = summary['det_lim_LB']/2.0

summary['ratio_kan'] = summary['cfu_kan']/summary['cfu_LB']
over = summary['ratio_kan']>1
summary['ratio_kan'] = summary['ratio_kan'].clip(upper=1)

summary['N2'] = summary['ratio_kan']*summary['cfu_LB']
summary['N1'] = np.where(over, summary['half_lim_LB'],
                         (1-summary['ratio_kan'])*summary['cfu_LB'])
summary['imputed_N1'] = over

# ── SPLIT, MERGE & GROWTH RATES ────────────────────────────────────────────
keys = ['strain1','strain2','rel_init_1','rel_init_2','replicate']
df0 = summary[summary['time']==0]
df1 = summary[summary['time']==1]
merged = pd.merge(df1, df0, on=keys, suffixes=('_1','_0'))

merged['m1'] = np.log10(merged['N1_1']/merged['N1_0'])
merged['m2'] = np.log10(merged['N2_1']/merged['N2_0'])

# proportions at each timepoint
merged['prop_mut_1'] = merged['cfu_kan_1'] / (
                       merged['cfu_LB_1']
                     + merged['cfu_kan_1'])
merged['prop_mut_0'] = merged['cfu_kan_0'] / (
                       merged['cfu_LB_0']
                     + merged['cfu_kan_0'])

# reconstruct absolute abundances
merged['N2_true_1'] = merged['prop_mut_1'] * (
                        merged['cfu_LB_1']
                      + merged['cfu_kan_1'])
merged['N2_true_0'] = merged['prop_mut_0'] * (
                        merged['cfu_LB_0']
                      + merged['cfu_kan_0'])
# 1) compute wild-type proportions at each timepoint
merged['prop_wt_1'] = merged['cfu_LB_1'] / (
                       merged['cfu_LB_1']
                     + merged['cfu_kan_1'])
merged['prop_wt_0'] = merged['cfu_LB_0'] / (
                       merged['cfu_LB_0']
                     + merged['cfu_kan_0'])

# 2) reconstruct absolute WT abundances
merged['N1_true_1'] = merged['prop_wt_1'] * (
                        merged['cfu_LB_1']
                      + merged['cfu_kan_1'])
merged['N1_true_0'] = merged['prop_wt_0'] * (
                        merged['cfu_LB_0']
                      + merged['cfu_kan_0'])

# 3) compute the “true” WT growth rate
#merged['m1'] = np.log10(merged['N1_true_1'] / merged['N1_true_0'])
# then growth rate
#merged['m2'] = np.log10(merged['N2_true_1'] / merged['N2_true_0'])

# ── ASSEMBLE CHESSON METRICS & PLOT ───────────────────────────────────────
out_rows = []
group_cols = ['strain1','strain2','glucose_1','succinate_1','replicate']

for grp_key, grp in merged.groupby(group_cols):
    if len(grp) != 2:
        print(f"Skipping incomplete replicate {grp_key}: only {len(grp)} rows")
        continue

    rare   = grp[grp['rel_init_1'] < grp['rel_init_2']].iloc[0]
    common = grp[grp['rel_init_1'] > grp['rel_init_2']].iloc[0]
    f1_r = rare['N1_0']/(rare['N1_0']+rare['N2_0'])
    f1_c = common['N1_0']/(common['N1_0']+common['N2_0'])
    
    if abs(f1_r - 0.05) > 0.2:
        f1_r = 1-f1_c
    if abs(f1_c - 0.95) > 0.2:
        f1_c = 1-f1_r
    sc1_r = rare['m1']-rare['m2']
    sc1_c = common['m1']-common['m2']
    sc2_r = common['m2']-common['m1']
    sc2_c = rare['m2']-rare['m1']
    niche = (sc1_r + sc2_r) /(f1_c-f1_r) #0.9
    fit = sc1_r - sc2_r
    coex = min(sc1_r,sc2_r)
    out_rows.append({
        'strain1': grp['strain1'].iloc[0],
        'strain2': grp['strain2'].iloc[0],
        'glucose': grp['glucose_1'].iloc[0],
        'succinate': grp['succinate_1'].iloc[0],
        'replicate': grp['replicate'].iloc[0],
        'f1_rare':   f1_r,
        'f1_common': f1_c,
        'SC1_rare':  sc1_r,
        'SC1_common':sc1_c,
        'SC2_rare':  sc2_r,
        'SC2_common':sc2_c,
        'niche_difference':niche,
        'fitness_difference':fit,
        'coexistence_strength':coex
    })

out = pd.DataFrame(out_rows)
#out.to_csv('per_replicate_SC_metrics_imputed_extraKanpipeline.csv', index=False)

# aggregate & plot
agg = out.groupby(['strain1','strain2','glucose','succinate']).agg(
    f1_rare_mean    = ('f1_rare','mean'),
    f1_rare_sem     = ('f1_rare', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    f1_common_mean  = ('f1_common','mean'),
    f1_common_sem   = ('f1_common', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    SC1_rare_mean   = ('SC1_rare','mean'),
    SC1_rare_sem    = ('SC1_rare', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    SC1_common_mean = ('SC1_common','mean'),
    SC1_common_sem  = ('SC1_common', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    SC2_rare_mean   = ('SC2_rare','mean'),
    SC2_rare_sem    = ('SC2_rare', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    SC2_common_mean = ('SC2_common','mean'),
    SC2_common_sem  = ('SC2_common', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
).reset_index()

for _, r in agg.iterrows():
    x    = [r.f1_rare_mean, r.f1_common_mean]
    xerr = [r.f1_rare_sem,  r.f1_common_sem]

    fig, ax = plt.subplots()
    ax.errorbar(x, [r.SC1_rare_mean, r.SC1_common_mean],
                 xerr=xerr, yerr=[r.SC1_rare_sem, r.SC1_common_sem],
                 fmt='-o', label=r.strain1)
    ax.errorbar(x, [r.SC2_rare_mean, r.SC2_common_mean],
                 xerr=xerr, yerr=[r.SC2_rare_sem, r.SC2_common_sem],
                 fmt='-o', label=r.strain2)

    ax.set_xlim(0,1)
    ax.set_xlabel('Initial strain1 fraction')
    ax.set_ylabel('Selection coefficient')
    ax.set_title(f"{r.strain1} vs {r.strain2} | Glc={r.glucose}, Succ={r.succinate}")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
