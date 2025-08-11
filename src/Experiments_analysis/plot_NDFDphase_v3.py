#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_phase_map_with_CI.py

Phase map: mean ± SEM niche vs. fitness differences,
and transformed phase map with confidence intervals.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ─── File paths ────────────────────────────────────────────────────────
METRICS_CSV = '/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/per_replicate_SC_metrics_pooled.csv'
FBA_CSV     = '/Users/brendonmcguinness/Documents/QLS/python_workspace/fba_stuff/fit_NDFDw0.csv'
# ────────────────────────────────────────────────────────────────────────

def transform_differences(df, base=10):
    """Given df with columns
       'Niche Differences', 'SC Differences', 'sem_niche', 'sem_fitness',
       compute FD_ratio, ND_norm, and leave sem’s for CI propagation."""
    out = df.copy()
    # FD on ratio scale
    out['FD_ratio'] = np.exp(out['SC Differences'])
    # raw ND → normalized [0,1]
    out['ND_norm'] = out['Niche Differences'] / (1 + out['Niche Differences'])
    return out

# ─── 1) Load & aggregate ───────────────────────────────────────────────
df = pd.read_csv(METRICS_CSV)

agg = df.groupby(
    ['strain1','strain2','glucose','succinate'],
    as_index=False
).agg(
    mean_niche   = ('niche_difference',  'mean'),
    sem_niche    = ('niche_difference',  lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    mean_fitness = ('fitness_difference','mean'),
    sem_fitness  = ('fitness_difference',lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    mean_coex    = ('coexistence_strength','mean'),
    sem_coex     = ('coexistence_strength',lambda x: x.std(ddof=1)/np.sqrt(len(x)))
)
# ─── 2) Marker mapping ─────────────────────────────────────────────────
pairs   = list(zip(agg['strain1'], agg['strain2']))
markers = ['o','s','^','v','<','>','P','*']
shape_map = {pair: markers[i % len(markers)] for i, pair in enumerate(pairs)}

# ─── 3) Raw phase map ──────────────────────────────────────────────────
# axis limits
max_nd = (agg['mean_niche'] + agg['sem_niche']).max()
max_fd = (np.abs(agg['mean_fitness']) + agg['sem_fitness']).max()
lim    = max(max_nd, max_fd) * 1.1
xgrid  = np.linspace(-0.5, lim, 200)

fig, ax = plt.subplots(figsize=(6,6))
cmap = plt.get_cmap('coolwarm')
norm = TwoSlopeNorm(
    vmin=agg['mean_coex'].min(),
    vcenter=0.0,
    vmax=agg['mean_coex'].max()
)

for _, row in agg.iterrows():
    pair = (row.strain1, row.strain2)
    mark = ('X' if pair==('ptsG','dctA') and row.glucose==5 and row.succinate==50
            else 'D' if pair==('ptsG','dctA') 
                       and row.glucose==27.5 and row.succinate==27.5
            else shape_map[pair])
    color = cmap(norm(row.mean_coex))

    ax.errorbar(
        row.mean_niche, row.mean_fitness,
        xerr=row.sem_niche, yerr=row.sem_fitness,
        fmt=mark, color=color, ecolor='gray', capsize=4
    )
    lbl = f"{row.strain1},{row.strain2}"
    if row.glucose==5 and row.succinate==50:
        lbl += "\nG=5, S=50"
    ax.annotate(lbl,
                xy=(row.mean_niche, row.mean_fitness),
                xytext=(5,5), textcoords='offset points',
                fontsize=9)

# FBA fits overlay
df_fit = pd.read_csv(FBA_CSV).rename(columns={
    'Niche Differences': 'fit_niche',
    'SC Differences':    'fit_fitness'
})
ko_to_pair = {
    (-10.0, -10.0): ('MG',   'MGK'),
    (-5.0,  -7.0 ): ('manX','dauA'),
    (-2.78, -2.5 ): ('ptsG','dctA'),
}
first = True
for _, r in df_fit.iterrows():
    pair = ko_to_pair.get((float(r.KO1), float(r.KO2)))
    if pair is None: continue
    mark = ('X' if pair==('ptsG','dctA') and np.isclose(r.Concentration,0.005,1e-3)
            else 'D' if pair==('ptsG','dctA')
            else shape_map[pair])
    ax.scatter(r.fit_niche, r.fit_fitness,
               marker=mark, s=100,
               facecolor='none', edgecolor='k',
               linewidth=1.5,
               label='FBA fit' if first else None)
    first = False

ax.plot(xgrid,  xgrid, 'k--', label='FD = +ND')
ax.plot(xgrid, -xgrid, 'k--', label='FD = –ND')

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Mean coexistence strength')
ax.set_xlim(-0.5, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel('Mean niche difference')
ax.set_ylabel('Mean fitness difference')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.legend(loc='upper left', fontsize='small')
plt.tight_layout()


# ─── 4) Transformed phase map with CI ───────────────────────────────────
# prepare transformed data
df_trans = agg.rename(columns={
    'mean_niche':   'Niche Differences',
    'mean_fitness': 'SC Differences'
})
df_trans['sem_coex'] = agg['sem_coex']
df_trans = transform_differences(df_trans)
df_trans['mean_coex'] = df_trans['mean_coex'] /2
# compute ND_norm_max across both data & fits
df_fit_t = df_fit.copy()
df_fit_t['ND_norm'] = df_fit_t['fit_niche'] / (1 + df_fit_t['fit_niche'])
df_fit_t['FD_ratio'] = np.exp(df_fit_t['fit_fitness'])
ND_norm_max = max(df_trans['ND_norm'].max(), df_fit_t['ND_norm'].max())

# stretch
df_trans['ND_norm_stretched'] = df_trans['ND_norm'] / ND_norm_max
df_fit_t['ND_norm_stretched'] = df_fit_t['ND_norm'] / ND_norm_max

# propagate SEM → CI bounds
nd_lo = (df_trans['Niche Differences'] - df_trans['sem_niche'])
nd_hi = (df_trans['Niche Differences'] + df_trans['sem_niche'])
df_trans['ND_norm_lo'] = nd_lo / (1 + nd_lo)
df_trans['ND_norm_hi'] = nd_hi / (1 + nd_hi)
df_trans['ND_lo_str'] = df_trans['ND_norm_lo'] / ND_norm_max
df_trans['ND_hi_str'] = df_trans['ND_norm_hi'] / ND_norm_max

df_trans['FD_lo'] = np.exp(df_trans['SC Differences'] - df_trans['sem_fitness'])
df_trans['FD_hi'] = np.exp(df_trans['SC Differences'] + df_trans['sem_fitness'])

# errors
xerr = np.vstack([
    df_trans['ND_norm_stretched'] - df_trans['ND_lo_str'],
    df_trans['ND_hi_str'] - df_trans['ND_norm_stretched']
])
yerr = np.vstack([
    df_trans['FD_ratio'] - df_trans['FD_lo'],
    df_trans['FD_hi'] - df_trans['FD_ratio']
])

# plot
x_tr = np.linspace(-0.1,1,500) 
ND_raw = (x_tr * ND_norm_max) / (1 - x_tr * ND_norm_max)
y_upper = np.exp(ND_raw)
y_lower = np.exp(-ND_raw)
y_max_input = 5
y_min, y_max = 0, y_max_input #df_trans['FD_ratio'].max()*1.05


# Updated snippet for the “Transformed phase map with CI” block
# (apply these edits in place of the original fig2 / ax2 section)

# … earlier code up through y_lower, y_upper, y_min, y_max_input …

fig2, ax2 = plt.subplots(figsize=(5.5,5.5))

# 1) Shade only between the transformed solid boundaries
ax2.fill_between(x_tr, y_lower, y_upper, facecolor='r', alpha=0.2)
ax2.fill_between(x_tr, y_upper, y_max, facecolor='b', alpha=0.2)
ax2.fill_between(x_tr, y_min, y_lower, facecolor='b', alpha=0.2)

# 2) Plot transformed boundaries as **solid** lines
ax2.plot(x_tr, y_upper, 'k-', lw=2, label='Transformed boundary')
ax2.plot(x_tr, y_lower, 'k-', lw=2)

# 3) Add the **theoretical** FD = 1–ND and FD = 1/(1–ND) as **dashed** lines
x_raw     = np.linspace(-0.1, 0.999, 500)
y_th1     = 1 - x_tr
y_th2     = 1 / (1 - x_tr)
#x_theor   = (x_raw / (1 + x_raw)) / ND_norm_max

ax2.plot(x_raw, y_th1, 'k--', lw=2, label="1 - ND' < FD' < 1/(1 - ND'')")
ax2.plot(x_raw, y_th2, 'k--', lw=2)
first_data = True
# 4) Plot **data points** in dark gray, no legend entries for markers
for _, row in df_trans.iterrows():
    pair = (row.strain1, row.strain2)
    if pair == ('ptsG','dctA'):
        mark = 'X' if (row.glucose == 5.0) else 'D'
    else:
        mark = shape_map[pair]
    ax2.scatter(
        row.ND_norm_stretched,
        row.FD_ratio,
        marker=mark,
        s=120,
        facecolor='dimgray',  # dark‐gray fill
        edgecolor='k',
        alpha=0.9,
        label='Data' if first_data else '_nolegend_'
    )
    first_data = False

# 5) Overlay confidence‐interval errorbars
ax2.errorbar(
    df_trans['ND_norm_stretched'],
    df_trans['FD_ratio'],
    xerr = xerr, yerr = yerr,
    fmt  = 'none',
    ecolor='gray',
    capsize=4,
    alpha=0.7
)

# 6) Plot FBA‐fit points and add only that legend entry
first = True
for _, r in df_fit_t.iterrows():
    pair = ko_to_pair.get((float(r.KO1), float(r.KO2)))
    if pair is None: 
        continue
    mark = ('X' if pair==('ptsG','dctA') and np.isclose(r.Concentration,0.005,1e-3)
            else 'D' if pair==('ptsG','dctA')
            else shape_map[pair])
    ax2.scatter(
        r.ND_norm_stretched,
        r.FD_ratio,
        marker=mark,
        s=120,
        facecolor='none',
        edgecolor='k',
        linewidth=1.5,
        label='FBA fit' if first else '_nolegend_'
    )
    first = False

# 7) Finalize axes, legend, grid, etc.
ax2.set_xlim(-0.1, 1)
ax2.set_ylim(0, y_max_input)
ax2.set_xlabel('Transformed Niche Difference', fontsize=14)
ax2.set_ylabel('Transformed Fitness Difference', fontsize=14)
ax2.legend(loc='upper left', fontsize='medium')
#ax2.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figure5_data_new.pdf')
plt.show()


"""
fig2, ax2 = plt.subplots(figsize=(5.5,5.5))
ax2.fill_between(x_tr, y_lower, y_upper, facecolor='r', alpha=0.2)
ax2.fill_between(x_tr, y_upper, y_max, facecolor='b', alpha=0.2)
ax2.fill_between(x_tr, y_min, y_lower, facecolor='b', alpha=0.2)
ax2.plot(x_tr, y_upper, 'k--', lw=2, label='Coexistence boundary')
ax2.plot(x_tr, y_lower, 'k--', lw=2, label='')

# ─── WITH this per‐row loop:
for _, row in df_trans.iterrows():
    pair = (row.strain1, row.strain2)
    # same toggling logic as before
    if pair == ('ptsG','dctA'):
        mark = 'X' if (row.glucose == 5.0) else 'D'
    else:
        mark = shape_map[pair]

    ax2.scatter(
        row.ND_norm_stretched,
        row.FD_ratio,
        marker=mark,
        s=120,
        facecolor=cmap(norm(row['mean_coex'])),
        edgecolor='k',
        alpha=0.9
    )

# ─── then add a dummy scatter for the legend entry “Data”
ax2.scatter(
    [], [],                          # zero‐length data
    marker='o',                      # pick any shape you like
    s=80,
    facecolor='gray',                # solid gray fill
    edgecolor='k',
    label='Data'
)

from matplotlib.lines import Line2D

# build a small legend that explains what each marker means
marker_legend = [
    Line2D([0], [0],
           marker='o', color='w',
           markerfacecolor='gray', markeredgecolor='k',
           markersize=8,
           label='No niche diff (1G:1S)'),
    Line2D([0], [0],
           marker='s', color='w',
           markerfacecolor='gray', markeredgecolor='k',
           markersize=8,
           label='Low niche diff (1G:1S)'),
    Line2D([0], [0],
           marker='D', color='w',
           markerfacecolor='gray', markeredgecolor='k',
           markersize=8,
           label='High niche diff (1G:1S)'),
    Line2D([0], [0],
           marker='X', color='w',
           markerfacecolor='gray', markeredgecolor='k',
           markersize=8,
           label='High niche diff (1G:10S)'),
]

# add it as a separate legend (you can tweak loc, fontsize, etc.)
leg2 = ax2.legend(handles=marker_legend,
                  title='Marker categories',
                  loc='upper right',
                  frameon=True,
                  fontsize='small')
ax2.add_artist(leg2)

# now your existing legend for "Coexistence boundary", "Data", "FBA fit"...
ax2.legend(loc='lower right', fontsize='small')
# ─── your existing FBA‐fit loop can stay as is, it will add “FBA fit” next
# ─── then call legend():
ax2.legend(loc='upper left', fontsize='small')
ax2.errorbar(
    df_trans['ND_norm_stretched'],
    df_trans['FD_ratio'],
    xerr=xerr,
    yerr=yerr,
    fmt='none', ecolor='gray', capsize=4, alpha=0.7
)

# FBA-fit points
first = True
for _, r in df_fit_t.iterrows():
    pair = ko_to_pair.get((float(r.KO1), float(r.KO2)))
    if pair is None: continue
    mark = ('X' if pair==('ptsG','dctA') and np.isclose(r.Concentration,0.005,1e-3)
            else 'D' if pair==('ptsG','dctA')
            else shape_map[pair])
    ax2.scatter(
        r.ND_norm_stretched, r.FD_ratio,
        marker=mark, s=120,
        facecolor='none', edgecolor='k', linewidth=1.5,
        label='FBA fit' if first else None
    )
    first = False

ax2.set_xlim(-0.1,1)
ax2.set_ylim(0, y_max_input)
ax2.set_xlabel('Niche Difference',fontsize=14)
ax2.set_ylabel('Fitness Difference',fontsize=14)
#ax2.set_title('Transformed Phase Map with CI')
ax2.legend(loc='upper left', fontsize='small')
ax2.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
#plt.savefig('data_fit_transformed_figure3_2legends_2.pdf')
plt.show()

plt.figure(figsize=(6,4))
plt.errorbar(
    df_trans['ND_norm_stretched'],
    df_trans['mean_coex'],
    yerr=df_trans['sem_coex'],
    fmt='o',
    ecolor='gray',
    capsize=4,
    alpha=0.8
)
plt.xlim((-0.1,1))
plt.xlabel('Niche Difference', fontsize=12)
plt.ylabel('Mean Coexistence Strength', fontsize=12)
plt.title('Coexistence vs. Niche Difference', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
"""