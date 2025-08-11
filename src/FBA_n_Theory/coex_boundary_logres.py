#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results_env_and_ko_separate.py

Load CSVs produced by COMETS driver in `output_dir`, merge metrics (niche, fitness,
coexistence, MGG, MGL, SC), and for a given carbon-pair (cs1, cs2) produce two separate
plots:

  1. `plot_env_splines`: parametric splines in trait-space (Niche vs SC) following
     environment direction (varying concentration ratio) for each KO != ko_start,
     colored by coexistence strength.

  2. `plot_ko_vector_fields`: vector arrows in transformed trait-space with theoretical 
     and transformed Chesson boundaries, and overlay transformed environment splines,
     showing KO direction for each step in KO list (e.g., -10→-9, …, -1→0).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import splprep, splev
import seaborn as sns


def compute_parametric_spline(x, y, s=0.1, num=200):
    deltas = np.hypot(np.diff(x), np.diff(y))
    dist = np.hstack(([0], np.cumsum(deltas)))
    u = dist / dist[-1]
    tck, _ = splprep([x, y], u=u, s=s)
    u_fine = np.linspace(0, 1, num)
    return splev(u_fine, tck)


def transform_differences(df, niche_col="Niche Differences", fitness_col="SC Differences", base=10):
    out = df.copy()
    out['FD_ratio'] = np.exp(out[fitness_col])
#    out['FD_ratio'] = 10 ** (out[fitness_col])
    out['ND_raw'] = base ** out[niche_col]
    out['ND_norm'] = out[niche_col] / (1.0 + out[niche_col])
    
    out['ND_norm_stretched'] = out['ND_norm'] / out['ND_norm'].max()
    out['ND_norm_stretched'] = (
         out['ND_norm'] -  out['ND_norm'].min()
    ) / (
         out['ND_norm'].max() -  out['ND_norm'].min()
    )
    return out


def load_and_merge(output_dir):
    files = {"niche":"niche_differences.csv","fitness":"fitness_differences.csv",
             "coex":"coexistence_results.csv","mgg":"mgg_diff_results.csv",
             "mgl":"mgl_diff_results.csv","sc":"sc_diff_results.csv"}
    col_map = {"niche":"Niche Differences","fitness":"Fitness Differences",
               "coex":"Coexistence Strength","mgg":"MGG Differences",
               "mgl":"MGL Differences","sc":"SC Differences"}
    data = {}
    for key, fname in files.items():
        df = pd.read_csv(os.path.join(output_dir, fname))
        df.drop(columns=["KO Strain Index"], errors='ignore', inplace=True)
        df["Value"] = pd.to_numeric(df["Value"], errors='coerce')
        if key in ["niche","fitness"]:
            df["Value"] = df["Value"].abs()
        df.rename(columns={"Value":col_map[key]}, inplace=True)
        data[key] = df
    merge_keys = ["Carbon Source 1","Carbon Source 2",
                  "KO Bound Source 1","KO Bound Source 2","Concentration"]
    df_all = data["niche"]
    for key in ["fitness","coex","mgg","mgl","sc"]:
        df_all = df_all.merge(data[key], on=merge_keys)
    s1 = np.sort(df_all["Concentration"].unique())
    df_all["Concentration_Carbon_Source_2"] = df_all["Concentration"].map(dict(zip(s1,s1[::-1])))
    return df_all


def plot_ko_vector_fields(df_all, cs1, cs2,
                          x_trait="Niche Differences", y_trait="SC Differences",
                          ko_start=-10):
    df_pair = df_all.query(
        "`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2"
    )
    # transform full slice
    df_t = transform_differences(df_pair, niche_col=x_trait, fitness_col=y_trait)

    fig, ax = plt.subplots(figsize=(8,6))
    norm = TwoSlopeNorm(vmin=df_t["Coexistence Strength"].min(),
                        vcenter=0,
                        vmax=df_t["Coexistence Strength"].max())
    cmap = plt.get_cmap('coolwarm')

    # scatter points
    ax.scatter(
        df_t['ND_norm_stretched'], df_t['FD_ratio'],
        c=df_t["Coexistence Strength"], cmap=cmap, norm=norm,
        alpha=0.8, edgecolor='k'
    )

    # theoretical Chesson boundaries
    x1 = np.linspace(0., 0.999, 2000)
    ax.plot(x1, 1 - x1, 'g--', lw=2, label='1-ND<FD<1/(1-ND)')
    ax.plot(x1, 1/(1 - x1), 'g--', lw=2)

    # transformed boundaries (from raw Niche_Diff units)
    t = np.linspace(0, df_pair[x_trait].max(), 500)
    M = (df_pair[x_trait] / (1 + df_pair[x_trait])).max()
    x_tr = (t/(1 + t)) / M
    ax.plot(x_tr, np.exp(t), 'k--', lw=1.5, label='Transformed FD=ND')
    ax.plot(x_tr, np.exp(-t), 'k--', lw=1.5)

    # overlay transformed environment splines per KO
    ko_list = sorted(df_pair["KO Bound Source 1"].unique())
    for ko in ko_list:
        if ko == ko_start: continue
        sub = df_pair.query(
            "`KO Bound Source 1`==@ko & `KO Bound Source 2`==@ko"
        )
        if len(sub) < 2: continue
        # raw spline in original space
        xs_raw, ys_raw = compute_parametric_spline(
            sub[x_trait].values, sub[y_trait].values, s=0.1
        )
        # transform spline points
        xs_t = (xs_raw / (1 + xs_raw)) / M
        ys_t = np.exp(ys_raw)
        median_cs = sub["Coexistence Strength"].median()
        ax.plot(
            xs_t, ys_t, '-', lw=2,
            color=cmap(norm(median_cs))
        )

    # draw KO vectors in transformed space
    concs = sorted(df_t['Concentration'].unique())
    for i in range(len(ko_list) - 1):
        src, dst = ko_list[i], ko_list[i+1]
        for conc in concs:
            s = df_t.query(
                "`KO Bound Source 1`==@src & Concentration==@conc"
            )
            e = df_t.query(
                "`KO Bound Source 1`==@dst & Concentration==@conc"
            )
            if len(s)==1 and len(e)==1:
                x0, y0 = s['ND_norm_stretched'].iloc[0], s['FD_ratio'].iloc[0]
                dx = e['ND_norm_stretched'].iloc[0] - x0
                dy = e['FD_ratio'].iloc[0] - y0
                cs_val = s["Coexistence Strength"].iloc[0]
                ax.quiver(
                    x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='tail', color=cmap(norm(cs_val)),
                    alpha=0.7, width=0.005
                )

    ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel='Niche Difference',
        ylabel='Fitnes Difference Ratio',
        title=f"Coexistence phase diagram for {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}"
    )
    ax.legend(loc='upper left')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label="Coexistence Strength")
    plt.ylim((-0.2,13.5))
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.show()

def plot_boolean_coexistence(df_all, cs1, cs2,
                              x_trait="Niche Differences", y_trait="SC Differences",
                              ko_start=-10):
    """
    Plot transformed boundaries, light-gray vectors & splines, and boolean dots.
    """
    df_pair = df_all.query(
        "`Carbon Source 1` == @cs1 & `Carbon Source 2` == @cs2"
    )
    df_t = transform_differences(df_pair, niche_col=x_trait, fitness_col=y_trait)
    # boolean flag
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0

    fig, ax = plt.subplots(figsize=(8, 6))
    # larger boolean dots
    colors = df_t['Coex_bool'].map({True: 'red', False: 'blue'})
    ax.scatter(
        df_t['ND_norm_stretched'], df_t['FD_ratio'],
        c=colors, s=100, edgecolor='k'
    )
    # boundaries
    x1 = np.linspace(0., 0.999, 2000)
    ax.plot(x1, 1 - x1, 'g--', lw=2)
    ax.plot(x1, 1/(1 - x1), 'g--', lw=2,label='Boundary: 1-ND<FD<1/(1-ND)')
    t = np.linspace(0, df_pair[x_trait].max(), 500)
    M = (df_pair[x_trait] / (1 + df_pair[x_trait])).max()
    x_tr = (t/(1 + t)) / M
    ax.plot(x_tr, np.exp(t), 'k--', lw=1.5,label='Boundary: Transformed FD=ND')
    ax.plot(x_tr, np.exp(-t), 'k--', lw=1.5)
    # light-gray splines
    ko_list = sorted(df_pair["KO Bound Source 1"].unique())
    for ko in ko_list:
        if ko == ko_start:
            continue
        sub = df_pair.query(
            "`KO Bound Source 1` == @ko & `KO Bound Source 2` == @ko"
        )
        if len(sub) < 2:
            continue
        xs_raw, ys_raw = compute_parametric_spline(
            sub[x_trait].values, sub[y_trait].values, s=0.1
        )
        xs_t = (xs_raw / (1 + xs_raw)) / M
        ys_t = np.exp(ys_raw)
        ax.plot(xs_t, ys_t, color='lightgray', lw=1.5, alpha=0.5)
    # light-gray vectors
    concs = sorted(df_t['Concentration'].unique())
    for i in range(len(ko_list) - 1):
        src, dst = ko_list[i], ko_list[i+1]
        for conc in concs:
            s = df_t.query(
                "`KO Bound Source 1` == @src & Concentration == @conc"
            )
            e = df_t.query(
                "`KO Bound Source 1` == @dst & Concentration == @conc"
            )
            if len(s) == 1 and len(e) == 1:
                x0, y0 = s['ND_norm_stretched'].iloc[0], s['FD_ratio'].iloc[0]
                dx = e['ND_norm_stretched'].iloc[0] - x0
                dy = e['FD_ratio'].iloc[0] - y0
                ax.quiver(
                    x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='tail', color='lightgray', alpha=0.5, width=0.005
                )
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel='Niche Difference',
        ylabel='Fitness Difference Ratio',
        title=f"Boolean Coexistence phase diagram for {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}"
    )
    ax.legend(fontsize='large',loc='upper left')

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.ylim((-0.1,2))
    plt.tight_layout()
    plt.show()

def plot_coex_prob(df_all,cs1,cs2,x_trait="Niche Differences", y_trait="SC Differences"):
    df_pair = df_all.query("`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2")
    df_t = transform_differences(df_pair, niche_col=x_trait, fitness_col=y_trait)
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0
    # 4. Map True/False → string labels, so that Seaborn will show “Coexist” vs “Exclude”
    df_t['Outcome'] = df_t['Coex_bool'].map({True: 'Coexist', False: 'Exclusion'})
    
    prop_df = (
        df_t['Outcome']
        .value_counts(normalize=True)          # get fraction of each label
        .loc[['Coexist', 'Exclusion']]           # ensure this order (won’t error if both exist)
        .rename_axis('Outcome')                # move index into a column
        .reset_index(name='Proportion')        # make “Proportion” column
    )
    
    # 6. Plot a bar chart of proportions
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x='Outcome',
        y='Proportion',
        data=prop_df,
        order=['Coexist', 'Exclusion'],
        palette={'Coexist': 'r', 'Exclusion': 'b'}
    )
    plt.ylim(0, 1)
    plt.ylabel('Probability')
    plt.xlabel('')
    plt.title(f'Coexistence Probability for {cs1} & {cs2}')
    plt.tight_layout()
    plt.show()
    
def plot_boolean_coexistence2_subset(
    df_all, cs1, cs2, ko_focus,
    x_trait="Niche Differences",
    y_trait="SC Differences"
):
    """
    Plot only the "equal-concentration" points for all KO bounds, and the full
    carbon-source ratio sweep (environment spline) for a specific KO bound.

    Points used to fit the spline are now colored by coexistence boolean,
    and purple quiver arrows connect equal-concentration points between consecutive KOs.
    The cyan spline is drawn below the data points using zorder.
    """
    # 1) Subset to this carbon-pair
    df_pair = df_all.query(
        "`Carbon Source 1` == @cs1 & `Carbon Source 2` == @cs2"
    )
    if df_pair.empty:
        raise ValueError(f"No data for {cs1} vs {cs2}")

    # 2) Transform entire slice
    df_t = transform_differences(df_pair, niche_col=x_trait, fitness_col=y_trait)
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0

    # 3) Compute M for stretching ND
    M = (df_pair[x_trait] / (1 + df_pair[x_trait])).max()

    # 4) Identify equal-concentration (midpoint) points:
    sub_mid = df_pair[df_pair['Concentration'] == df_pair['Concentration_Carbon_Source_2']]
    if sub_mid.empty:
        raise ValueError("No equal-concentration data found; check midpoint definitions.")
    sub_mid_t = transform_differences(sub_mid, niche_col=x_trait, fitness_col=y_trait)
    sub_mid_t['Coex_bool'] = sub_mid_t['Coexistence Strength'] > 0

    # 5) Prepare figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # 6) Plot equal-concentration points for all KO bounds
    for flag in [True, False]:
        sub_flag = sub_mid_t[sub_mid_t['Coex_bool'] == flag]
        ax.scatter(
            sub_flag['ND_norm_stretched'], sub_flag['FD_ratio'],
            c=('red' if flag else 'blue'), s=100, edgecolor='k',
            label=('Coexist' if flag else 'Exclusion'), zorder=3
        )

    # 7) Purple quiver arrows connecting equal-concentration points of consecutive KOs
    ko_mid_list = sorted(sub_mid_t['KO Bound Source 1'].unique())
    for i in range(len(ko_mid_list)-1):
        src_ko = ko_mid_list[i]
        dst_ko = ko_mid_list[i+1]
        s = sub_mid_t[sub_mid_t['KO Bound Source 1'] == src_ko]
        e = sub_mid_t[sub_mid_t['KO Bound Source 1'] == dst_ko]
        if len(s)==1 and len(e)==1:
            x0 = (s[x_trait].iloc[0] / (1 + s[x_trait].iloc[0])) / M
            y0 = np.exp(s[y_trait].iloc[0])
            x1 = (e[x_trait].iloc[0] / (1 + e[x_trait].iloc[0])) / M
            y1 = np.exp(e[y_trait].iloc[0])
            dx = x1 - x0
            dy = y1 - y0
            ax.quiver(
                x0, y0, dx, dy,
                angles='xy', scale_units='xy', scale=1,
                pivot='tail', color='purple', alpha=0.7, width=0.005,
                label='KO flux change' if i==0 else '', zorder=2
            )

    # 8) Theoretical boundaries
    x1 = np.linspace(0, 0.999, 2000)
    ax.plot(x1, 1 - x1, 'g--', lw=2, zorder=1)
    ax.plot(x1, 1 / (1 - x1), 'g--', lw=2, label='Boundary: 1-ND<FD<1/(1-ND)', zorder=1)

    # 9) Transformed FD=±ND boundaries
    t = np.linspace(0, df_pair[x_trait].max(), 500)
    df_up = pd.DataFrame({x_trait: t, y_trait: t})
    df_lo = pd.DataFrame({x_trait: t, y_trait: -t})
    df_up_t = transform_differences(df_up, niche_col=x_trait, fitness_col=y_trait, base=10)
    df_lo_t = transform_differences(df_lo, niche_col=x_trait, fitness_col=y_trait, base=10)
    ax.plot(
        df_up_t['ND_norm_stretched'], df_up_t['FD_ratio'], 'k--', lw=1.5,
        label='Boundary: FD = ND (transformed)', zorder=1
    )
    ax.plot(
        df_lo_t['ND_norm_stretched'], df_lo_t['FD_ratio'], 'k--', lw=1.5, zorder=1
    )

    # 10) Plot full environment spline for ko_focus (line only), under data using zorder
    sub_focus = df_pair.query(
        "`KO Bound Source 1` == @ko_focus & `KO Bound Source 2` == @ko_focus"
    )
    if len(sub_focus) < 2:
        raise ValueError(f"Insufficient data for KO = {ko_focus}; need at least 2 points.")
    xs_raw, ys_raw = compute_parametric_spline(
        sub_focus[x_trait].values, sub_focus[y_trait].values, s=0.1
    )
    xs_t = (xs_raw / (1 + xs_raw)) / M
    ys_t = np.exp(ys_raw)
    ax.plot(
        xs_t, ys_t, color='cyan', lw=2,
        label=f'Env spline (KO={ko_focus})', zorder=1
    )

    # 11) Plot only the original data points used for the spline, with red/blue per coexistence
    sub_focus_t = transform_differences(sub_focus, niche_col=x_trait, fitness_col=y_trait, base=10)
    sub_focus_t['Coex_bool'] = sub_focus_t['Coexistence Strength'] > 0
    nd_pts = (sub_focus_t[x_trait] / (1 + sub_focus_t[x_trait])) / M
    fd_pts = np.exp(sub_focus_t[y_trait])
    for flag in [True, False]:
        mask = sub_focus_t['Coex_bool'] == flag
        ax.scatter(
            nd_pts[mask], fd_pts[mask],
            c=('red' if flag else 'blue'), edgecolor='k', s=80,
            label=('Spline data Coexist' if flag else 'Spline data Exclusion'), zorder=3
        )

    # 12) Final formatting
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel='ND_norm_stretched',
        ylabel='FD_ratio'
    )
    ax.legend(fontsize='medium', loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.ylim((-0.1, 2))
    plt.tight_layout()
    plt.show()
    


def plot_boundaries_only(df_all, cs1, cs2,
                         x_trait="Niche Differences",
                         y_trait="SC Differences"):
    """
    Plots only:
      • Raw boundaries: FD = 1 - ND  and  FD = 1/(1-ND)  (green dashed)
      • Transformed boundaries: FD=ND and FD=-ND in the transformed space (black dashed)
    """
    # 1) subset + transform
    df_pair = df_all.query(
        "`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2"
    )
    df_t = transform_differences(
        df_pair, niche_col=x_trait, fitness_col=y_trait, base=10
    )

    fig, ax = plt.subplots(figsize=(8,6))

    # 2) raw‐trait boundaries (green dashed)
    x1 = np.linspace(0, 0.999, 2000)
    ax.plot(x1, 1 - x1,       'g--', lw=2, label='FD = 1 - ND')
    ax.plot(x1, 1/(1 - x1),   'g--', lw=2, label='FD = 1/(1 - ND)')

    # 3) transformed boundaries (black dashed)
    t = np.linspace(0, df_pair[x_trait].max(), 500)
    df_up   = pd.DataFrame({ x_trait: t, y_trait:  t   })
    df_lo   = pd.DataFrame({ x_trait: t, y_trait: -t   })
    df_up_t = transform_differences(df_up,
                                    niche_col=x_trait,
                                    fitness_col=y_trait,
                                    base=10)
    df_lo_t = transform_differences(df_lo,
                                    niche_col=x_trait,
                                    fitness_col=y_trait,
                                    base=10)

    ax.plot(df_up_t['ND_norm_stretched'],
            df_up_t['FD_ratio'],
            'k--', lw=1.5,
            label='FD = +ND (transformed)')
    ax.plot(df_lo_t['ND_norm_stretched'],
            df_lo_t['FD_ratio'],
            'k--', lw=1.5,
            label='FD = -ND (transformed)')

    # 4) finalize
    ax.set_xlabel('ND_norm_stretched')
    ax.set_ylabel('FD_ratio')
    ax.set_ylim(0,6)
    ax.set_xlim(-0.05,1)

    ax.set_title(f"{cs1.replace('EX_','')} vs {cs2.replace('EX_','')}: Boundaries Only")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_shading_with_raw_theory(df_all, cs1, cs2,
                                 x_trait="Niche Differences",
                                 y_trait="SC Differences"):
    """
    • Shade between transformed ±ND in red, outside in blue.
    • Plot raw theoretical FD=1-ND & FD=1/(1-ND) (only x is stretched,
      y is left raw) as black dashed.
    """
    # 1) subset & full transform for your data‐driven boundaries
    df_pair = df_all.query(
        "`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2"
    )
    df_t = transform_differences(
        df_pair, niche_col=x_trait, fitness_col=y_trait, base=10
    )

    # build the two *transformed* curves FD=±ND
    t = np.linspace(0, df_pair[x_trait].max(), 500)
    df_up   = pd.DataFrame({ x_trait: t, y_trait:  t   })
    df_lo   = pd.DataFrame({ x_trait: t, y_trait: -t   })
    df_up_t = transform_differences(df_up, niche_col=x_trait,
                                    fitness_col=y_trait, base=10)
    df_lo_t = transform_differences(df_lo, niche_col=x_trait,
                                    fitness_col=y_trait, base=10)

    x_tr    = df_up_t['ND_norm_stretched']
    y_upper = df_up_t['FD_ratio']
    y_lower = df_lo_t['FD_ratio']

    # 2) compute the raw-theoretical curves
    x_raw = np.linspace(0, 0.999, 2000)
    y_th1 = 1 - x_raw
    y_th2 = 1 / (1 - x_raw)

    # but stretch *only* the x-coordinate so it lives on your ND_norm_stretched axis:
    M = (df_pair[x_trait] / (1 + df_pair[x_trait])).max()
    x_theor_stretched = (x_raw / (1.0 + x_raw)) / M

    # 3) pick y-limits from your transformed FD data
    y_min, y_max = df_t['FD_ratio'].min(), df_t['FD_ratio'].max()
    pad = 0.1 * (y_max - y_min)
    y_min, y_max = y_min - pad, y_max + pad

    # 4) plot
    fig, ax = plt.subplots(figsize=(6,5))

    # a) red wedge between transformed ±ND
    ax.fill_between(
        x_tr, y_lower, y_upper,
        facecolor='r', alpha=0.2,
        label='Transformed coexistence'
    )
    # b) blue above
    ax.fill_between(
        x_tr, y_upper, y_max,
        facecolor='b', alpha=0.2,
        label='Transformed exclusion'
    )
    # c) blue below
    ax.fill_between(
        x_tr, y_min, y_lower,
        facecolor='b', alpha=0.2,
        label='_nolegend_'
    )

    # d) raw theoretical boundaries (black dashed), un‐transformed in y
    ax.plot(
        x_raw, y_th1,
        'k--', lw=2, label="1 - ND' < FD' <  1/(1 - ND')"
    )
    ax.plot(
        x_raw, y_th2,
        'k--', lw=2
    )
    ax.plot(x_tr, y_upper,'k-',lw=2,label="Transformed boundary")
    ax.plot(x_tr, y_lower,'k-',lw=2)

    # 5) finish
    ax.set_xlim(0, x_tr.max())
    ax.set_ylim(-0.1, 4.2)
    ax.set_xlabel("Transformed Niche Difference (ND')",fontsize=14)
    ax.set_ylabel("Transformed Fitness Difference (FD')",fontsize=14)
    #ax.set_title(f"{cs1.replace('EX_','')} vs {cs2.replace('EX_','')}")
    ax.legend(loc='best',fontsize=13.5)
    #ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('transformed_ex_fig2_v2.pdf')
    plt.show()

def plot_shading_with_raw_theory_andPoints(df_all, cs1, cs2,
                                 x_trait="Niche Differences",
                                 y_trait="SC Differences",ko_start=-10):
    """
    • Shade between transformed ±ND in red, outside in blue.
    • Plot raw theoretical FD=1-ND & FD=1/(1-ND) (only x is stretched,
      y is left raw) as black dashed.
    """
    min_y, max_y = -0.05, 4.2 #13.3 #2.3 # 13.3 # for succ2.3
    min_x = -0.03
    # boolean point colors
    coex_colors = {True: 'red', False: 'blue'}
    spline_color = 'cyan'
    vector_color = 'purple'
    
    # 1) subset & full transform for your data‐driven boundaries
    df_pair = df_all.query(
        "`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2"
    )
    df_t = transform_differences(
        df_pair, niche_col=x_trait, fitness_col=y_trait, base=10
    )
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0

    # build the two *transformed* curves FD=±ND
    t = np.linspace(0, df_pair[x_trait].max(), 2000)
    df_up   = pd.DataFrame({ x_trait: t, y_trait:  t   })
    df_lo   = pd.DataFrame({ x_trait: t, y_trait: -t   })
    df_up_t = transform_differences(df_up, niche_col=x_trait,
                                    fitness_col=y_trait, base=10)
    df_lo_t = transform_differences(df_lo, niche_col=x_trait,
                                    fitness_col=y_trait, base=10)

    x_tr    = df_up_t['ND_norm_stretched']
    y_upper = df_up_t['FD_ratio']
    y_lower = df_lo_t['FD_ratio']

    # 2) compute the raw-theoretical curves
    x_raw = np.linspace(min_x, 0.999, 2000)
    y_th1 = 1 - x_raw
    y_th2 = 1 / (1 - x_raw)

    # but stretch *only* the x-coordinate so it lives on your ND_norm_stretched axis:
    M = (df_pair[x_trait] / (1 + df_pair[x_trait])).max()
    x_theor_stretched = (x_raw / (1.0 + x_raw)) / M

    # 3) pick y-limits from your transformed FD data
    y_min, y_max = df_t['FD_ratio'].min(), df_t['FD_ratio'].max()
    pad = 0.1 * (y_max - y_min)
    y_min, y_max = y_min - pad, y_max + pad

    # 4) plot
    fig, ax = plt.subplots(figsize=(6,5))

    # a) red wedge between transformed ±ND
    ax.fill_between(
        x_tr, y_lower, y_upper,
        facecolor='r', alpha=0.2#,
        #label='Transformed coexistence'
    )
    # b) blue above
    ax.fill_between(
        x_tr, y_upper, y_max,
        facecolor='b', alpha=0.2#,
        #label='Transformed exclusion'
    )
    # c) blue below
    ax.fill_between(
        x_tr, y_min, y_lower,
        facecolor='b', alpha=0.2,
        label='_nolegend_'
    )
    # c) blue below
    # you need your y‐limits first:
    y_min, y_max = ax.get_ylim()  # or whatever you set them to
    
    # create an array of y
    yy = np.linspace(y_min, y_max, 10000)
    
    ax.fill_betweenx(
        yy,
        min_x,
        0,
        facecolor='blue',
        alpha=0.2,
        linewidth=0,      # <-- no outline
        edgecolor='none', # <-- absolutely no edge
        zorder=0
    )
    
    # ...follow with your other plotting calls...

    # d) raw theoretical boundaries (black dashed), un‐transformed in y
    ax.plot(
        x_raw, y_th1,
        'k--', lw=2, label="1 - ND' < FD' <  1/(1 - ND')"
    )
    ax.plot(
        x_raw, y_th2,
        'k--', lw=2
    )
    ax.plot(x_tr, y_upper,'k-',lw=2,label="Transformed boundary")
    ax.plot(x_tr, y_lower,'k-',lw=2)

    # boolean points
    for flag in [True, False]:
        sub = df_t[df_t['Coex_bool'] == flag]
        ax.scatter(
            sub['ND_norm_stretched'], sub['FD_ratio'],
            c=coex_colors[flag], s=100, edgecolor='k',
            label='Coexist' if flag else 'Exclusion',clip_on=False
        )

    # boundaries (unchanged)
    """
    x1 = np.linspace(-0.5, 0.999, 2000)
    ax.plot(x1, 1 - x1, 'k--', lw=2)
    ax.plot(x1, 1/(1 - x1), 'k--', lw=2,
            label='Boundary')
    """
    # … the rest of your spline/vector code …
    M = (df_pair[x_trait]/(1+df_pair[x_trait])).max()
    #x_tr = (t/(1.0+t)) / M
    # splines in olive
    spline_label = False
    ko_list = sorted(df_pair['KO Bound Source 1'].unique())
    for ko in ko_list:
        if ko == ko_start: continue
        sub = df_pair.query("`KO Bound Source 1`==@ko & `KO Bound Source 2`==@ko")
        if len(sub) < 2: continue
        xs_raw, ys_raw = compute_parametric_spline(
            sub[x_trait].values, sub[y_trait].values, s=0.1
        )
        xs_t = (xs_raw/(1+xs_raw)) / M
        ys_t = np.exp(ys_raw)
        ax.plot(
            xs_t, ys_t, color=spline_color, lw=2, alpha=0.6,
            label='Relative concentration' if not spline_label else ''
        )
        spline_label = True

    # vectors in purple
    vector_label = False
    concs = sorted(df_t['Concentration'].unique())
    for i in range(len(ko_list) - 1):
        src, dst = ko_list[i], ko_list[i+1]
        for conc in concs:
            s = df_t.query("`KO Bound Source 1`==@src & Concentration==@conc")
            e = df_t.query("`KO Bound Source 1`==@dst & Concentration==@conc")
            if len(s)==1 and len(e)==1:
                x0, y0 = s['ND_norm_stretched'].iloc[0], s['FD_ratio'].iloc[0]
                dx = e['ND_norm_stretched'].iloc[0] - x0
                dy = e['FD_ratio'].iloc[0] - y0
                ax.quiver(
                    x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='tail', color=vector_color, alpha=0.4, width=0.006,
                    label='Uptake flux change' if not vector_label else ''
                )
                vector_label = True

    
    #ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel="Transformed Niche Difference (ND')",
        ylabel="Transformed Fitness Difference (FD')"#,
        #title=f"Boolean coexistence & changes: {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}"
    )
    ax.legend(fontsize='medium', loc='upper left')
    #ax.grid(True, linestyle='--', alpha=0.5)
    plt.ylim((-0.05, 4.2)) # 4.2 13.3
    plt.xlim((min_x,1.00))
    plt.tight_layout()
    plt.savefig('gluc_succ_allpoints_fig3new.pdf')
    plt.show()
    
    
def plot_boolean_coexistence2(df_all, cs1, cs2,
                              x_trait="Niche Differences", y_trait="SC Differences",
                              ko_start=-10):
    """
    Plot transformed trait-space with boolean coexistence:
      • Points colored red/blue by flag
      • Splines in olive labeled 'Relative supply point change'
      • Vectors in purple labeled 'Uptake flux change'
      • Overlaid theoretical & transformed boundaries
      • No colorbar
    """
    df_pair = df_all.query("`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2")
    df_t = transform_differences(df_pair, niche_col=x_trait, fitness_col=y_trait)
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0

    # boolean point colors (keep red/blue)
    coex_colors = {True: 'red', False: 'blue'}
    # line colors
    spline_color = 'cyan'
    vector_color = 'purple'

    fig, ax = plt.subplots(figsize=(8, 6))
    # plot boolean points
    for flag in [True, False]:
        sub = df_t[df_t['Coex_bool'] == flag]
        ax.scatter(
            sub['ND_norm_stretched'], sub['FD_ratio'],
            c=coex_colors[flag], s=100, edgecolor='k',
            label='Coexist' if flag else 'Exclusion'
        )

    # boundaries (unchanged)
    x1 = np.linspace(0, 0.999, 2000)
    ax.plot(x1, 1 - x1, 'g--', lw=2)
    ax.plot(x1, 1/(1 - x1), 'g--', lw=2,label='Boundary: 1-ND<FD<1/(1-ND)')
    t = np.linspace(0, df_pair[x_trait].max(), 500)
    
    df_up   = pd.DataFrame({ x_trait: t, y_trait: t })
    df_up_t = transform_differences(df_up, niche_col=x_trait, fitness_col=y_trait, base=10)
    # 2) the “lower” boundary: fitness_col = – niche_col
    df_lo   = pd.DataFrame({ x_trait: t, y_trait: -t })
    df_lo_t = transform_differences(df_lo, niche_col=x_trait, fitness_col=y_trait, base=10)

    M = (df_pair[x_trait]/(1+df_pair[x_trait])).max()
    x_tr = (t/(1.0+t)) / M
    #ax.plot(x_tr, np.exp(t), 'k--', lw=1.5)
    #ax.plot(x_tr, np.exp(-t), 'k--', lw=1.5,label='Boundary: Transformed FD=ND')
    
    # plot them
    ax.plot(df_up_t['ND_norm_stretched'], df_up_t['FD_ratio'], 'k--', lw=1.5,
            label='Boundary: FD = ND (transformed)')
    ax.plot(df_lo_t['ND_norm_stretched'], df_lo_t['FD_ratio'], 'k--', lw=1.5)
#    ax.plot(x_tr, 10**(t), 'k--', lw=1.5)
#    ax.plot(x_tr, 10**(-t), 'k--', lw=1.5,label='Boundary: Transformed FD=ND')

    # splines in olive
    spline_label = False
    ko_list = sorted(df_pair['KO Bound Source 1'].unique())
    for ko in ko_list:
        if ko == ko_start: continue
        sub = df_pair.query("`KO Bound Source 1`==@ko & `KO Bound Source 2`==@ko")
        if len(sub) < 2: continue
        xs_raw, ys_raw = compute_parametric_spline(
            sub[x_trait].values, sub[y_trait].values, s=0.1
        )
        xs_t = (xs_raw/(1+xs_raw)) / M
        ys_t = np.exp(ys_raw)
        ax.plot(
            xs_t, ys_t, color=spline_color, lw=1.5, alpha=0.6,
            label='Relative supply point change' if not spline_label else ''
        )
        spline_label = True

    # vectors in purple
    vector_label = False
    concs = sorted(df_t['Concentration'].unique())
    for i in range(len(ko_list) - 1):
        src, dst = ko_list[i], ko_list[i+1]
        for conc in concs:
            s = df_t.query("`KO Bound Source 1`==@src & Concentration==@conc")
            e = df_t.query("`KO Bound Source 1`==@dst & Concentration==@conc")
            if len(s)==1 and len(e)==1:
                x0, y0 = s['ND_norm_stretched'].iloc[0], s['FD_ratio'].iloc[0]
                dx = e['ND_norm_stretched'].iloc[0] - x0
                dy = e['FD_ratio'].iloc[0] - y0
                ax.quiver(
                    x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='tail', color=vector_color, alpha=0.4, width=0.004,
                    label='Uptake flux change' if not vector_label else ''
                )
                vector_label = True

    ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel='ND_norm_stretched', ylabel='FD_ratio',
        title=f"Boolean coexistence & changes: {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}"
    )
    ax.legend(fontsize='large', loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    #plt.ylim((-0.2,13.5))
    plt.ylim((-0.1,2))
    plt.tight_layout()
    plt.show()
    
def plot_boolean_coexistence3(df_all, cs1, cs2,
                              x_trait="Niche Differences", y_trait="SC Differences",
                              ko_start=-10):
    """
    Plot transformed trait-space with boolean coexistence:
      • Points colored red/blue by flag
      • Splines in olive labeled 'Relative supply point change'
      • Vectors in purple labeled 'Uptake flux change'
      • Overlaid theoretical & transformed boundaries
      • No colorbar
    """
    df_pair = df_all.query("`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2")
    df_t = transform_differences(df_pair, niche_col=x_trait, fitness_col=y_trait)
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0

    # boolean point colors
    coex_colors = {True: 'red', False: 'blue'}
    spline_color = 'cyan'
    vector_color = 'purple'

    fig, ax = plt.subplots(figsize=(6.3, 5))

    # ─── NEW: fill coexistence band ─────────────────────
    xx_fill = np.linspace(-0.05, 1.0, 2000)
    y_bottom = 1 - xx_fill
    y_top    = 1 / (1 - xx_fill)
    # choose the y-limits you’ll enforce later
    min_y, max_y = -0.05, 2.3 # 13.3 # for succ2.3

    ax.fill_between(xx_fill, y_bottom, y_top, color='r', alpha=0.2)
    ax.fill_between(xx_fill, min_y, y_bottom, color='b', alpha=0.2)
    ax.fill_between(xx_fill, y_top, max_y, color='b', alpha=0.2)
    # ────────────────────────────────────────────────────

    # boolean points
    for flag in [True, False]:
        sub = df_t[df_t['Coex_bool'] == flag]
        ax.scatter(
            sub['ND_norm_stretched'], sub['FD_ratio'],
            c=coex_colors[flag], s=100, edgecolor='k',
            label='Coexist' if flag else 'Exclusion'
        )

    # boundaries (unchanged)
    x1 = np.linspace(-0.5, 0.999, 2000)
    ax.plot(x1, 1 - x1, 'k--', lw=2)
    ax.plot(x1, 1/(1 - x1), 'k--', lw=2,
            label='Boundary')

    # … the rest of your spline/vector code …
    M = (df_pair[x_trait]/(1+df_pair[x_trait])).max()
    #x_tr = (t/(1.0+t)) / M
    # splines in olive
    spline_label = False
    ko_list = sorted(df_pair['KO Bound Source 1'].unique())
    for ko in ko_list:
        if ko == ko_start: continue
        sub = df_pair.query("`KO Bound Source 1`==@ko & `KO Bound Source 2`==@ko")
        if len(sub) < 2: continue
        xs_raw, ys_raw = compute_parametric_spline(
            sub[x_trait].values, sub[y_trait].values, s=0.1
        )
        xs_t = (xs_raw/(1+xs_raw)) / M
        ys_t = np.exp(ys_raw)
        ax.plot(
            xs_t, ys_t, color=spline_color, lw=1.5, alpha=0.6,
            label='Relative supply point change' if not spline_label else ''
        )
        spline_label = True

    # vectors in purple
    vector_label = False
    concs = sorted(df_t['Concentration'].unique())
    for i in range(len(ko_list) - 1):
        src, dst = ko_list[i], ko_list[i+1]
        for conc in concs:
            s = df_t.query("`KO Bound Source 1`==@src & Concentration==@conc")
            e = df_t.query("`KO Bound Source 1`==@dst & Concentration==@conc")
            if len(s)==1 and len(e)==1:
                x0, y0 = s['ND_norm_stretched'].iloc[0], s['FD_ratio'].iloc[0]
                dx = e['ND_norm_stretched'].iloc[0] - x0
                dy = e['FD_ratio'].iloc[0] - y0
                ax.quiver(
                    x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='tail', color=vector_color, alpha=0.4, width=0.004,
                    label='Uptake flux change' if not vector_label else ''
                )
                vector_label = True

    
    #ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel='Niche Difference',
        ylabel='Fitness Difference'#,
        #title=f"Boolean coexistence & changes: {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}"
    )
    ax.legend(fontsize='medium', loc='upper left')
    #ax.grid(True, linestyle='--', alpha=0.5)
    plt.ylim((-0.05, max_y))
    plt.xlim((-0.05,1.00))
    plt.tight_layout()
    plt.savefig('gluc_succ_allpoints_fig3new.pdf')
    plt.show()
    
def plot_lv_nd_fd(n_samples=200, low=0.0, high=1.0, seed=None):
    """
    Sample LV alphas in [0,1] and compute Chesson's ND and fitness ratio (FD) without log.
    Scatter ND vs FD showing coexistence where 1-ND < FD < 1/(1-ND) in red and
    exclusion otherwise in blue, overlaying the coexistence boundary band.
    """
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(low, high, size=(n_samples, 4))
    # compute ND and FD for each parameter set
    a11, a12, a21, a22 = alphas.T
    rho = np.sqrt((a12 * a21) / (a11 * a22))
    ND = 1.0 - rho
    FD = np.sqrt((a11 * a12) / (a22 * a21))

    # define coexistence condition: 1-ND < FD < 1/(1-ND)
    lower = 1.0 - ND
    upper = 1.0 / (1.0 - ND)
    coex_mask = (FD > lower) & (FD < upper)

    plt.figure(figsize=(6,6))
    # plot coexistence points in red and exclusion in blue
    plt.scatter(ND[coex_mask], FD[coex_mask], c='red', s=30, alpha=0.7, label='Coexistence')
    plt.scatter(ND[~coex_mask], FD[~coex_mask], c='blue', s=30, alpha=0.7, label='Exclusion')

    # overlay coexistence boundary curves
    x = np.linspace(0, 1, 200)
    plt.plot(x, 1 - x, 'k--', lw=1.5, label='FD = 1 - ND')
    plt.plot(x, 1.0/(1.0 - x), 'k--', lw=1.5, label='FD = 1/(1 - ND)')

    plt.xlabel('ND = 1 - √(a12·a21 / a11·a22)')
    plt.ylabel('FD = √(a11·a12 / a22·a21)')
    plt.title('LV samples: Niche vs Fitness ratio with Chesson boundaries')
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, ls='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 6)
    plt.tight_layout()
    plt.show()
    
def plot_lv_nd_fd_offdiag(a11=1.0, a22=1.0, n_samples=200, seed=None):
    """
    Fix diagonal LV coefficients a11 and a22, sample off-diagonals a12 and a21 in [0,1],
    compute Chesson's ND and FD, and scatter with coexistence band coloring.
    """
    rng = np.random.default_rng(seed)
    a12 = rng.uniform(0.0, 1.0, size=n_samples)
    a21 = rng.uniform(0.0, 1.0, size=n_samples)
    a11_arr = np.full(n_samples, a11)
    a22_arr = np.full(n_samples, a22)
    # compute ND and FD
    rho = np.sqrt((a12 * a21) / (a11_arr * a22_arr))
    ND = 1.0 - rho
    FD = np.sqrt((a11_arr * a12) / (a22_arr * a21))
    # coexistence condition
    lower = 1.0 - ND
    upper = 1.0 / (1.0 - ND)
    coex_mask = (FD > lower) & (FD < upper)
    plt.figure(figsize=(6,6))
    plt.scatter(ND[coex_mask], FD[coex_mask], c='red', s=30, alpha=0.7, label='Coexistence')
    plt.scatter(ND[~coex_mask], FD[~coex_mask], c='blue', s=30, alpha=0.7, label='Exclusion')
    x = np.linspace(0,1,200)
    plt.plot(x, 1 - x, 'k--', lw=1.5)
    plt.plot(x, 1/(1 - x), 'k--', lw=1.5)
    plt.xlabel('ND = 1 - √(a12·a21 / a11·a22)')
    plt.ylabel('FD = √(a11·a12 / a22·a21)')
    plt.title(f'LV off-diagonal sampling (a11={a11}, a22={a22})')
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, ls='--', alpha=0.5)
    plt.xlim(0,1)
    plt.ylim(0,6)
    plt.tight_layout()
    plt.show()

# --- New: strategic ND-FD sampling via inversion ---
def sample_alphas_from_nd_fd(ND_vals, FD_vals, a11=1.0, a22=1.0):
    """
    Given arrays of target ND and FD and arbitrary self-regulation a11, a22,
    compute off-diagonal LV alphas (a12,a21) via:
      rho = 1 - ND,
      A = rho^2 * a11 * a22,
      B = FD^2 * (a22 / a11),
      a12 = sqrt(A * B) = rho * FD * a22,
      a21 = sqrt(A / B) = rho * a11 / FD.
    Filters for 0 <= a12,a21 <= 1. Returns alphas shape (m,4) and mask of length ND_vals.
    """
    rho = 1.0 - ND_vals
    # compute off-diagonals satisfying both niche and fitness constraints
    a12 = rho * FD_vals * a22
    a21 = rho * a11 / FD_vals
    # stack full alpha vectors
    alphas = np.vstack([np.full_like(rho, a11), a12, a21, np.full_like(rho, a22)]).T
    # filter only those within [0,1]
    mask = (a12 >= 0) & (a12 <= 1) & (a21 >= 0) & (a21 <= 1)
    return alphas[mask], mask

def plot_target_nd_fd(grid_points=50):
    """
    Show strategic sampling: uniform grid of ND and FD targets,
    invert to alphas, then plot those points in ND-FD space colored by feasibility.
    """
    ND_vals = np.linspace(0.0, 1.0, grid_points)
    FD_vals = np.linspace(0.1, 6.0, grid_points)
    ND_grid, FD_grid = np.meshgrid(ND_vals, FD_vals)
    ND_flat = ND_grid.ravel()
    FD_flat = FD_grid.ravel()
    alphas, mask = sample_alphas_from_nd_fd(ND_flat, FD_flat)
    ND_ok = ND_flat[mask]
    FD_ok = FD_flat[mask]
    ND_bad = ND_flat[~mask]
    FD_bad = FD_flat[~mask]

    plt.figure(figsize=(6,6))
    plt.scatter(ND_ok, FD_ok, c='red', s=20, alpha=0.6, label='Coexistence')
    plt.scatter(ND_bad, FD_bad, c='blue', s=20, alpha=0.5, label='Exclusion')
    x = np.linspace(0,1,100)
    plt.plot(x, 1 - x, 'k--', lw=1.5)
    plt.plot(x, 1/(1 - x), 'k--', lw=1.5)
    plt.xlabel('Niche Difference')
    plt.ylabel('Fitness Difference Ratio')
    #plt.title('Strategic ND-FD sampling: feasibility of alphas [0,1]')
    plt.legend(fontsize='large',loc='upper left')
    plt.grid(True, ls='--', alpha=0.5)
    plt.xlim(0,1)
    plt.ylim(0,6)
    plt.tight_layout()
    plt.show()
    
def plot_boolean_coexistence_quiver(
    df_all, cs1, cs2,
    x_trait="Niche Differences",
    y_trait="SC Differences",
    ko_start=-10
):
    """
    Like plot_boolean_coexistence2, but instead of splines uses:
     • Cyan arrows for environment changes (same KO, successive Concentration)
     • Purple arrows for KO flux changes (successive KO, same Concentration)
    """
    # 1) Subset & transform
    df_pair = df_all.query(
        "`Carbon Source 1` == @cs1 & `Carbon Source 2` == @cs2"
    )
    if df_pair.empty:
        raise ValueError(f"No data for {cs1} vs {cs2}")
    df_t = transform_differences(df_pair, niche_col=x_trait, fitness_col=y_trait)
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0

    # 2) Precompute stretch factor
    M = (df_pair[x_trait] / (1 + df_pair[x_trait])).max()

    # 3) Start plotting
    fig, ax = plt.subplots(figsize=(8,6))

    # 4) Boolean points
    for flag, col, label in [(True,'red','Coexist'), (False,'blue','Exclusion')]:
        sel = df_t[df_t['Coex_bool'] == flag]
        ax.scatter(
            sel['ND_norm_stretched'], sel['FD_ratio'],
            c=col, s=80, edgecolor='k', label=label, zorder=3
        )
    # 5) Cyan arrows for environment change (same KO, successive Concentrations)
    ko_list = sorted(df_t['KO Bound Source 1'].unique())
    """

    for ko in ko_list:
        sub = df_t[df_t['KO Bound Source 1'] == ko]
        concs = sorted(sub['Concentration'].unique())
        for i in range(len(concs)-1):
            s = sub[sub['Concentration'] == concs[i]]
            e = sub[sub['Concentration'] == concs[i+1]]
            if len(s)==1 and len(e)==1:
                x0, y0 = s['ND_norm_stretched'].iloc[0], s['FD_ratio'].iloc[0]
                dx = e['ND_norm_stretched'].iloc[0] - x0
                dy = e['FD_ratio'].iloc[0] - y0
                ax.quiver(
                    x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='tail', color='cyan', alpha=0.7, width=0.005,
                    label='Env change' if (ko==ko_list[0] and i==0) else '',
                    zorder=2
                )
                """
    # 6) Purple arrows for KO flux change (consecutive KO, same Concentration)
    for i in range(len(ko_list)-1):
        src, dst = ko_list[i], ko_list[i+1]
        for conc in sorted(df_t['Concentration'].unique()):
            s = df_t.query(
                "`KO Bound Source 1`==@src & Concentration==@conc"
            )
            e = df_t.query(
                "`KO Bound Source 1`==@dst & Concentration==@conc"
            )
            if len(s)==1 and len(e)==1:
                x0, y0 = s['ND_norm_stretched'].iloc[0], s['FD_ratio'].iloc[0]
                dx = e['ND_norm_stretched'].iloc[0] - x0
                dy = e['FD_ratio'].iloc[0] - y0
                ax.quiver(
                    x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    pivot='tail', color='purple', alpha=0.6, width=0.005,
                    label='KO flux change' if i==0 and conc==sorted(df_t['Concentration'])[0] else '',
                    zorder=2
                )

    # 7) Chesson boundaries
    x1 = np.linspace(0,0.999,2000)
    ax.plot(x1, 1 - x1, 'g--', lw=2, label='1-ND < FD < 1/(1-ND)', zorder=1)
    ax.plot(x1, 1/(1 - x1), 'g--', lw=2, zorder=1)

    # 8) Transformed FD=±ND boundaries
    t = np.linspace(0, df_pair[x_trait].max(), 500)
    df_up = pd.DataFrame({x_trait:t, y_trait:t})
    df_lo = pd.DataFrame({x_trait:t, y_trait:-t})
    up_t = transform_differences(df_up, niche_col=x_trait, fitness_col=y_trait, base=10)
    lo_t = transform_differences(df_lo, niche_col=x_trait, fitness_col=y_trait, base=10)
    ax.plot(up_t['ND_norm_stretched'], up_t['FD_ratio'], 'k--', lw=1.5, label='FD=ND (transf)', zorder=1)
    ax.plot(lo_t['ND_norm_stretched'], lo_t['FD_ratio'], 'k--', lw=1.5, zorder=1)

    # 9) Final touches
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel='ND_norm_stretched',
        ylabel='FD_ratio',
        title=f"Quiver-only coexistence: {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}"
    )
    ax.legend(loc='upper left', fontsize='medium')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.ylim((-0.1, 2))
    plt.tight_layout()
    plt.show()
    
def plot_quiver_subset(
    df_all, cs1, cs2, selection,
    x_trait="Niche Differences",
    y_trait="SC Differences"
):
    """
    Plot only the points you list in `selection` (and arrows linking them),
    labeling each as KO1=…, KO2=…, c1=…, c2=0.55−c1.
    """
    # 1) subset & transform
    df = df_all.query(
        "`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2"
    )
    if df.empty:
        raise ValueError(f"No data for {cs1} vs {cs2}")
    df_t = transform_differences(df, niche_col=x_trait, fitness_col=y_trait)
    df_t['Coex_bool'] = df_t['Coexistence Strength'] > 0

    # 2) lookup coords
    coords = {}
    for ko1, ko2, c1 in selection:
        row = df_t.query(
            "`KO Bound Source 1`==@ko1 & `KO Bound Source 2`==@ko2 & Concentration==@c1"
        )
        if len(row) != 1:
            raise ValueError(f"Selection {(ko1,ko2,c1)} not found or not unique")
        coords[(ko1,ko2,c1)] = (
            row['ND_norm_stretched'].iat[0],
            row['FD_ratio'].iat[0],
            row['Coex_bool'].iat[0]
        )

    # 3) plot setup & boundaries
    fig, ax = plt.subplots(figsize=(8,6))
    x1 = np.linspace(0,0.999,2000)
    ax.plot(x1, 1-x1, 'g--', lw=2, label='1-ND<FD<1/(1-ND)')
    ax.plot(x1, 1/(1-x1), 'g--', lw=2)
    t = np.linspace(0, df[x_trait].max(), 500)
    df_up = pd.DataFrame({x_trait:t, y_trait:t})
    df_lo = pd.DataFrame({x_trait:t, y_trait:-t})
    up_t = transform_differences(df_up, niche_col=x_trait, fitness_col=y_trait, base=10)
    lo_t = transform_differences(df_lo, niche_col=x_trait, fitness_col=y_trait, base=10)
    ax.plot(up_t['ND_norm_stretched'], up_t['FD_ratio'], 'k--', lw=1.5, label='FD=ND (transf)')
    ax.plot(lo_t['ND_norm_stretched'], lo_t['FD_ratio'], 'k--', lw=1.5)

    # 4) scatter & label
    for idx, key in enumerate(selection):
        x, y, coex = coords[key]
        color = 'red' if coex else 'blue'
        ax.scatter(x, y, c=color, s=100, edgecolor='k',
                   label=f"KO={key[0]},{key[1]} @ {key[2]:g}" if idx == 0 else None,
                   zorder=3)
        # annotate just offset to the upper right
        c2 = 0.055 - key[2]
        label = f"KO1={key[0]}, KO2={key[1]}\nc1={key[2]:.4g}, c2={c2:.4g}"
        ax.annotate(
            label, (x, y),
            xytext=(5, 5), textcoords='offset points',
            fontsize='medium', zorder=4
        )

    # 5) arrows between successive selection entries
    for i in range(len(selection)-1):
        a, b = selection[i], selection[i+1]
        x0, y0, _ = coords[a]
        x1, y1, _ = coords[b]
        ax.quiver(
            x0, y0, x1-x0, y1-y0,
            angles='xy', scale_units='xy', scale=1,
            pivot='tail', color='black', alpha=0.8, width=0.005,
            label='link' if i==0 else None, zorder=2
        )

    # 6) final touches
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set(
        xlabel='ND_norm_stretched',
        ylabel='FD_ratio',
        title=f"Subset quiver with labels: {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}"
    )
    #ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.5)
    # adjust y‐limit to fit annotations if needed
    ys = [y for (_,(_,y,_)) in coords.items()]
    #ax.set_ylim(-0.1, max(ys)*1.2)
    ax.set_ylim(-0.2, 6)

    plt.tight_layout()
    plt.show()

def plot_quiver_subset_raw(
    df_all, cs1, cs2, selection,
    x_trait="Niche Differences",
    y_trait="SC Differences"
):
    """
    Like plot_quiver_subset, but plots raw ND vs. raw FD (no transforms).
    `selection` is a list of (ko1, ko2, conc) tuples to plot & link.
    Returns a DataFrame of the (x,y) coords for each selected point.
    """
    # 1) subset to this carbon‐pair
    df = df_all.query(
        "`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2"
    )
    if df.empty:
        raise ValueError(f"No data for {cs1} vs {cs2}")
    df = df.copy()
    df['Coex_bool'] = df['Coexistence Strength'] > 0

    # 2) look up coordinates for each selected point
    coords = {}
    for ko1, ko2, c in selection:
        sub = df.query(
            "`KO Bound Source 1`==@ko1 & `KO Bound Source 2`==@ko2 & Concentration==@c"
        )
        if len(sub) != 1:
            raise ValueError(f"Selection {(ko1,ko2,c)} not found or ambiguous")
        row = sub.iloc[0]
        coords[(ko1,ko2,c)] = (
            row[x_trait],
            row[y_trait],
            bool(row['Coex_bool'])
        )

    # 3) plotting
    fig, ax = plt.subplots(figsize=(8,6))
    # theoretical FD = ±ND in raw space
    xs = np.linspace(
        min(v[0] for v in coords.values()),
        max(v[0] for v in coords.values()),
        200
    )
    ax.plot(xs,  xs, 'g--', lw=2, label='FD = +ND')
    ax.plot(xs, -xs, 'g--', lw=2, label='FD = -ND')

    # scatter & annotate
    for idx, key in enumerate(selection):
        x, y, coex = coords[key]
        col = 'red' if coex else 'blue'
        ax.scatter(x, y, c=col, s=100, edgecolor='k',
                   label=f"KO={key[0]},{key[1]} @ {key[2]:g}" if idx==0 else None,
                   zorder=3)
        c2 = 0.55 - key[2]
        txt = f"KO1={key[0]}, KO2={key[1]}\nc1={key[2]:.3g}, c2={c2:.3g}"
        ax.annotate(txt, (x,y), xytext=(5,5), textcoords='offset points',
                    fontsize='small', zorder=4)

    # link points
    for i in range(len(selection)-1):
        a,b = selection[i], selection[i+1]
        x0,y0,_ = coords[a]; x1,y1,_ = coords[b]
        ax.quiver(x0, y0, x1-x0, y1-y0,
                  angles='xy', scale_units='xy', scale=1,
                  pivot='tail', color='k', alpha=0.8, width=0.005,
                  label='link' if i==0 else None, zorder=2)

    # finishing touches
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set(xlabel=x_trait, ylabel=y_trait,
           title=f"Subset quiver (raw) for {cs1.replace('EX_','')} vs {cs2.replace('EX_','')}")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4) build and return a DataFrame of the coords
    out = []
    for (ko1,ko2,c), (x,y,coex) in coords.items():
        out.append({
            'KO1': ko1,
            'KO2': ko2,
            'Concentration': c,
            x_trait: x,
            y_trait: y,
            'Coexistence?': coex
        })
    return pd.DataFrame(out)

def _lookup_transformed_coords_for_selection(
    df_pair, selection, x_trait="Niche Differences", y_trait="SC Differences"
):
    """
    Internal: for a given carbon-pair slice (df_pair), return a DataFrame with
    transformed coords for the requested (KO1, KO2, Concentration) tuples.

    Returns columns:
      ['KO1','KO2','Concentration','NDp','FDp','Coex_bool']
    where NDp is ND' (stretched) and FDp is FD' (exp of SC).
    """
    # stretch factor M based on the full pair slice (consistent with your shading funcs)
    M = (df_pair[x_trait] / (1.0 + df_pair[x_trait])).max()

    rows = []
    for (ko1, ko2, c1) in selection:
        row = df_pair.query(
            "`KO Bound Source 1`==@ko1 & `KO Bound Source 2`==@ko2 & Concentration==@c1"
        )
        if len(row) != 1:
            raise ValueError(
                f"Selection {(ko1, ko2, c1)} not found or not unique for this carbon pair."
            )
        r = row.iloc[0]
        # transformed coordinates to align with your shading/boundary space
        ndp = (r[x_trait] / (1.0 + r[x_trait])) / M
        fdp = np.exp(r[y_trait])
        coex_bool = bool(r["Coexistence Strength"] > 0)
        rows.append({
            "KO1": ko1, "KO2": ko2, "Concentration": c1,
            "NDp": ndp, "FDp": fdp, "Coex_bool": coex_bool
        })
    return pd.DataFrame(rows)


def plot_shading_with_raw_theory_subset_points(
    df_all, cs1, cs2, selection,
    x_trait="Niche Differences", y_trait="SC Differences",
    min_x=-0.03, ylim=(-0.05, 4.2),
    hide_ko_pairs={(-10, -10), (0, 0)},
    point_size=110
):
    """
    Like your shading plots, but ONLY plots the requested points in `selection`
    and hides sentinel (-10,-10) or (0,0) points by alpha=0.
    No lines/splines/vectors are drawn.

    Parameters
    ----------
    selection : list of tuples
        [(ko1, ko2, c1), ...] with exact values present in the DataFrame.
    hide_ko_pairs : set of tuple
        Any (KO1, KO2) pair in this set will be plotted with alpha=0.
    """
    # 1) subset to this carbon pair
    df_pair = df_all.query("`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2")
    if df_pair.empty:
        raise ValueError(f"No data for {cs1} vs {cs2}")

    # 2) transformed coords for the selected points
    pts = _lookup_transformed_coords_for_selection(
        df_pair, selection, x_trait=x_trait, y_trait=y_trait
    )

    # 3) Build transformed ±ND boundaries (for shading)
    t = np.linspace(0, df_pair[x_trait].max(), 2000)
    # use your existing transform function to keep the same shape/styling
    df_up   = pd.DataFrame({ x_trait: t, y_trait:  t })
    df_lo   = pd.DataFrame({ x_trait: t, y_trait: -t })
    df_up_t = transform_differences(df_up, niche_col=x_trait, fitness_col=y_trait, base=10)
    df_lo_t = transform_differences(df_lo, niche_col=x_trait, fitness_col=y_trait, base=10)

    x_tr    = df_up_t['ND_norm_stretched'].values
    y_upper = df_up_t['FD_ratio'].values
    y_lower = df_lo_t['FD_ratio'].values

    # 4) Raw-theory curves (as in your plot_shading_with_raw_theory*)
    x_raw = np.linspace(min_x, 0.999, 2000)
    y_th1 = 1.0 - x_raw
    y_th2 = 1.0 / (1.0 - x_raw)

    # 5) Plot
    fig, ax = plt.subplots(figsize=(6.3, 5))

    # shaded coexistence band in transformed ±ND
    ax.fill_between(x_tr, y_lower, y_upper, facecolor='r', alpha=0.20)
    ax.fill_between(x_tr, y_upper, ylim[1], facecolor='b', alpha=0.20)
    ax.fill_between(x_tr, ylim[0], y_lower, facecolor='b', alpha=0.20)

    # optional: shade the x<0 region lightly (keeps your look from "andPoints")
    yy = np.linspace(ylim[0], ylim[1], 1000)
    ax.fill_betweenx(yy, min_x, 0, facecolor='blue', alpha=0.20, linewidth=0, edgecolor='none', zorder=0)

    # raw-theoretical curves (black dashed), untransformed in y (matches your original)
    ax.plot(x_raw, y_th1, 'k--', lw=2, label="1 - ND' < FD' < 1/(1 - ND')")
    ax.plot(x_raw, y_th2, 'k--', lw=2)

    # transformed boundary lines (solid black)
    ax.plot(x_tr, y_upper, 'k-', lw=2, label="Transformed boundary")
    ax.plot(x_tr, y_lower, 'k-', lw=2)

    # 6) Scatter ONLY the selected points
    for _, r in pts.iterrows():
        color = 'red' if r['Coex_bool'] else 'blue'
        alpha = 0.0 if ( (r['KO1'], r['KO2']) in hide_ko_pairs ) else 1.0
        ax.scatter(
            r['NDp'], r['FDp'], s=point_size, c=color, edgecolor='k',
            alpha=alpha, zorder=3
        )

    # 7) Final styling
    ax.set_xlabel("Transformed Niche Difference (ND')", fontsize=14)
    ax.set_ylabel("Transformed Fitness Difference (FD')", fontsize=14)
    ax.set_xlim((min_x, 1.0))
    ax.set_ylim(ylim)
    ax.legend(loc='upper left', fontsize=12.5)
    plt.tight_layout()
    #plt.savefig('twopoints_transformedphaseplane.pdf')
    plt.show()


def plot_points_subset_transformed_only(
    df_all, cs1, cs2, selection,
    x_trait="Niche Differences", y_trait="SC Differences",
    xlim=(-0.03, 1.0), ylim=(-0.05, 4.2),
    hide_ko_pairs={(-10, -10), (0, 0)},
    point_size=110
):
    """
    Minimalist: plot ONLY the requested points in transformed ND′–FD′ space.
    No shading, no boundaries, no lines/vectors.
    """
    df_pair = df_all.query("`Carbon Source 1`==@cs1 & `Carbon Source 2`==@cs2")
    if df_pair.empty:
        raise ValueError(f"No data for {cs1} vs {cs2}")

    pts = _lookup_transformed_coords_for_selection(
        df_pair, selection, x_trait=x_trait, y_trait=y_trait
    )

    fig, ax = plt.subplots(figsize=(6.3, 5))
    for _, r in pts.iterrows():
        color = 'red' if r['Coex_bool'] else 'blue'
        alpha = 0.0 if ( (r['KO1'], r['KO2']) in hide_ko_pairs ) else 1.0
        ax.scatter(
            r['NDp'], r['FDp'], s=point_size, c=color, edgecolor='k',
            alpha=alpha, zorder=3
        )

    ax.set_xlabel("Transformed Niche Difference (ND')", fontsize=14)
    ax.set_ylabel("Transformed Fitness Difference (FD')", fontsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()



def main():
    output_dir = 'coex_data_mut1_vs_mut2_refactored_test'
    #output_dir = 'fitted_exp_data_june23th_highinit1e4'
    df_all = load_and_merge(output_dir)
    cs1, cs2 = "EX_glc__D_e", "EX_cit_e"

    #plot_ko_vector_fields(df_all, cs1, cs2)
    #plot_boolean_coexistence3(df_all, cs1, cs2)
    #subset = [(-10,-10,0.0275), (-5.625,-6.25,0.0275),  (-3,-3,0.0275), (-3,-3,0.005)]
    #subset = [(-10, -10, 0.0275), (-5.0, -7, 0.0275), (-2.78, -2.5, 0.0275), (-2.78, -2.5, 0.005),(0, 0, 0.0275)]
    #outfit = plot_quiver_subset_raw(df_all, "EX_glc__D_e", "EX_succ_e", subset)
    #outfit.to_csv('fit_NDFDw0.csv')
    #plot_boolean_coexistence3(df_all, cs1, cs2)
    plot_shading_with_raw_theory(df_all, cs1, cs2)
    plot_shading_with_raw_theory_andPoints(df_all, cs1, cs2)
    plot_shading_with_raw_theory_subset_points(df_all, cs1, cs2, [(-10,-10,0.0275),(-6,-6,0.0275), (-1,-1,0.0275), (0,0,0.0275)])
    #plot_boolean_coexistence2_subset(df_all, cs1, cs2, ko_focus=-3)
    #plot_lv_nd_fd(n_samples=500, low=0.1, high=2.0, seed=123)
    #plot_lv_nd_fd_offdiag(a11=0.5, a22=0.5, n_samples=1000, seed=42)
    #plot_target_nd_fd(grid_points=50)
    #plot_coex_prob(df_all, cs1, cs2)

if __name__ == "__main__":
    main()
