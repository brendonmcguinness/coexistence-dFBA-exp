#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:31:46 2025

@author: brendonmcguinness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# === 1. Load your data ===
# 1) Load your main merged table

dir_name = 'coex_data_mut1_vs_mut2_refactored_test' #coex_data_mut1_vs_mut2_withR_allCarb
#dir_name = 'coex_data_mut1_vs_mut2_withR_allCarb'
df_niche = pd.read_csv(f"{dir_name}/niche_differences.csv")

#df_mgg  = pd.read_csv(f"{dir_name}/maxgrowthgraddiff.csv") #"mgg_diff_results.csv")
#df_mgl  = pd.read_csv(f"{dir_name}/maxgrowthlogdiff.csv") #"mgl_diff_results.csv")
df_coex = pd.read_csv(f"{dir_name}/coexistence_results.csv")
#df_win  = pd.read_csv(f"{dir_name}/winners.csv") #"winner_results.csv")
#df_sc = pd.read_csv(f"{dir_name}/scdiff.csv") #"sc_diff_results.csv")

#for different environments
df_win  = pd.read_csv(f"{dir_name}/winner_results.csv") #"winner_results.csv")
df_sc = pd.read_csv(f"{dir_name}/sc_diff_results.csv") #"sc_diff_results.csv")
df_mgg  = pd.read_csv(f"{dir_name}/mgg_diff_results.csv") #"mgg_diff_results.csv")
df_mgl  = pd.read_csv(f"{dir_name}/mgl_diff_results.csv") #"mgl_diff_results.csv")
#df_niche["Value"] = pd.to_numeric(df_niche["Value"], errors="coerce").abs()

# rename Value columns
#df_niche = df_niche.rename(columns={"Value":"Niche_Diff"})
df_mgg = df_mgg.rename(columns={"Value":"MGG_Diff"})
df_mgl = df_mgl.rename(columns={"Value":"MGL_Diff"})
df_coex= df_coex.rename(columns={"Value":"Coexistence_Strength"})
df_win = df_win.rename(columns={"Value":"Winner"})
df_sc = df_sc.rename(columns={"Value":"SC_Diff"})

# Merge all of them on the four keys
keys = ["Carbon Source 1","Carbon Source 2","Concentration","KO Bound Source 1","KO Bound Source 2", "KO Strain Index"]

df = (
    df_niche[ keys + ["Value"]].rename(columns={"Value":"Niche_Diff"})
    .merge(df_mgg[ keys + ["MGG_Diff"] ], on=keys)
    .merge(df_mgl[ keys + ["MGL_Diff"] ], on=keys)
    .merge(df_coex[keys + ["Coexistence_Strength"]], on=keys)
    .merge(df_win [keys + ["Winner"]], on=keys)
    .merge(df_sc [keys + ["SC_Diff"]], on=keys)
)
df['Niche_Diff'] = df['Niche_Diff'].abs()

"""
dfs_indexed = []
for df, valcol in zip([df_niche,df_mgg,df_mgl,df_coex,df_win,df_sc],["Niche_Diff","MGG_Diff","MGL_Diff","Coexistence_Strength","Winner","SC_Diff"]):
    df2 = df.drop_duplicates(subset=keys).set_index(keys)[[valcol]]
    dfs_indexed.append(df2)
    print(df2)
#df_all = dfs_indexed[0].join(dfs_indexed[1:], how="inner").reset_index()
df_all = pd.concat(dfs_indexed,axis=1,join="inner").reset_index()
print(f"Joined rows: {len(df_all)}")
 """   

# === 2. Compute the boolean label ===
df['Coex_bool'] = df['Coexistence_Strength'] > 0.0
df['SC_Diff_abs'] = df['SC_Diff'] #.abs()
df['SC_MGL_abs'] = df['MGL_Diff'].abs()


# === 3. Prepare features and labels ===
X = df[['Niche_Diff', 'SC_Diff_abs']].values
#X = df[['Niche_Diff', 'SC_MGL_abs']].values
y = df['Coex_bool'].astype(int).values

# === 4. Fit a classifier ===
# Option A: Unregularized logistic regression (scikit-learn ≥0.22)
clf = LogisticRegression(
    penalty='l2',
    C=1e6,
    solver='lbfgs',
    tol=1e-4,
    max_iter=2000
).fit(X, y)

# === 5. Compute analytic decision boundary ===
w0, w1 = clf.coef_[0]             # weights for Niche_Diff (x) and SC_Diff_abs (y)
b      = clf.intercept_[0]       # bias term
x_min, x_max = df['Niche_Diff'].min(), df['Niche_Diff'].max()
x_vals = np.array([x_min, x_max])
y_vals = -(w0*x_vals + b) / w1    # solve w0*x + w1*y + b = 0 for y


m=-w0 / w1
c = -b / w1
print(f"Decision boundary: y = {m:.3f} * x + {c:.3f}")
equation_text = f"$y = {m:.3f} x + {c:.3f}$"#

boundary1 = np.linspace(0,df['Niche_Diff'].max(),100)

plt.figure(figsize=(8, 6))
plt.plot(boundary1,boundary1,'k--',lw=2, label='Decision boundary')
plt.plot(boundary1,-boundary1,'k--',lw=2)

# plot the analytic boundary
#plt.plot(x_vals, y_vals, 'k--', lw=2, label='Decision boundary')
#plt.text(-0.05, 3.5, equation_text)
# then your scatter
sc = plt.scatter(
    df['Niche_Diff'], df['SC_Diff_abs'],
    c=df['Coex_bool'], cmap='bwr',
    alpha=0.8, s=60, edgecolor='k'
)
plt.xlabel('Niche Differences')
plt.ylabel('Fitness Differences (SC diff)')
plt.legend()
plt.show()

# 1) make an x‐grid from 0 to your max niche difference
x_min, x_max = 0.0, df['Niche_Diff'].max()+0.01
x_vals = np.linspace(x_min, x_max, 500)

# 2) theoretical bounds
y_upper = + x_vals       # y =  x
y_lower = - x_vals       # y = -x

# 3) plot limits (pad a bit)
y_min = -5 #df['SC_Diff_abs'].min() - 1.0
y_max = 5 #df['SC_Diff_abs'].max() + 1.0

fig, ax = plt.subplots(figsize=(6,5))

# 4) fill the wedge red (coexistence)
ax.fill_between(
    x_vals,
    y_lower,        # bottom of wedge
    y_upper,        # top of wedge
    facecolor='r', alpha=0.2,
    label='Coexistence'
)

# 5) fill above the wedge blue
ax.fill_between(
    x_vals,
    y_upper,        # from the top of the wedge…
    y_max,          # …up to the plot top
    facecolor='b', alpha=0.2,
    label='Exclusion'
)

# 6) fill below the wedge blue
ax.fill_between(
    x_vals,
    y_min,          # from the plot bottom…
    y_lower,        # …up to the bottom of the wedge
    facecolor='b', alpha=0.2,
    label='_nolegend_'   # hides this extra patch in the legend
)

# 7) draw your logistic‐regression decision boundary on top
y_boundary = m * x_vals + c
ax.plot(x_vals, y_upper, 'k-', lw=2, label='FD=ND')
ax.plot(x_vals, y_lower, 'k-', lw=2)

# 8) scatter your points
ax.scatter(
    df['Niche_Diff'], df['SC_Diff_abs'],
    c=df['Coex_bool'], cmap='bwr',
    edgecolor='k', s=60, alpha=0.0
)

# 9) polish
ax.set_xlim(x_min, x_max)
ax.set_ylim(-5,5)
ax.set_xlabel('Raw Niche Difference (ND)',fontsize=14)
ax.set_ylabel('Raw Fitness Difference (FD)',fontsize=14)
ax.legend(loc='upper left',fontsize=14)
plt.tight_layout()
#plt.savefig('untransformed_fig2_nopoints_boundary_FDNDlabel.pdf')
plt.show()
