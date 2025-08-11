
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep over initial_mean_pop and total carbon concentration;
compute ND, FD, and average time‐to‐zero‐growth for each combo.
"""
import cometspy as c
from cobra.io import load_model
import numpy as np
import matplotlib.pyplot as plt
from helpers import returnSCMatrixWithBiomass

def iso_line_for_t(t_target, t_mat, xs, ys):
    """
    Return y(x) such that t_mat(y, x) = t_target by linear interpolation
    down each column j (varying y over rows i).
    xs: 1D array for x (initial_mean_pops) length nP
    ys: 1D array for y (total_concs)      length nT
    """
    y_iso = np.full_like(xs, np.nan, dtype=float)
    for j in range(len(xs)):
        col = t_mat[:, j] - t_target
        seg = np.where(np.diff(np.sign(col)) != 0)[0]
        if seg.size:
            i = seg[0]
            y1, y2 = ys[i], ys[i+1]
            t1, t2 = t_mat[i, j], t_mat[i+1, j]
            y_iso[j] = y1 + (t_target - t1) * (y2 - y1) / (t2 - t1)
    return y_iso

def extend_yiso_to_edges(x_lin, y_lin, y_domain, do_log=True):
    """
    Extend an iso-line y(x) (possibly with NaNs) to the full x-range by
    interpolating gaps and extrapolating edges. If do_log=True, do it in log10 space.
    y_domain = (ymin, ymax) in linear units for clipping.
    """
    x = np.asarray(x_lin, float)
    y = np.asarray(y_lin, float)
    ymin, ymax = y_domain

    # mask of finite points
    m = np.isfinite(y)
    if m.sum() < 2:
        # not enough points to build a line; return original
        return y.copy()

    if do_log:
        xw = np.log10(x)
        yw = np.log10(np.clip(y, ymin, None))  # ensure positive for log
        # For points where y is NaN, we just keep NaN for now
        yw[~m] = np.nan
    else:
        xw = x.copy()
        yw = y.copy()

    # --- Fill interior NaNs by interpolation over finite points
    # indices of finite points
    idx = np.flatnonzero(np.isfinite(yw))
    x_f = xw[idx]
    y_f = yw[idx]

    # Build a full y* initialized with NaNs
    yw_full = np.full_like(xw, np.nan, dtype=float)

    # Interpolate over the convex hull (between first and last finite x)
    yw_full = np.interp(xw, x_f, y_f, left=np.nan, right=np.nan)

    # --- Extrapolate to the left using the first two finite points
    if idx.size >= 2:
        i0, i1 = idx[0], idx[1]
        slope_left = (yw[i1] - yw[i0]) / (xw[i1] - xw[i0])
        left_mask = xw < xw[i0]
        yw_full[left_mask] = yw[i0] + slope_left * (xw[left_mask] - xw[i0])

        # Extrapolate to the right using the last two finite points
        j0, j1 = idx[-2], idx[-1]
        slope_right = (yw[j1] - yw[j0]) / (xw[j1] - xw[j0])
        right_mask = xw > xw[j1]
        yw_full[right_mask] = yw[j1] + slope_right * (xw[right_mask] - xw[j1])

    # Convert back from log if needed and clip to domain
    if do_log:
        y_ext = np.power(10.0, yw_full)
    else:
        y_ext = yw_full

    # Clip to the heatmap's y-domain (prevents overshoot past the data range)
    y_ext = np.clip(y_ext, ymin, ymax)
    return y_ext
# ─── Configuration ─────────────────────────────────────────────────────────────
exchange_rxns = ["EX_glc__D_e", "EX_cit_e"]
M = 2

# load & KO models
mut1 = c.model(load_model("iJO1366"))
mut2 = c.model(load_model("iJO1366"))
for rxn in exchange_rxns:
    mut1.change_bounds(rxn, -10, 1000)
    mut2.change_bounds(rxn, -10, 1000)
ko1 = ko2 = -3
source1, source2 = exchange_rxns
mut1.change_bounds(source1, ko1, 1000)
mut2.change_bounds(source2, ko2, 1000)
mut1.id = f"{source1}_KO_{ko1}"
mut2.id = f"{source2}_KO_{ko2}"

# ─── Sweep parameters ──────────────────────────────────────────────────────────
N = 5
initial_mean_pops = np.logspace(-7, -2, N)     # 5 points from 1e-7 → 1e-2
total_concs       = np.logspace(-4,-1,N)  # total carbon to test

# prepare storage: shape (len(total_concs), len(initial_mean_pops))
nT = len(total_concs)
nP = len(initial_mean_pops)

ND_mat      = np.zeros((nT, nP))
FD_mat      = np.zeros((nT, nP))
tzero_mat   = np.zeros((nT, nP))

# ─── Double‐loop sweep ─────────────────────────────────────────────────────────
for iT, TotC in enumerate(total_concs):
    # set equal halves
    s1_conc = np.array([TotC/2])
    s2_conc = np.array([TotC/2])

    for iP, imp in enumerate(initial_mean_pops):
        # run CRM→LV + time‐series
        s_wt, s_mut, wt_freq, media_list, biomass_list = returnSCMatrixWithBiomass(
            mut1, mut2,
            source1, source2,
            s1_conc, s2_conc,
            N=M,
            init_ratio_diff=1e-3,
            init_mean_pop=imp
        )

        # collect per‐rep & per‐strain max‐growth & first-zero times
        t_zeros = []
        for bio in biomass_list:
            t = bio["t"]
            for model in (mut1, mut2):
                dy = np.gradient(bio[model.id], t)
                # first zero crossing
                zeros = np.where(dy == 0)[0]
                if zeros.size:
                    t_zeros.append(t[zeros[0]])
        tzero_mat[iT, iP] = np.nanmean(t_zeros)

        # compute ND & FD
        NO = (s_wt[0] - s_wt[-1]) / (wt_freq[0] - wt_freq[-1])
        ND_mat[iT, iP] = abs(NO)
        FD_mat[iT, iP] = abs(s_wt[0] - s_mut[1])

# ─── Example plotting ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,4))
for iT, TotC in enumerate(total_concs):
    ax.plot(tzero_mat[iT], ND_mat[iT],  'o-', label=f"ND, total={TotC}")
    ax.plot(tzero_mat[iT], FD_mat[iT],  's--', label=f"FD, total={TotC}")
ax.set_xlabel("avg time-to-zero-growth (h)")
ax.set_ylabel("Difference")
ax.set_title("ND & FD vs. avg t_zero\nfor different total C")
ax.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# ─── Example plotting ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,4))
for iT, TotP in enumerate(initial_mean_pops):
    ax.plot(tzero_mat[:,iT], ND_mat[:,iT],  'o-', label=f"ND, total={TotP:2f}")
    ax.plot(tzero_mat[:,iT], FD_mat[:,iT],  's--', label=f"FD, total={TotP:2f}")
ax.set_xlabel("Mean time-to-zero-growth (h)")
ax.set_ylabel("Difference")
ax.set_title("ND & FD vs. avg t_zero\nfor different total P")
ax.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# flatten your matrices
t_flat  = tzero_mat.flatten()
ND_flat = ND_mat.flatten()
FD_flat = FD_mat.flatten()

# for plotting ND vs FD, draw diagonal ND=FD line
max_ndfd = max(ND_flat.max(), FD_flat.max())

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))

# ── Panel 1: ND vs FD, colored by t_zero ─────────────────────────────
sc = ax1.scatter(ND_flat, FD_flat, c=t_flat, cmap='plasma', s=60, edgecolor='k')
ax1.plot([0, max_ndfd], [0, max_ndfd], 'k--', label='ND = FD')
ax1.set_xlabel("Niche diff (ND)")
ax1.set_ylabel("Fitness diff (FD)")
ax1.set_title("ND vs FD (color → t_zero)")
ax1.legend()
cbar = fig.colorbar(sc, ax=ax1)
cbar.set_label("avg time-to-zero-growth (h)")

# ── Panel 2: Δ = ND – FD vs t_zero ────────────────────────────────
delta = ND_flat - FD_flat
sc2 = ax2.scatter(t_flat, delta, c=t_flat, cmap='plasma', s=60, edgecolor='k')
ax2.axhline(0, color='k', linestyle='--', label='ND = FD (Δ=0)')
ax2.set_xlabel("avg time-to-zero-growth (h)")
ax2.set_ylabel("Δ = ND – FD")
ax2.set_title("Δ vs t_zero\n(sign above 0 → coexistence)")
ax2.legend()
cbar2 = fig.colorbar(sc2, ax=ax2)
cbar2.set_label("avg time-to-zero-growth (h)")

plt.suptitle("Time-to-zero as the bottleneck linking inputs → coexistence/exclusion", y=1.03)
plt.tight_layout()
plt.show()

# assume total_concs (len nT) and initial_mean_pops (len nP) defined,
# and tzero_mat is shape (nT, nP)
#no log xis on
fig, ax = plt.subplots(figsize=(6,5))
# imshow with origin='lower' so lowest TotC at bottom
im = ax.imshow(tzero_mat,
               aspect='auto',
               origin='lower',
               cmap='viridis',
               extent=[
                   np.log10(initial_mean_pops[0]), 
                   np.log10(initial_mean_pops[-1]),
                   np.log10(total_concs[0]),
                   np.log10(total_concs[-1])
               ])


# X ticks back in real space
xticks = np.log10(initial_mean_pops)
ax.set_xticks(xticks)
yticks = np.log10(total_concs)
ax.set_yticks(yticks)
ax.set_xticklabels([f"{x:.0e}" for x in initial_mean_pops])
ax.set_yticklabels([f"{y:.0e}" for y in total_concs])

ax.set_xlabel("Initial mean biomass (log scale)")
ax.set_ylabel("Total carbon concentration (log scale)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean time to zero‐growth (h)")
#ax.set_title("Heatmap of t_zero vs init_pop & total_C")
plt.tight_layout()
plt.show()

# Option A: sorted line
idx = np.argsort(t_flat)
t_s  = t_flat[idx]
ND_s = ND_flat[idx]
FD_s = FD_flat[idx]

ND_ts = (ND_s / (1.0 + ND_s)) / 0.8
#ND_ts = np.log(ND_s)
FD_ts = np.exp(FD_s)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(t_s, ND_s,  'o-', label='ND')
ax.plot(t_s, FD_s,  's--', label='FD')
#ax.axhline(0, color='k', linestyle=':')
ax.set_xlabel("avg time-to-zero-growth (h)")
ax.set_ylabel("Difference")
ax.set_title("ND & FD vs. avg t_zero (all sweeps)")
ax.legend()
plt.tight_layout()
plt.show()

# (after you’ve built and sorted your flat arrays)

# 1) compute Δ = ND – FD on the sorted data
delta = ND_s - FD_s
delta = 1/(1-ND_ts) - FD_ts
# 2) find the first sign‐change in Δ
ix = np.where(np.diff(np.sign(delta)) != 0)[0]
if ix.size:
    i = ix[0]
    # linear interpolation between (t_s[i], Δ[i]) and (t_s[i+1], Δ[i+1])
    t1, t2 = t_s[i],   t_s[i+1]
    d1, d2 = delta[i], delta[i+1]
    t_star = t1 - d1 * (t2 - t1)/(d2 - d1)
else:
    raise RuntimeError("No intersection found!")
    
# build the iso‐line by interpolating down each column of tzero_mat
x = initial_mean_pops
y = total_concs
y_iso = np.empty_like(x)
y_iso[:] = np.nan


for j in range(len(x)):
    col = tzero_mat[:, j] - t_star
    # find the first sign‐change in this column
    seg = np.where(np.diff(np.sign(col)) != 0)[0]
    if seg.size:
        i = seg[0]
        # linearly interpolate between (y[i], t_i) and (y[i+1], t_{i+1})
        y1, y2     = y[i],   y[i+1]
        t1, t2     = tzero_mat[i, j],   tzero_mat[i+1, j]
        y_iso[j] = y1 + (t_star - t1)/(t2 - t1)*(y2 - y1)
    # else leave as NaN (no crossing in that column)

# now overplot it on your heatmap axis:
fig, ax = plt.subplots(figsize=(6,4))
pcm = ax.pcolormesh(
    initial_mean_pops, total_concs, tzero_mat,
    shading='auto', cmap='viridis'
)
plt.colorbar(pcm, ax=ax, label="avg time to zero‐growth (h)")

# continuous red line:
ax.plot(
    x, y_iso,
    color='red',
    linewidth=8,
    solid_capstyle='round',
    zorder=10
)

#ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel("initial_mean_pop")
ax.set_ylabel("total carbon conc")
#ax.set_title("Heatmap of t_zero\n(red line = critical $t^*$)")
plt.tight_layout()
plt.show()

ND_clean_ts = np.delete(ND_ts,[6,9])
FD_clean_ts = np.delete(FD_ts,[6,9])
t_clean_ts = np.delete(t_s,[6,9])
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(t_clean_ts, 1-ND_clean_ts, '--', color='k', label="1-ND'", lw=2)
ax.plot(t_clean_ts, FD_clean_ts,  '-', color='g', label="FD'", lw=2)
ax.plot(t_clean_ts, 1/(1-ND_clean_ts), '--', color='k', label="1/(1-ND')", lw=2)
#ax.axhline(0, color='k', linestyle=':')
ax.set_xlabel("Mean time to zero growth (h)")
ax.set_ylabel("Transformed difference")
#ax.set_title("ND & FD vs. avg t_zero (all sweeps)")
ax.legend()
plt.tight_layout()
#plt.savefig('t2nogrowth_FD_ND.pdf')
plt.show()

# === Shaded band between (1 - ND') and 1/(1 - ND'), red elsewhere; mark crossing with upper curve ===
NDp = ND_clean_ts         # ND'
FDp = FD_clean_ts         # FD'
t   = t_clean_ts          # time (sorted)

# Curves: lower = 1 - ND', upper = 1/(1 - ND')
eps = 1e-6
lower = 1.0 - NDp
upper = 1.0 / np.clip(1.0 - NDp, eps, None)  # avoid divide-by-zero

# A robust y-limits estimate (pad a bit so fills look clean)
y_all = np.concatenate([FDp[np.isfinite(FDp)],
                        lower[np.isfinite(lower)],
                        upper[np.isfinite(upper)]])
ylo = np.nanmin(y_all); yhi = np.nanmax(y_all)
pad = 0.05 * (yhi - ylo if np.isfinite(yhi - ylo) and (yhi - ylo) > 0 else 1.0)
ylo -= pad; yhi += pad

def find_crossing(x, y1, y2):
    """Return x* where y1(x*) == y2(x*) via first sign change and linear interpolation."""
    d = y1 - y2
    finite = np.isfinite(d)
    x_f = x[finite]; d_f = d[finite]
    if x_f.size < 2:
        return None
    idx = np.where(np.diff(np.sign(d_f)) != 0)[0]
    if idx.size == 0:
        return None
    k = idx[0]
    x1, x2 = x_f[k],   x_f[k+1]
    d1, d2 = d_f[k],   d_f[k+1]
    return x1 - d1 * (x2 - x1) / (d2 - d1)

# Crossing with the UPPER curve (FD' = 1/(1 - ND'))
t_int_upper = find_crossing(t, FDp, upper)
FD_at_upper = np.interp(t_int_upper, t, FDp) if t_int_upper is not None else None

fig, ax = plt.subplots(figsize=(4,3))

# Plot curves first
ax.plot(t, lower, '--', label="1 - ND' & 1/(1 - ND')", lw=2,color='k')
ax.plot(t, FDp,   '-',  label="FD'",      lw=2, color='g')
ax.plot(t, upper, '--', lw=2,color='k')

# Fix y-limits before fills so "outside" fills are well-defined
ax.set_ylim(ylo, yhi)

# Masks for valid fills
mask_band = np.isfinite(lower) & np.isfinite(upper)
mask_low  = np.isfinite(lower)
mask_up   = np.isfinite(upper)

# 1) Blue band between lower and upper
ax.fill_between(t, lower, upper, where=mask_band, alpha=0.2, color='r',label='Coexistence')

# 2) Red below the band
ax.fill_between(t, ylo*np.ones_like(t), lower, where=mask_low, alpha=0.2, color='b', label="Exclusion")

# 3) Red above the band
ax.fill_between(t, upper, yhi*np.ones_like(t), where=mask_up, alpha=0.2, color='b')

# Mark the intersection with the UPPER curve and a vertical guide
if t_int_upper is not None and np.isfinite(FD_at_upper):
    ax.scatter([t_int_upper], [FD_at_upper], zorder=5, edgecolor='k', facecolor='w')
    ax.axvline(t_int_upper, color='r', linestyle='-', linewidth=1.5,label='Time at intersection')
valid = mask_band & np.isfinite(lower) & np.isfinite(upper)
if valid.any():
    ax.set_xlim(t[valid][0], t[valid][-1])   # tight to the filled region
ax.set_xlabel("Mean time to zero growth (h)")
ax.set_ylabel("Transformed difference")
ax.legend(fontsize='small')
plt.tight_layout()
plt.savefig('time_critical_time.pdf')
plt.show()


# ─── Heatmap (imshow) with iso-line at the intersection time ─────────
#tzero_mat = tzero_mat[1:,1:]
#initial_mean_pops = initial_mean_pops[1:]
#total_concs = total_concs[1:]
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(
    tzero_mat,
    aspect='auto',
    origin='lower',
    cmap='viridis',
    extent=[
        np.log10(initial_mean_pops[0]),
        np.log10(initial_mean_pops[-1]),
        np.log10(total_concs[0]),
        np.log10(total_concs[-1])
    ]
)

# ticks & labels back to real-space formatting
#ax.set_xticks(np.log10(initial_mean_pops))
#ax.set_yticks(np.log10(total_concs))
#ax.set_xticklabels([f"{x:.0e}" for x in initial_mean_pops])
#ax.set_yticklabels([f"{y:.0e}" for y in total_concs])

ax.set_xlabel("Log initial mean biomass (gDW)")
ax.set_ylabel("Log total carbon concentration (M)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean time to zero‐growth (h)")

# ← NEW: iso-line for t = t_int (convert to log-space for this axis)
y_iso_int = iso_line_for_t(t_int_upper, tzero_mat, initial_mean_pops, total_concs)
# Build extended curve from your existing iso-line
y_iso_ext = extend_yiso_to_edges(
    initial_mean_pops,           # x in linear units
    y_iso_int,                   # iso-line y (may contain NaNs)
    (total_concs.min(), total_concs.max()),
    do_log=True                  # because imshow uses log10 axes
)
ymax = total_concs.max()
ymin = total_concs.min()
# Replace clipping with masking so the curve disappears at the bounds
y_iso_ext[(y_iso_ext <= ymin) | (y_iso_ext >= ymax)] = np.nan# Plot it in log space on the imshow figure:
ax.plot(np.log10(initial_mean_pops), np.log10(y_iso_ext),
        color='red', linewidth=3, solid_capstyle='round', zorder=10, clip_on=False)
ylo, yhi = ax.get_ylim()
plt.tight_layout()
plt.show()