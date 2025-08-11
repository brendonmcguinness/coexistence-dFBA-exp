#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 23:25:40 2025

@author: brendonmcguinness
"""
import numpy as np
import matplotlib.pyplot as plt

def _shade_full(ax, nd_line, fd_low, fd_high, fd_min, fd_max):
    # coexistence band in lightcoral
    ax.fill_between(nd_line, fd_low, fd_high,
                    where=fd_high>=fd_low,
                    color='r', alpha=0.2,label='Coexistence')
    # outside band (below low, above high) in lightblue
    ax.fill_between(nd_line, fd_min, fd_low,
                    where=fd_low>=fd_min,
                    color='b', alpha=0.2,label='Exclusion')
    ax.fill_between(nd_line, fd_high, fd_max,
                    where=fd_high<=fd_max,
                    color='b', alpha=0.2)
    # boundaries
    ax.plot(nd_line, fd_low,  'k--', lw=1.5)
    ax.plot(nd_line, fd_high,'k--', lw=1.5)
    ax.legend()


def plot_cross_quiver(
    grid_size=10,
    nd_range=(0.0, 1.0),
    fd_range=(0.0, 7.0),
    arrow_length=0.04
):
    # 1) Regular grid in ND–FD space
    ND_vals = np.linspace(*nd_range, grid_size)
    FD_vals = np.linspace(*fd_range, grid_size)
    ND, FD = np.meshgrid(ND_vals, FD_vals)

    # 2) Coexistence boundary curves
    nd_line = np.linspace(0, 1, 400)
    fd_low  = 1.0 - nd_line
    fd_high = 1.0 / (1.0 - nd_line)
    # cap upper curve at fd_range[1]
    fd_high = np.minimum(fd_high, fd_range[1])

    # 3) Begin plotting
    fig, ax = plt.subplots(figsize=(4,4))

    # 3a) Shade full plane with our scheme
    _shade_full(ax, nd_line, fd_low, fd_high, fd_range[0], fd_range[1])

    # 4) Build constant horizontal & vertical arrow fields
    U_h = np.ones_like(ND) * arrow_length   # horizontal component
    V_h = np.zeros_like(ND)                # no vertical component

    U_v = np.zeros_like(ND)                # no horizontal component
    V_v = np.ones_like(ND) * arrow_length  # vertical component

    # 5) Plot double-sided, light-gray horizontal arrows
    """
    ax.quiver(ND, FD,  U_h,  V_h,
              color='gray', pivot='mid', scale=0.5, width=0.008,alpha=0.2)
    ax.quiver(ND, FD, -U_h, -V_h,
              color='gray', pivot='mid', scale=0.5, width=0.008,alpha=0.2)

    # 6) Plot double-sided, light-gray vertical arrows
    ax.quiver(ND, FD,  U_v,  V_v,
              color='gray', pivot='mid', scale=0.5, width=0.008,alpha=0.2)
    ax.quiver(ND, FD, -U_v, -V_v,
              color='gray', pivot='mid', scale=0.5, width=0.008,alpha=0.2)
    """
    # 7) Final touches
    ax.set_xlim(*nd_range)
    ax.set_ylim(*fd_range)

    ax.set_xlabel('Niche Difference')
    ax.set_ylabel('Fitness Difference')
    ax.set_ylim((0,6))
    #ax.set_title(f'Interdependent Δ(ND,FD) with weights={weights}')
    #ax.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    #plt.savefig('independentLVfig1_NOQUIV.pdf')
    plt.show()




if __name__ == "__main__":
    plot_cross_quiver(grid_size=10, arrow_length=0.1)
