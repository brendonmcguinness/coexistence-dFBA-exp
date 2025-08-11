#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:22:01 2025

@author: brendonmcguinness
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress


def flatten_post_max_to_constant(df, value_col='biomass', time_col='time', group_cols=['strain', 'carbon']):
    def flat_group(g):
        g = g.sort_values(time_col).copy()
        max_val = g[value_col].max()
        max_time = g[g[value_col] == max_val][time_col].min()
        g.loc[g[time_col] >= max_time, value_col] = max_val
        return g

    return df.groupby(group_cols, group_keys=False).apply(flat_group)


results = []

datasets = {
    #"glucose": "unprocessed_data_glc_32h.csv",
    "succinate": "unprocessed_data_suc_32h.csv"
}
"""
datasets = {
    "glucose": "unprocessed_data.csv",
    "succinate": "unprocessed_data_succ.csv"
}
"""
od_bio_flag = 1
if od_bio_flag:
    od_bio = 'ODb'
else:
    od_bio = 'biomass'
    
for dataset_name, file_path in datasets.items():
    df_raw = pd.read_csv(file_path)

    # Compute mean biomass
    #df_mean = df_raw.groupby(['strain', 'carbon', 'time'], as_index=False)['biomass'].mean()
    df_mean = df_raw.groupby(['strain', 'carbon', 'time'], as_index=False)[od_bio].mean()
    # Flatten biomass after max
    df_flat = flatten_post_max_to_constant(df_mean, value_col=od_bio, group_cols=['strain', 'carbon'])
    df_flat.to_csv(f'processed_data_{dataset_name}_{od_bio}_2.csv', index=False)

    # Estimate growth rates
    for (strain, carbon), group in df_flat.groupby(['strain', 'carbon']):
        group = group.sort_values('time')
        group = group[group[od_bio] > 0].copy()
        group['log_biomass'] = np.log(group[od_bio])

        max_biomass = group[od_bio].max()
        max_time = group[group[od_bio] == max_biomass]['time'].min()
        group_exp = group[group['time'] < max_time]

        if len(group_exp) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(group_exp['time'], group_exp['log_biomass'])
            gradients = np.gradient(group_exp[od_bio], group_exp['time'])
            avg_gradient = np.max(gradients)

            time_arr = group_exp['time'].to_numpy()
            bio_arr = group_exp['log_biomass'].to_numpy()
            best_slope = float('-inf')
            best_grad = float('-inf')
            window = 6

            for i in range(len(time_arr)):
                t0 = time_arr[i]
                mask = (time_arr >= t0) & (time_arr <= t0 + window)
                if np.sum(mask) >= 2:
                    t_win = time_arr[mask]
                    y_win = bio_arr[mask]
                    win_slope, _, win_r_value, _, win_stderr = linregress(t_win, y_win)
                    win_grad = np.gradient(y_win, t_win).mean()
                    if win_slope > best_slope:
                        best_slope = win_slope
                        best_stderr = win_stderr
                        best_r_value = win_r_value
                    if win_grad > best_grad:
                        best_avg_grad = win_grad

            results.append({
                'dataset': dataset_name,
                'strain': strain,
                'carbon': carbon,
                'mu_max': slope,
                'mu_grad': avg_gradient,
                'mu_grad_2h_window': best_avg_grad,
                'mu_2h_window': best_slope,
                'mu_2h_error': best_stderr,
                'mu_2h_r2': best_r_value**2,
                'r_squared': r_value**2,
                'n_points': len(group_exp)
            })

# Combine and save all results
df_mu_all = pd.DataFrame(results)
#df_mu_all.to_csv('mu_max_estimates_combined.csv', index=False)
