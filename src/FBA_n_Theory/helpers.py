#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:45:33 2023

@author: brendonmcguinness
"""
import cometspy as c
#from cobra import test
import cobra
from cobra.io import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob
from scipy.stats import linregress
import networkx as nx

#calc fitness difference in each initial pop add this

#os.environ['GUROBI_HOME'] = 'C:\\gurobi902\\win64'
os.environ['COMETS_HOME'] = '/Applications/COMETS'
os.environ['GUROBI_COMETS_HOME'] = '/Library/gurobi1003/macos_universal2'
os.environ['GRB_LICENSE_FILE']='/Library/gurobi1003/macos_universal2/gurobi.lic' # load the models and perform the mutation

print("Running helpers.py from:", __file__)
def extract_growth_metrics_from_csv(filepath: str, target: str, bound: int):
    try:
        df = pd.read_csv(filepath)
        biomass_col = f"EX_{target}_e_KO_{bound}"
        if biomass_col not in df.columns or "t" not in df.columns:
            return None, None
        biomass = df[biomass_col]
        time = df["t"]
        carrying_capacity = biomass.max()
        growth_rate = np.gradient(biomass, time).max()
        return growth_rate, carrying_capacity
    except:
        return None, None

def analyze_growth_for_min_coexistence_point(df_merged: pd.DataFrame, biomass_dir: str) -> pd.DataFrame:
    records = []
    df_merged["Carbon Pair"] = df_merged["Carbon Source 1"] + " + " + df_merged["Carbon Source 2"]
    #df_merged["Carbon Pair"] = df_merged.apply(
    #    lambda row: " + ".join(sorted([row["Carbon Source 1"], row["Carbon Source 2"]])),
    #    axis=1
    #)

    for pair, group in df_merged.groupby("Carbon Pair"):
        if len(group) <= 1:
            continue
        min_row = group.iloc[1:].loc[group.iloc[1:]["Coexistence Strength"].idxmin()]

        # Extract metadata
        cs1 = min_row["Carbon Source 1"].replace("EX_", "").replace("_e", "")
        cs2 = min_row["Carbon Source 2"].replace("EX_", "").replace("_e", "")
        ko1 = cs1
        ko2 = cs2
        bound1 = min_row["KO Bound Source 1"]
        bound2 = min_row["KO Bound Source 2"]

        # Find matching CSV files
        pattern = f"*Mut1_KO=EX_{ko1}_e,_Bound={bound1}_vs_Mut2_KO=EX_{ko2}_e,_Bound={bound2}_Fitness_*_Monoculture_biomass.csv"
        files = glob.glob(os.path.join(biomass_dir, pattern))

        # Collect strain-specific files
        f_strain1 = [f for f in files if "Strain1" in f]
        f_strain2 = [f for f in files if "Strain2" in f]

        # Extract metrics
        r1, k1 = extract_growth_metrics_from_csv(f_strain1[0], ko1, bound1) if f_strain1 else (None, None)
        r2, k2 = extract_growth_metrics_from_csv(f_strain2[0], ko2, bound2) if f_strain2 else (None, None)

        if None not in [r1, r2, k1, k2]:
            records.append({
                "Carbon Pair": pair,
                "r1": r1,
                "r2": r2,
                "K1": k1,
                "K2": k2,
                "Abs Growth Rate Diff": abs(r1 - r2),
                "Abs Carrying Capacity Diff": abs(k1 - k2),
                "Min Coexistence Strength": min_row["Coexistence Strength"]
            })

    return pd.DataFrame(records)


def nicheOverlap(wt,mut,succ_conc=0.1,glc_conc=0.01,coex=False):
    N=2
    wt_init_pop = np.linspace(0.01,0.99,N)*5e-6 # was 1e-7
    mut_init_pop = np.linspace(0.01,0.99,N)[::-1]*5e-6 #was 1e-77
    wt_freq = np.linspace(0.01,0.99,N)
    m_wt = np.zeros(N)
    m_mut = np.zeros(N)
    s_wt = np.zeros(N)
    
    
    for j in range(N):
        
    # set its initial biomass, 5e-6 gr at coordinate [0,0]
        wt.initial_pop = [0, 0, wt_init_pop[j]]
        mut.initial_pop = [0, 0, mut_init_pop[j]]
    
    # create an empty layout
        test_tube = c.layout()
    
        # add the models to the test tube
        test_tube.add_model(wt)
        test_tube.add_model(mut)
        
        # Add glucose to the media 
        test_tube.set_specific_metabolite('succ_e', succ_conc)
        test_tube.set_specific_metabolite('glc__D_e', glc_conc)
        # Add typical trace metabolites and oxygen coli as static
        trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
        for i in trace_metabolites:
            test_tube.set_specific_metabolite(i, 1000)
            test_tube.set_specific_static(i, 1000)

        comp_params = c.params()
        comp_params.set_param('maxCycles', 240)
        comp_params.set_param('writeMediaLog', True)
        comp_params.set_param('writeFluxLog', True)
        
        
        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()
    
        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
    
        #myplot = biomass.drop(columns=['cycle']).plot(x = 't')
        #myplot.set_ylabel("Biomass (gr.)")
        
        m_wt[j] = np.log10(biomass[wt.id].iloc[-1]/biomass[wt.id][0])
        m_mut[j] = np.log10(biomass[mut.id].iloc[-1]/biomass[mut.id][0])
        
    s_wt = m_wt-m_mut
    s_mut = m_mut-m_wt
    
    print(s_wt)
    print(s_mut)
    niche_overlap = (s_wt[0]-s_wt[-1])/(wt_freq[0]-wt_freq[-1])
    
    if not coex:
        return niche_overlap
    else:
        return niche_overlap, np.min(np.array([s_wt[0],s_mut[1]]))
    
def nicheOverlapSaveTimeSeries(wt,mut,succ_conc=0.1,glc_conc=0.01):
    N=2
    wt_init_pop = np.linspace(0.01,0.99,N)*5e-6 # was 1e-7
    mut_init_pop = np.linspace(0.01,0.99,N)[::-1]*5e-6 #was 1e-77
    wt_freq = np.linspace(0.01,0.99,N)
    m_wt = np.zeros(N)
    m_mut = np.zeros(N)
    s_wt = np.zeros(N)
    biomass_list = []
    media_list = []

    
    for j in range(N):
        
    # set its initial biomass, 5e-6 gr at coordinate [0,0]
        wt.initial_pop = [0, 0, wt_init_pop[j]]
        mut.initial_pop = [0, 0, mut_init_pop[j]]
    
    # create an empty layout
        test_tube = c.layout()
    
        # add the models to the test tube
        test_tube.add_model(wt)
        test_tube.add_model(mut)
        
        # Add glucose to the media 
        test_tube.set_specific_metabolite('succ_e', succ_conc)
        test_tube.set_specific_metabolite('glc__D_e', glc_conc)
        # Add typical trace metabolites and oxygen coli as static
        trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
        for i in trace_metabolites:
            test_tube.set_specific_metabolite(i, 1000)
            test_tube.set_specific_static(i, 1000)
            
        comp_params = c.params()
        comp_params.set_param('maxCycles', 240)
        comp_params.set_param('writeMediaLog', True)
        comp_params.set_param('writeFluxLog', True)
        #comp_params.set_param('MediaLogRate',1)

        
        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()
    
        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
        
        biomass_list.append(biomass)
        media = comp_assay.media.copy()
        media = media[media.conc_mmol<900]
        media_list.append(media)
        #myplot = biomass.drop(columns=['cycle']).plot(x = 't')
        #myplot.set_ylabel("Biomass (gr.)")
        
        m_wt[j] = np.log10(biomass[wt.id].iloc[-1]/biomass[wt.id][0])
        m_mut[j] = np.log10(biomass[mut.id].iloc[-1]/biomass[mut.id][0])
        
    s_wt = m_wt-m_mut
    s_mut = m_mut-m_wt
    
    print(s_wt)
    print(s_mut)
    niche_overlap = (s_wt[0]-s_wt[-1])/(wt_freq[0]-wt_freq[-1])
    

    return niche_overlap, np.min(np.array([s_wt[0],s_mut[1]])), biomass_list, media_list

def nicheOverlapSaveTimeSeriesMultiple(wt,mut,s1,s2,s1_conc=0.1,s2_conc=0.01):
    N=2
    wt_init_pop = np.linspace(0.01,0.99,N)*5e-6 # was 1e-7
    mut_init_pop = np.linspace(0.01,0.99,N)[::-1]*5e-6 #was 1e-77
    wt_freq = np.linspace(0.01,0.99,N)
    m_wt = np.zeros(N)
    m_mut = np.zeros(N)
    s_wt = np.zeros(N)
    biomass_list = []
    media_list = []
    print('Source1',s1,'with conc',s1_conc)
    print('Source2',s2,'with conc',s2_conc)


    
    for j in range(N):
        
    # set its initial biomass, 5e-6 gr at coordinate [0,0]
        wt.initial_pop = [0, 0, wt_init_pop[j]]
        mut.initial_pop = [0, 0, mut_init_pop[j]]
    
    # create an empty layout
        test_tube = c.layout()
    
        # add the models to the test tube
        test_tube.add_model(wt)
        test_tube.add_model(mut)
        
        # Add glucose to the media 
        test_tube.set_specific_metabolite(s1[3:], s1_conc)
        test_tube.set_specific_metabolite(s2[3:], s2_conc)
        
                # Now set media concentrations
        #test_tube.set_specific_metabolite('EX_glc__D_e', 0.05)  # Set glucose in the media
        #test_tube.set_specific_metabolite('EX_gal_e', 0.05)  # Set galactose in the media
        # Add typical trace metabolites and oxygen coli as static
        trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
        for i in trace_metabolites:
            test_tube.set_specific_metabolite(i, 1000)
            test_tube.set_specific_static(i, 1000)
            
        if len(test_tube.models) == 0:
            print("⚠️ No models were added to the layout! Ensure test_tube.add_model() is used.")
            
            
        comp_params = c.params()
        comp_params.set_param('maxCycles', 240)
        comp_params.set_param('writeMediaLog', True)
        comp_params.set_param('writeFluxLog', True)
        #comp_params.set_param('MediaLogRate',1)


        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()
    
        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
        
        biomass_list.append(biomass)
        media = comp_assay.media.copy()
        media = media[media.conc_mmol<900]
        media_list.append(media)
        #myplot = biomass.drop(columns=['cycle']).plot(x = 't')
        #myplot.set_ylabel("Biomass (gr.)")
        
        m_wt[j] = np.log10(biomass[wt.id].iloc[-1]/biomass[wt.id][0])
        m_mut[j] = np.log10(biomass[mut.id].iloc[-1]/biomass[mut.id][0])
        
    s_wt = m_wt-m_mut
    s_mut = m_mut-m_wt
    
    print(s_wt)
    print(s_mut)
    niche_overlap = (s_wt[0]-s_wt[-1])/(wt_freq[0]-wt_freq[-1])
    

    return niche_overlap, np.min(np.array([s_wt[0],s_mut[1]])), biomass_list, media_list

def nicheOverlapSaveTimeSeriesMultipleChooseTime(wt,mut,s1,s2,s1_conc=0.1,s2_conc=0.01,max_cyc=240,init_mean_pop=5e-6):
    N=2
    wt_init_pop = np.linspace(0.01,0.99,N)*init_mean_pop #5e-6 # was 1e-7
    mut_init_pop = np.linspace(0.01,0.99,N)[::-1]*init_mean_pop #5e-6 #was 1e-77
    wt_freq = np.linspace(0.01,0.99,N)
    m_wt = np.zeros(N)
    m_mut = np.zeros(N)
    s_wt = np.zeros(N)
    biomass_list = []
    media_list = []
    print('Source1',s1,'with conc',s1_conc)
    print('Source2',s2,'with conc',s2_conc)


    
    for j in range(N):
        
    # set its initial biomass, 5e-6 gr at coordinate [0,0]
        wt.initial_pop = [0, 0, wt_init_pop[j]]
        mut.initial_pop = [0, 0, mut_init_pop[j]]
    
    # create an empty layout
        test_tube = c.layout()
    
        # add the models to the test tube
        test_tube.add_model(wt)
        test_tube.add_model(mut)
        
        # Add glucose to the media 
        test_tube.set_specific_metabolite(s1[3:], s1_conc)
        test_tube.set_specific_metabolite(s2[3:], s2_conc)
        
                # Now set media concentrations
        #test_tube.set_specific_metabolite('EX_glc__D_e', 0.05)  # Set glucose in the media
        #test_tube.set_specific_metabolite('EX_gal_e', 0.05)  # Set galactose in the media
        # Add typical trace metabolites and oxygen coli as static
        trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
        for i in trace_metabolites:
            test_tube.set_specific_metabolite(i, 1000)
            test_tube.set_specific_static(i, 1000)
            
        if len(test_tube.models) == 0:
            print("⚠️ No models were added to the layout! Ensure test_tube.add_model() is used.")
            
            
        comp_params = c.params()
        comp_params.set_param('maxCycles', max_cyc)
        comp_params.set_param('writeMediaLog', True)
        comp_params.set_param('writeFluxLog', True)
        #comp_params.set_param('MediaLogRate',1)


        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()
    
        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
        
        biomass_list.append(biomass)
        media = comp_assay.media.copy()
        media = media[media.conc_mmol<900]
        media_list.append(media)
        #myplot = biomass.drop(columns=['cycle']).plot(x = 't')
        #myplot.set_ylabel("Biomass (gr.)")
        
        m_wt[j] = np.log10(biomass[wt.id].iloc[-1]/biomass[wt.id][0])
        m_mut[j] = np.log10(biomass[mut.id].iloc[-1]/biomass[mut.id][0])
        
    s_wt = m_wt-m_mut
    s_mut = m_mut-m_wt
    
    print(s_wt)
    print(s_mut)
    niche_overlap = (s_wt[0]-s_wt[-1])/(wt_freq[0]-wt_freq[-1])
    
    if s_wt[0] > s_mut[1]:
        winner = wt.id
    else:
        winner = mut.id
    return niche_overlap, np.min(np.array([s_wt[0],s_mut[1]])), biomass_list, media_list, winner, s_wt[0]-s_mut[1]


def transform_differences(df, 
                          niche_col='Niche_Diff', 
                          fitness_col='SC_Diff', 
                          base=10):
    """
    Given a DataFrame with columns for log10 niche differences and
    log fitness differences, compute:
      - FD_ratio            = exp(fitness_diff)
      - ND_raw              = base ** niche_diff
      - ND_norm             = niche_diff / (1 + niche_diff)
      - ND_norm_stretched   = ND_norm / max(ND_norm)
    
    Returns a new DataFrame with those four columns.
    """
    # work on a copy
    out = df.copy()
    
    # fitness-transform: from log scale back to ratio
    out['FD_ratio'] = np.exp(out[fitness_col])
    
    # niche-transform: raw, then normalized
    out['ND_raw'] = base ** out[niche_col]
    out['ND_norm'] = out[niche_col] / (1.0 + out[niche_col])
    
    # stretch to [0,1]
    max_norm = out['ND_norm'].max()
    out['ND_norm_stretched'] = out['ND_norm'] / max_norm
    
    return out[['FD_ratio', 'ND_raw', 'ND_norm', 'ND_norm_stretched']]

# — example usage —
# transformed = transform_differences(df, niche_col='Niche_Diff', fitness_col='SC_Diff')
# plt.scatter(transformed['ND_norm_stretched'], transformed['FD_ratio'], …)

# in helpers.py
def compute_nadh_yield(cobra_model, substrate, uptake=1.0,
                       nadh_rxn_id="NADH16pp"):
    """
    Fix EX_substrate to exactly –uptake, maximize NADH‐dehydrogenase flux,
    and return that objective value as mmol NADH per mmol substrate.
    Falls back to any reaction containing 'NADH' if NADH16pp isn't present.
    """
    m = cobra_model.copy()
    ex = m.reactions.get_by_id(f"EX_{substrate}")
    ex.lower_bound = -uptake
    ex.upper_bound = 0.0

    if nadh_rxn_id not in m.reactions:
        # pick the first NADH‐producing reaction
        nadh_rxn_id = next(r.id for r in m.reactions if "NADH" in r.id.upper())

    m.objective = nadh_rxn_id
    sol = m.optimize()
    return sol.objective_value

def get_flux_path_lengths(comp_assay, wt, substrate, cycle=0):
    """
    Returns the unweighted and flux-weighted path lengths in the metabolic network
    from the exchange reaction for `substrate` to the biomass reaction.

    Parameters:
    - comp_assay: a COMETS assay object after run()
    - wt: a cobra_model wrapped in c.model
    - substrate: string, e.g. "glc__D_e" (without the 'EX_' prefix)
    - cycle: integer time point to extract fluxes (default: 0)

    Returns:
    - path: list of nodes (alternating reaction & metabolite IDs)
    - unweighted_length: int, number of reaction steps in the path
    - weighted_length: float, sum of 1/|flux| over reaction steps
    """
    # 1) Extract fluxes at the given cycle
    flux_df = comp_assay.flux_log
    sol = flux_df[flux_df.cycle == cycle].set_index('reaction')['flux'].to_dict()

    # 2) Build bipartite graph of reactions ↔ metabolites
    G = nx.DiGraph()
    reaction_ids = {r.id for r in wt.model.reactions}
    for rxn in wt.model.reactions:
        for met in rxn.metabolites:
            G.add_edge(rxn.id, met.id)
            G.add_edge(met.id, rxn.id)

    # 3) Define start and end nodes
    start = f"EX_{substrate}"
    try:
        biomass_rxn = wt.model.reactions.get_by_id('BIOMASS_Ecoli_core_w_GAM').id
    except KeyError:
        # fallback: first reaction containing 'biomass'
        biomass_rxn = next(r.id for r in wt.model.reactions if 'biomass' in r.id.lower())

    # 4) Find the shortest unweighted path
    path = nx.shortest_path(G, source=start, target=biomass_rxn)

    # 5) Compute unweighted path length (count reaction nodes)
    unweighted_length = sum(1 for n in path if n in reaction_ids)

    # 6) Compute flux-weighted length: sum of 1/|flux| for each reaction node
    weighted_length = 0.0
    for n in path:
        if n in reaction_ids:
            flux_val = sol.get(n, 0.0)
            if flux_val != 0:
                weighted_length += 1.0 / abs(flux_val)
            else:
                weighted_length += float('inf')

    return path, unweighted_length, weighted_length

# Example usage:
# path, unweighted, weighted = get_flux_path_lengths(comp_assay, mut1, "glc__D_e", cycle=0)
# print("Path:", path)
# print("Unweighted length:", unweighted)
# print("Flux-weighted length:", weighted)


def returnSCMatrix(wt,mut,s1,s2,s1_conc=0.1,s2_conc=0.01,max_cyc=240, N=2, init_ratio_diff = 1e-2, init_mean_pop = 5e-6):
    #N=2
    wt_init_pop = np.linspace(init_ratio_diff,1-init_ratio_diff,N)*init_mean_pop # was 1e-7
    mut_init_pop = np.linspace(init_ratio_diff,1-init_ratio_diff,N)[::-1]*init_mean_pop #was 1e-77
    wt_freq = np.linspace(init_ratio_diff,1-init_ratio_diff,N)

    m_wt = np.zeros(N)
    m_mut = np.zeros(N)
    s_wt = np.zeros(N)
    biomass_list = []
    media_list = []
    print('Source1',s1,'with conc',s1_conc)
    print('Source2',s2,'with conc',s2_conc)


    
    for j in range(N):
        
    # set its initial biomass, 5e-6 gr at coordinate [0,0]
        wt.initial_pop = [0, 0, wt_init_pop[j]]
        mut.initial_pop = [0, 0, mut_init_pop[j]]
    
    # create an empty layout
        test_tube = c.layout()
    
        # add the models to the test tube
        test_tube.add_model(wt)
        test_tube.add_model(mut)
        
        # Add glucose to the media 
        test_tube.set_specific_metabolite(s1[3:], s1_conc)
        test_tube.set_specific_metabolite(s2[3:], s2_conc)
        
                # Now set media concentrations
        #test_tube.set_specific_metabolite('EX_glc__D_e', 0.05)  # Set glucose in the media
        #test_tube.set_specific_metabolite('EX_gal_e', 0.05)  # Set galactose in the media
        # Add typical trace metabolites and oxygen coli as static
        trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
        for i in trace_metabolites:
            test_tube.set_specific_metabolite(i, 1000)
            test_tube.set_specific_static(i, 1000)
            
        if len(test_tube.models) == 0:
            print("⚠️ No models were added to the layout! Ensure test_tube.add_model() is used.")
            
            
        comp_params = c.params()
        comp_params.set_param('maxCycles', max_cyc)
        comp_params.set_param('writeMediaLog', True)
        comp_params.set_param('writeFluxLog', True)
        comp_params.set_param('MediaLogRate',1)


        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()
    
        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
        
        biomass_list.append(biomass)
        media = comp_assay.media.copy()
        media = media[media.conc_mmol<900]
        media_list.append(media)
        #myplot = biomass.drop(columns=['cycle']).plot(x = 't')
        #myplot.set_ylabel("Biomass (gr.)")
        
        m_wt[j] = np.log10(biomass[wt.id].iloc[-1]/biomass[wt.id][0])
        m_mut[j] = np.log10(biomass[mut.id].iloc[-1]/biomass[mut.id][0])
        
    s_wt = m_wt-m_mut
    s_mut = m_mut-m_wt
    

    #niche_overlap = (s_wt[0]-s_wt[-1])/(wt_freq[0]-wt_freq[-1])
    

    return s_wt, s_mut, wt_freq, media_list

def returnSCMatrixWithBiomass(wt,mut,s1,s2,s1_conc=0.1,s2_conc=0.01,max_cyc=240, N=2, init_ratio_diff = 1e-2, init_mean_pop = 5e-6):
    #N=2
    wt_init_pop = np.linspace(init_ratio_diff,1-init_ratio_diff,N)*init_mean_pop # was 1e-7
    mut_init_pop = np.linspace(init_ratio_diff,1-init_ratio_diff,N)[::-1]*init_mean_pop #was 1e-77
    wt_freq = np.linspace(init_ratio_diff,1-init_ratio_diff,N)

    m_wt = np.zeros(N)
    m_mut = np.zeros(N)
    s_wt = np.zeros(N)
    biomass_list = []
    media_list = []
    print('Source1',s1,'with conc',s1_conc)
    print('Source2',s2,'with conc',s2_conc)


    
    for j in range(N):
        
    # set its initial biomass, 5e-6 gr at coordinate [0,0]
        wt.initial_pop = [0, 0, wt_init_pop[j]]
        mut.initial_pop = [0, 0, mut_init_pop[j]]
    
    # create an empty layout
        test_tube = c.layout()
    
        # add the models to the test tube
        test_tube.add_model(wt)
        test_tube.add_model(mut)
        
        # Add glucose to the media 
        test_tube.set_specific_metabolite(s1[3:], s1_conc)
        test_tube.set_specific_metabolite(s2[3:], s2_conc)
        
                # Now set media concentrations
        #test_tube.set_specific_metabolite('EX_glc__D_e', 0.05)  # Set glucose in the media
        #test_tube.set_specific_metabolite('EX_gal_e', 0.05)  # Set galactose in the media
        # Add typical trace metabolites and oxygen coli as static
        trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
        for i in trace_metabolites:
            test_tube.set_specific_metabolite(i, 1000)
            test_tube.set_specific_static(i, 1000)
            
        if len(test_tube.models) == 0:
            print("⚠️ No models were added to the layout! Ensure test_tube.add_model() is used.")
            
            
        comp_params = c.params()
        comp_params.set_param('maxCycles', max_cyc)
        comp_params.set_param('writeMediaLog', True)
        comp_params.set_param('writeFluxLog', True)
        comp_params.set_param('MediaLogRate',1)


        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()
    
        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
        
        biomass_list.append(biomass)
        media = comp_assay.media.copy()
        media = media[media.conc_mmol<900]
        media_list.append(media)
        #myplot = biomass.drop(columns=['cycle']).plot(x = 't')
        #myplot.set_ylabel("Biomass (gr.)")
        
        m_wt[j] = np.log10(biomass[wt.id].iloc[-1]/biomass[wt.id][0])
        m_mut[j] = np.log10(biomass[mut.id].iloc[-1]/biomass[mut.id][0])
        
    s_wt = m_wt-m_mut
    s_mut = m_mut-m_wt
    

    #niche_overlap = (s_wt[0]-s_wt[-1])/(wt_freq[0]-wt_freq[-1])
    

    return s_wt, s_mut, wt_freq, media_list, biomass_list

def analyze_biomass_growth_mult(biomass_dir: str) -> pd.DataFrame:
    results = []

    for filepath in glob.glob(os.path.join(biomass_dir, "*Fitness_Strain*_Monoculture_biomass.csv")):
        try:
            filename = os.path.basename(filepath)

            match = re.search(
                r"EX_([^,]+)_e_EX_([^,]+)_e_Conc([\d\.]+)_KO(-?\d+)_(-?\d+)_Mut1_KO=EX_([^,]+)_e,_Bound=(-?\d+)_vs_Mut2_KO=EX_([^,]+)_e,_Bound=(-?\d+)_Fitness_(Strain\d+)_Monoculture_biomass.csv",
                filename
            )

            if not match:
                continue

            source1, source2, concentration, ko_mut1, ko_mut2, mut1, bound_mut1, mut2, bound_mut2, strain = match.groups()
            concentration = float(concentration)
            ko_mut1 = int(ko_mut1)
            ko_mut2 = int(ko_mut2)
            bound_mut1 = int(bound_mut1)
            bound_mut2 = int(bound_mut2)

            df = pd.read_csv(filepath)

            ko_strain = mut1 if "Strain1" in strain else mut2
            ko_bound = bound_mut1 if "Strain1" in strain else bound_mut2
            biomass_col = f"EX_{ko_strain}_e_KO_{ko_bound}"

            if biomass_col not in df.columns or "t" not in df.columns:
                continue

            biomass_values = df[biomass_col]
            time_values = df["t"]

            valid = (biomass_values > 0)
            biomass_values = biomass_values[valid]
            time_values = time_values[valid]

            if len(time_values) < 3:
                continue

            carrying_capacity = biomass_values.max()
            growth_rates = np.gradient(biomass_values, time_values)
            max_growth_rate = growth_rates.max()

            # log transform
            log_biomass = np.log(biomass_values)
            max_biomass = biomass_values.max()
            plateau_time = time_values[biomass_values == max_biomass].min()
            exp_mask = time_values < plateau_time

            if sum(exp_mask) < 2:
                continue

            # method 1: slope of linear regression on log biomass
            slope, _, _, _, _ = linregress(time_values[exp_mask], log_biomass[exp_mask])

            # method 2: mean gradient of log biomass
            gradients = np.diff(log_biomass[exp_mask]) / np.diff(time_values[exp_mask])
            avg_gradient = gradients.mean()

            results.append({
                "Carbon Source": ko_strain,
                "KO Bound": ko_bound,
                "Concentration": concentration,
                "Carrying Capacity (K)": carrying_capacity,
                "Max Growth Rate (r)": max_growth_rate,
                "Log Slope (µ_regress)": slope,
                "Avg Gradient (µ_avg)": avg_gradient
            })

        except Exception:
            continue

    df_results = pd.DataFrame(results)

    if df_results.empty:
        raise ValueError("No valid data was processed. Check the filename format and CSV column headers.")

    df_avg = (
        df_results
        .groupby("Carbon Source")[[
            "Max Growth Rate (r)",
            "Carrying Capacity (K)",
            "Log Slope (µ_regress)",
            "Avg Gradient (µ_avg)"
        ]]
        .mean()
        .reset_index()
    )
    df_avg["Growth Rate Rank"] = df_avg["Max Growth Rate (r)"].rank(ascending=False)
    df_avg["Carrying Capacity Rank"] = df_avg["Carrying Capacity (K)"].rank(ascending=False)

    return df_avg

def analyze_biomass_growth(biomass_dir: str) -> pd.DataFrame:
    results = []

    for filepath in glob.glob(os.path.join(biomass_dir, "*Fitness_Strain*_Monoculture_biomass.csv")):
        try:
            filename = os.path.basename(filepath)

            match = re.search(
                r"EX_([^,]+)_e_EX_([^,]+)_e_Conc([\d\.]+)_KO(-?\d+)_(-?\d+)_Mut1_KO=EX_([^,]+)_e,_Bound=(-?\d+)_vs_Mut2_KO=EX_([^,]+)_e,_Bound=(-?\d+)_Fitness_(Strain\d+)_Monoculture_biomass.csv",
                filename
            )

            if not match:
                continue

            source1, source2, concentration, ko_mut1, ko_mut2, mut1, bound_mut1, mut2, bound_mut2, strain = match.groups()
            concentration = float(concentration)
            ko_mut1 = int(ko_mut1)
            ko_mut2 = int(ko_mut2)
            bound_mut1 = int(bound_mut1)
            bound_mut2 = int(bound_mut2)

            df = pd.read_csv(filepath)

            ko_strain = mut1 if "Strain1" in strain else mut2
            ko_bound = bound_mut1 if "Strain1" in strain else bound_mut2
            biomass_col = f"EX_{ko_strain}_e_KO_{ko_bound}"

            if biomass_col not in df.columns or "t" not in df.columns:
                continue

            biomass_values = df[biomass_col]
            time_values = df["t"]

            carrying_capacity = biomass_values.max()
            growth_rates = np.gradient(biomass_values, time_values)
            max_growth_rate = growth_rates.max()

            results.append({
                "Carbon Source": ko_strain,
                "KO Bound": ko_bound,
                "Concentration": concentration,
                "Carrying Capacity (K)": carrying_capacity,
                "Max Growth Rate (r)": max_growth_rate
            })

        except Exception:
            continue

    df_results = pd.DataFrame(results)

    if df_results.empty:
        raise ValueError("No valid data was processed. Check the filename format and CSV column headers.")

    df_avg = (
        df_results
        .groupby("Carbon Source")[["Max Growth Rate (r)", "Carrying Capacity (K)"]]
        .mean()
        .reset_index()
    )
    df_avg["Growth Rate Rank"] = df_avg["Max Growth Rate (r)"].rank(ascending=False)
    df_avg["Carrying Capacity Rank"] = df_avg["Carrying Capacity (K)"].rank(ascending=False)

    return df_avg


def fitnessDifference(wt,mut,succ_conc=0.1,glc_conc=0.01,lagT=False):
        strains = [wt,mut]
        K = np.zeros(len(strains))
        time2NoGrowth = np.zeros(len(strains))
        wt.initial_pop = [0, 0, 5e-6]
        mut.initial_pop = [0, 0, 5e-6]

        for j,strain in enumerate(strains):
            # create an empty layout
            test_tube = c.layout()
            test_tube.add_model(strain)
            test_tube.set_specific_metabolite('succ_e', succ_conc)
            test_tube.set_specific_metabolite('glc__D_e', glc_conc)
            # Add typical trace metabolites and oxygen coli as static
            trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                                 'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
    
            for k in trace_metabolites:
                test_tube.set_specific_metabolite(k, 1000)
                test_tube.set_specific_static(k, 1000)
                
            comp_params = c.params()
            comp_params.set_param('maxCycles', 240)

            comp_assay = c.comets(test_tube, comp_params)
            comp_assay.run()
            biomass = comp_assay.total_biomass
            time2NoGrowth[j] = np.argmax(np.gradient(biomass[strain.id])) / 10
            #print(biomass)
            K[j] = biomass[strain.id][240]
        
        if not lagT:
            return np.log10(K[0]/K[1])
        else:
            return np.log10(K[0]/K[1]),time2NoGrowth
        
def fitnessDifferenceSaveTimeSeries(wt,mut,succ_conc=0.1,glc_conc=0.01):
        strains = [wt,mut]
        K = np.zeros(len(strains))
        time2NoGrowth = np.zeros(len(strains))
        wt.initial_pop = [0, 0, 5e-6]
        mut.initial_pop = [0, 0, 5e-6]
        biomass_list = []
        media_list = []
        for j,strain in enumerate(strains):
            # create an empty layout
            test_tube = c.layout()
            test_tube.add_model(strain)
            test_tube.set_specific_metabolite('succ_e', succ_conc)
            test_tube.set_specific_metabolite('glc__D_e', glc_conc)
            # Add typical trace metabolites and oxygen coli as static
            trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                                 'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
    
            for k in trace_metabolites:
                test_tube.set_specific_metabolite(k, 1000)
                test_tube.set_specific_static(k, 1000)
                
            comp_params = c.params()
            comp_params.set_param('maxCycles', 240)
            comp_params.set_param('writeMediaLog', True)

            comp_assay = c.comets(test_tube, comp_params)
            comp_assay.run()
            biomass = comp_assay.total_biomass
            biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
            
            biomass_list.append(biomass)
            media = comp_assay.media.copy()
            media = media[media.conc_mmol<900]
            media_list.append(media)
            time2NoGrowth[j] = np.argmax(np.gradient(biomass[strain.id])) / 10
            #print(biomass)
            K[j] = biomass[strain.id][240]
        
        return np.log10(K[0]/K[1]),biomass_list,media_list
    
def fitnessDifferenceSaveTimeSeriesMultiple(wt,mut,s1,s2,s1_conc=0.0275,s2_conc=0.0275):
        strains = [wt,mut]
        K = np.zeros(len(strains))
        time2NoGrowth = np.zeros(len(strains))
        wt.initial_pop = [0, 0, 5e-6]
        mut.initial_pop = [0, 0, 5e-6]
        biomass_list = []
        media_list = []
        for j,strain in enumerate(strains):
            # create an empty layout
            test_tube = c.layout()
            test_tube.add_model(strain)
            test_tube.set_specific_metabolite(s1[3:], s1_conc)
            test_tube.set_specific_metabolite(s2[3:], s2_conc)
            # Add typical trace metabolites and oxygen coli as static
            trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                                 'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
    
            for k in trace_metabolites:
                test_tube.set_specific_metabolite(k, 1000)
                test_tube.set_specific_static(k, 1000)
                
            comp_params = c.params()
            comp_params.set_param('maxCycles', 240)
            comp_params.set_param('writeMediaLog', True)

            comp_assay = c.comets(test_tube, comp_params)
            comp_assay.run()
            biomass = comp_assay.total_biomass
            biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
            
            biomass_list.append(biomass)
            media = comp_assay.media.copy()
            media = media[media.conc_mmol<900]
            media_list.append(media)
            time2NoGrowth[j] = np.argmax(np.gradient(biomass[strain.id])) / 10
            #print(biomass)
            K[j] = biomass[strain.id][240]
        
        return np.log10(K[0]/K[1]),biomass_list,media_list

def fitnessDifferenceSaveTimeSeriesMultipleRandChooseTime(wt, mut, s1, s2, s1_conc=0.0275, s2_conc=0.0275, max_cyc=240):

    strains = [wt, mut]
    K = np.zeros(len(strains))
    time2NoGrowth = np.zeros(len(strains))
    max_growth_gradient = np.zeros(len(strains))
    max_growth_logfit = np.zeros(len(strains))
    wt.initial_pop = [0, 0, 5e-6]
    mut.initial_pop = [0, 0, 5e-6]
    biomass_list = []
    media_list = []

    for j, strain in enumerate(strains):
        test_tube = c.layout()
        test_tube.add_model(strain)
        test_tube.set_specific_metabolite(s1[3:], s1_conc)
        test_tube.set_specific_metabolite(s2[3:], s2_conc)

        trace_metabolites = [
            'ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
            'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e'
        ]
        for k in trace_metabolites:
            test_tube.set_specific_metabolite(k, 1000)
            test_tube.set_specific_static(k, 1000)

        comp_params = c.params()
        comp_params.set_param('maxCycles', max_cyc)
        comp_params.set_param('writeMediaLog', True)

        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()

        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
        biomass_list.append(biomass)

        media = comp_assay.media.copy()
        media = media[media.conc_mmol < 900]
        media_list.append(media)

        biomass_vals = biomass[strain.id]
        time_vals = biomass['t']

        # time to no growth
        growth_rate = np.gradient(biomass_vals, time_vals)
        time2NoGrowth[j] = np.argmax(growth_rate) / 10
        K[j] = biomass_vals[max_cyc]

        # restrict to t <= 2h for gradient method
        mask_2h = time_vals <= 2.0
        t_2h = time_vals[mask_2h]
        y_2h = biomass_vals[mask_2h]
        if len(t_2h) > 2:
            grad = np.diff(y_2h) / np.diff(t_2h)
            max_growth_gradient[j] = grad.mean()

        # restrict to t <= time2NoGrowth[j] for log-linear fit
        mask_logfit = time_vals <= time2NoGrowth[j]
        t_log = time_vals[mask_logfit]
        y_log = biomass_vals[mask_logfit]
        y_log = y_log[y_log > 0]
        t_log = t_log[:len(y_log)]
        if len(t_log) > 2:
            log_y = np.log(y_log)
            slope, _, _, _, _ = linregress(t_log, log_y)
            max_growth_logfit[j] = slope
    
    mgg_list = max_growth_gradient.tolist()
    mgl_list = max_growth_logfit.tolist()
    mgg_diff = mgg_list[0] - mgg_list[1]
    mgl_diff = mgl_list[0] - mgl_list[1]
    
    return (
        np.log10(K[0] / K[1]),
        biomass_list,
        media_list,
        mgg_diff,
        mgl_diff
    )
  
    
def monoRunNBounds(wt,N=10,max_bound=-20,succ_conc=0.1,glc_conc=0.1):
    b = np.linspace(0,max_bound,N)

    biomass_list = np.zeros((N,481,3))
    #wt.change_bounds('EX_succ_e', -20,1000) #-1000,1000

    for j in range(N):
        wt.change_bounds('EX_glc__D_e',b[j],1000)#-1000,1000

    # set its initial biomass, 5e-6 gr at coordinate [0,0]
        wt.initial_pop = [0, 0, 5e-6]
    
    # create an empty layout
        test_tube = c.layout()
    
        # add the models to the test tube
        test_tube.add_model(wt)
        
        # Add glucose to the media 
        test_tube.set_specific_metabolite('succ_e', succ_conc)
        test_tube.set_specific_metabolite('glc__D_e', glc_conc)
        # Add typical trace metabolites and oxygen coli as static
        trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
        for i in trace_metabolites:
            test_tube.set_specific_metabolite(i, 1000)
            test_tube.set_specific_static(i, 1000)
            
        comp_params = c.params()
        comp_params.set_param('maxCycles', 480)
        comp_params.set_param('writeMediaLog', True)
        comp_params.set_param('writeFluxLog', True)
        
        
        comp_assay = c.comets(test_tube, comp_params)
        comp_assay.run()
    
        biomass = comp_assay.total_biomass
        biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
        biomass_list[j,:,:] = biomass
        media = comp_assay.media.copy()
        media = media[media.conc_mmol<900]
        glc_ts = media[media['metabolite']=='glc__D_e']
        succ_ts = media[media['metabolite']=='succ_e']
        carbons = pd.concat([glc_ts, succ_ts])
        
        fig, ax = plt.subplots()
        carbons.groupby('metabolite').plot(x='cycle', ax =ax, y='conc_mmol')
        #ax.set_yscale('log')
        ax.set_xlim((0,480))
        ax.legend(('glucose','succinate'))
        ax.set_ylabel("Concentration (mmol)")
        
    plt.figure()   
    for j in range(N):
        plt.plot(biomass_list[j,:,2],biomass_list[j,:,1],label='bound={0:.1f}'.format(b[j]))
    plt.xlabel('t')
    plt.ylabel('biomass [g]')
    plt.legend()
    plt.show()

    return np.vstack((b,biomass_list[:,-1,1]))

def monoRunNBoundsChangeConc(wt,M=5,N=3,max_bound=-10):
    b = np.linspace(0,max_bound,M)
    glc = np.linspace(0.001,0.1,N)
    suc = np.linspace(0.001,0.1,N)
    

    total_growth = np.zeros((M,N,N))
    biomass_list = np.zeros((M,N,N,481,3))
    #wt.change_bounds('EX_succ_e', -20,1000) #-1000,1000
    
    #loop through glc,suc,bounds
    for k in range(len(glc)):
        for l in range(len(suc)):
            for j in range(M):
                wt.change_bounds('EX_succ_e',b[j],1000)#-1000,1000
                
                # set its initial biomass, 5e-6 gr at coordinate [0,0]
                wt.initial_pop = [0, 0, 5e-6]
                
                # create an empty layout
                test_tube = c.layout()
                
                # add the models to the test tube
                test_tube.add_model(wt)
            
                test_tube.set_specific_metabolite('glc__D_e', glc[k])
                test_tube.set_specific_metabolite('succ_e', suc[l])
        # Add typical trace metabolites and oxygen coli as static
                trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                             'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
        
                for i in trace_metabolites:
                    test_tube.set_specific_metabolite(i, 1000)
                    test_tube.set_specific_static(i, 1000)
                    
                comp_params = c.params()
                comp_params.set_param('maxCycles', 480)
                comp_params.set_param('writeMediaLog', True)
                comp_params.set_param('writeFluxLog', True)
                
                
                comp_assay = c.comets(test_tube, comp_params)
                comp_assay.run()
            
                biomass = comp_assay.total_biomass
                biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
                biomass_list[j,k,l,:,:] = biomass
                total_growth[j,k,l] = biomass_list[j,k,l,-1,1]
                
    return b,total_growth

def doubleMonoRunNBoundsChangeConc(wt,carb='glc__D_e',carbb='glc__D_e',M=5,N=5,max_bound=-10):
    t_max = 240
    b = np.linspace(0,max_bound,M)
    conc = np.linspace(0.001,0.02,N)    
    total_growth = np.zeros((M,N))
    biomass_list = np.zeros((M,N,t_max+1,3))
    #wt.change_bounds('EX_succ_e', -20,1000) #-1000,1000
    
    #loop through glc,suc,bounds
    for k in range(len(conc)):
        for j in range(M):
            wt.change_bounds('EX_'+carbb,b[j],1000)#-1000,1000
            #wt.change_bounds('EX_'+'succ_e',b[j],1000)#-1000,1000
            
            # set its initial biomass, 5e-6 gr at coordinate [0,0]
            wt.initial_pop = [0, 0, 5e-6]
            
            # create an empty layout
            test_tube = c.layout()
            
            # add the models to the test tube
            test_tube.add_model(wt)
        
            test_tube.set_specific_metabolite(carb, conc[k])
            #test_tube.set_specific_metabolite('succ_e', conc[k])
            # Add typical trace metabolites and oxygen coli as static
            trace_metabolites = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu2_e', 'fe2_e', 'fe3_e', 'h_e', 'k_e', 'h2o_e', 'mg2_e',
                         'mn2_e', 'mobd_e', 'na1_e', 'ni2_e', 'nh4_e', 'o2_e', 'pi_e', 'so4_e', 'zn2_e']
    
            for i in trace_metabolites:
                test_tube.set_specific_metabolite(i, 1000)
                test_tube.set_specific_static(i, 1000)
                
            comp_params = c.params()
            comp_params.set_param('maxCycles', t_max)
            comp_params.set_param('writeMediaLog', True)
            comp_params.set_param('writeFluxLog', True)
            
            
            comp_assay = c.comets(test_tube, comp_params)
            comp_assay.run()
        
            biomass = comp_assay.total_biomass
            biomass['t'] = biomass['cycle'] * comp_assay.parameters.all_params['timeStep']
            biomass_list[j,k,:,:] = biomass
            total_growth[j,k] = biomass_list[j,k,-1,1]
            
            pd.DataFrame({'bound':np.tile(b,N),'od':total_growth.flatten(),'conc':np.repeat(conc,M)})
            
    return pd.DataFrame({'bound':np.tile(b,N),'od':total_growth.flatten(),'conc':np.repeat(conc,M)})
    
def flux_distance(wt,mut,comp_assay):
    wt_flux = (comp_assay.fluxes_by_species[wt.id]).to_numpy()
    mut_flux = (comp_assay.fluxes_by_species[mut.id]).to_numpy()
    
    mask = np.nonzero(np.sum(wt_flux, axis = 1))[0]
    wt_flux_mask = wt_flux[np.ix_(mask, mask)]
    mask_m = np.nonzero(np.sum(mut_flux, axis = 1))[0]
    mut_flux_mask = mut_flux[np.ix_(mask_m, mask_m)]   
    
    norm_wt = wt_flux_mask / wt_flux_mask.max()
    norm_mut = mut_flux_mask / mut_flux_mask.max()
    return np.linalg.norm(norm_wt-norm_mut)

def flux_cosim(wt,mut,comp_assay,rxns=None):
    wt_flux = (comp_assay.fluxes_by_species[wt.id]).to_numpy()
    mut_flux = (comp_assay.fluxes_by_species[mut.id]).to_numpy()
    wt_dict_flux = comp_assay.fluxes_by_species[wt.id]
    mut_dict_flux = comp_assay.fluxes_by_species[mut.id]
    if rxns==None:
        exwt = (wt_dict_flux[list(wt.reactions[wt.reactions['EXCH'] == True].REACTION_NAMES)]).to_numpy()
        exmut = (mut_dict_flux[list(wt.reactions[wt.reactions['EXCH'] == True].REACTION_NAMES)]).to_numpy()
    else:
        exwt = (wt_dict_flux[rxns]).to_numpy()
        exmut = (mut_dict_flux[rxns]).to_numpy()
    
    cos = np.zeros(wt_flux.shape[0])
    for i in range(wt_flux.shape[0]):
        #cos[i] = np.dot(wt_flux[i,:],mut_flux[i,:]) / (np.linalg.norm(wt_flux[i,:])*np.linalg.norm(mut_flux[i,:]))
        cos[i] = np.dot(exwt[i,:],exmut[i,:]) / ((np.linalg.norm(exwt[i,:])*np.linalg.norm(exmut[i,:]))+0.001)

    return cos

def flux_cosim_df(wtdf,mutdf):
    cos = np.zeros(wtdf.shape[0])
    wtnp = wtdf.to_numpy()
    mtnp = mutdf.to_numpy()
    for i in range(wtdf.shape[0]):
        #cos[i] = np.dot(wt_flux[i,:],mut_flux[i,:]) / (np.linalg.norm(wt_flux[i,:])*np.linalg.norm(mut_flux[i,:]))
        cos[i] = np.dot(wtnp[i,:],mtnp[i,:]) / ((np.linalg.norm(wtnp[i,:])*np.linalg.norm(mtnp[i,:]))+0.001)

    return cos


def jaccard_fluxdata(fluxdata,min_thresh=1e-06):
    '''
    from https://github.com/sotarotakano/MetabolicHierarchy/blob/main/JaccardPathway.py
    compute jaccard distance between every set of fluxdata in a dataframe,
    in cafba, forward and reverse reactions are considered as single reaction'''
    Dist = np.array([])
    for i in range(0,fluxdata.shape[0]):
        flux_i = fluxdata.iloc[i,:].to_numpy()
        active_flux_i = {x for x in fluxdata.iloc[i,abs(flux_i) > min_thresh].index}
        active_flux_i = {x.replace("_rev","") for x in active_flux_i}
        for j in range(i+1,fluxdata.shape[0]):
            flux_j = fluxdata.iloc[j,:].to_numpy()
            active_flux_j = {x for x in fluxdata.iloc[j,abs(flux_j) > min_thresh].index}
            active_flux_j = {x.replace("_rev","") for x in active_flux_j}
            j_dist = 1-(len(active_flux_i & active_flux_j)/len(active_flux_i | active_flux_j))
            Dist = np.insert(Dist,len(Dist),j_dist) 
    return Dist



def sanitize_filename(text):
    """Replaces spaces and special characters in filenames with underscores."""
    text = re.sub(r"[()\s]", "_", text)  # Replace spaces & parentheses with underscores
    return text

def extract_time_series(simulation_data, carbon_pair, knockout_source, experiment_type, s1_conc, s2_conc):
    """
    Extracts biomass and media time series data for both invasion experiments 
    (one where strain1 invades, one where strain2 invades).
    """
    from decimal import Decimal
    
    precise_s1 = Decimal(str(s1_conc))  
    carbon_key = (carbon_pair, precise_s1)

    if carbon_key not in simulation_data:
        print(f"❌ No data found for {carbon_pair} at concentration {s1_conc}")
        print(f"Available keys: {list(simulation_data.keys())}")
        return None, None, None, None
    
    exp_key = f"WT vs Mut1 (KO of {carbon_pair[0]})" if knockout_source == carbon_pair[0] else f"WT vs Mut2 (KO of {carbon_pair[1]})"

    if exp_key not in simulation_data[carbon_key]:
        print(f"⚠️ No data found for {exp_key} in simulation_data")
        return None, None, None, None

    # Extract biomass and media
    biomass_data = simulation_data[carbon_key][exp_key]["Biomass"]
    media_data = simulation_data[carbon_key][exp_key]["Media"]

    # If the data is not stored as a list, wrap it into a list
    if isinstance(biomass_data, pd.DataFrame):
        biomass_data = [biomass_data]  # Convert single DataFrame to list
    if isinstance(media_data, pd.DataFrame):
        media_data = [media_data]  # Convert single DataFrame to list

    if len(biomass_data) < 2 or len(media_data) < 2:
        print(f"⚠️ Warning: Expected two invasion experiments, but found only {len(biomass_data)}")
        return None, None, None, None

    return biomass_data[0], biomass_data[1], media_data[0], media_data[1]


def save_simulation_data_to_csv(simulation_data, output_dir="simulation_outputs"):
    """
    Saves biomass and media time series for each experiment into structured CSV files.
    Handles both invasion directions in `Niche` and both monoculture tests in `Fitness`.
    """
    os.makedirs(output_dir, exist_ok=True)

    for (source1, source2), concentration in simulation_data.keys():
        for exp_key, exp_data in simulation_data[(source1, source2), concentration].items():  
            for exp_type in ["Niche", "Fitness"]:  # Handle both experiment types
                if exp_type in exp_data:
                    for invasion_case, result_data in exp_data[exp_type].items():
                        biomass_data = result_data["Biomass"]
                        media_data = result_data["Media"]
                        exp_key_san = sanitize_filename(exp_key)
                        # Save Biomass
                        if isinstance(biomass_data, pd.DataFrame) and not biomass_data.empty:
                            biomass_file = f"{output_dir}/biomass_{source1}_{source2}_{concentration}_{exp_key_san}_{exp_type}_{invasion_case}.csv"
                            biomass_data.to_csv(biomass_file, index=False)
                            print(f"✅ Saved: {biomass_file}")

                        # Save Media
                        if isinstance(media_data, pd.DataFrame) and not media_data.empty:
                            media_file = f"{output_dir}/media_{source1}_{source2}_{concentration}_{exp_key_san}_{exp_type}_{invasion_case}.csv"
                            media_data["t"] = media_data["cycle"] * 0.1  # Assuming TIME_STEP is known
                            media_data.to_csv(media_file, index=False)
                            print(f"✅ Saved: {media_file}")
                            
def save_results_to_csv(results_dict, filename):
    """
    Converts a dictionary with 2D numpy arrays into a structured CSV file.
    Alternates the "Invasion Experiment" index (1,2,1,2...) to reflect KO strain order.
    """
    rows = []
    
    for (carbon_pair, concentration), array in results_dict.items():
        for col_idx in range(array.shape[1]):  # Loop over columns first
            for invasion_idx in range(array.shape[0]):  # Loop over 2 invasion experiments
                rows.append({
                    "Carbon Source 1": carbon_pair[0],
                    "Carbon Source 2": carbon_pair[1],
                    "Concentration": concentration,
                    "KO Strain Index": (col_idx % 2) + 1,  # Alternating KO strain (1,2,1,2...)
                    "Value": array[invasion_idx, col_idx]  # Store value
                })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}")
