import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import cometspy as c
from cobra.io import load_model
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

def run_one_cycle(cobra_model, substrate, uptake=10.0, dt=1.0):
    # wrap and allow uptake
    w = c.model(cobra_model)
    #w.set_biomass_reaction("BIOMASS_Ec_iJO1366_WT_53p95M")
    w.change_bounds(f"EX_{substrate}", -uptake, 1000.0)
    w.initial_pop = [0,0,1e-6]
    L = c.layout()
    L.add_model(w)
    L.set_specific_metabolite(substrate, 0.0275)
    # traces
    trace_mets = ['ca2_e','cl_e','cobalt2_e','cu2_e','fe2_e','fe3_e','h_e',
                  'k_e','h2o_e','mg2_e','mn2_e','mobd_e','na1_e','ni2_e',
                  'nh4_e','o2_e','pi_e','so4_e','zn2_e']
    for tm in trace_mets:
        L.set_specific_metabolite(tm,1e3); L.set_specific_static(tm,1e3)
    p = c.params()
    p.set_param('maxCycles', 1)
    p.set_param('timeStep', dt)
    p.set_param('writeFluxLog', True)
    p.set_param('FluxLogRate', 1)
    assay = c.comets(L, p)
    assay.run()
    # μ estimate
    bm = assay.total_biomass
    N0, N1 = bm.iloc[0][w.id], bm.iloc[-1][w.id]
    mu = (N1/N0 - 1)/dt
    # flux dict
    flux_df = assay.fluxes_by_species[w.id]
    row = flux_df.iloc[0]
    # only keep reaction columns
    rids = {r.id for r in cobra_model.reactions}
    sol  = {c: float(row[c]) for c in row.index if c in rids}
    return mu, sol

# def reaction_path_metrics(sol, cobra_model, substrate, biomass_rxn):
#     # build reaction-only undirected graph
#     active = {rid:v for rid,v in sol.items() if abs(v)>1e-12}
#     G = nx.Graph()
#     for rid,v in active.items():
#         rxn = cobra_model.reactions.get_by_id(rid)
#         for m in rxn.metabolites:
#             for other in m.reactions:
#                 oid = other.id
#                 if oid in active and oid!=rid:
#                     # cost on edge = 1/|flux of target reaction|
#                     G.add_edge(rid, oid, weight=1/abs(active[oid]))
#     src = f"EX_{substrate}"
#     # shortest path & length
#     try:
#         dist = nx.dijkstra_path_length(G, src, biomass_rxn, weight='weight')
#         path = nx.dijkstra_path(G, src, biomass_rxn, weight='weight')
#     except nx.NetworkXNoPath:
#         dist,path = np.nan, []
#     # bottleneck = min flux along path
#     bottleneck = min(abs(sol[r]) for r in path) if path else np.nan
#     return dist, bottleneck, path


def reaction_path_metrics(sol, cobra_model, substrate, biomass_rxn):
    """
    Compute weighted shortest-path length and bottleneck flux
    from EX_<substrate> to biomass reaction, excluding uptake itself.
    """
    # 1) Active reactions = those with nonzero flux
    active = {rid: v for rid, v in sol.items() if abs(v) > 1e-12}
    G = nx.Graph()

    # 2) Build the reaction–reaction graph
    for rid, flux_val in active.items():
        rxn = cobra_model.reactions.get_by_id(rid)
        for met in rxn.metabolites:
            for other in met.reactions:
                oid = other.id
                if oid in active and oid != rid:
                    G.add_edge(rid, oid, weight=1.0/abs(active[oid]))

    # 3) Add biomass node if needed
    if biomass_rxn not in G:
        G.add_node(biomass_rxn)
        bio_rxn = cobra_model.reactions.get_by_id(biomass_rxn)
        for met in bio_rxn.metabolites:
            for pred in met.reactions:
                pid = pred.id
                if pid in active:
                    G.add_edge(pid, biomass_rxn, weight=0.0)

    src = f"EX_{substrate}"
    # 4) Compute shortest path & cost
    try:
        dist = nx.dijkstra_path_length(G, src, biomass_rxn, weight='weight')
        path = nx.dijkstra_path(G, src, biomass_rxn, weight='weight')
    except nx.NetworkXNoPath:
        return np.nan, np.nan, []

    # 5) Bottleneck = smallest non-uptake flux on the path
    # exclude biomass_rxn and the uptake reaction itself (src)
    other_fluxes = [abs(sol[r]) for r in path if r in sol and r not in {biomass_rxn, src}]
    if other_fluxes:
        bottleneck = min(other_fluxes)
    else:
        # fallback if no other reactions: include all except biomass
        bottleneck = min(abs(sol[r]) for r in path if r in sol and r != biomass_rxn)

    return dist, bottleneck, path


def cofactor_sums(sol, cobra_model):
    cof = {}
    for cof_id, sign in [('ATP','atp_c'),('NADH','nadh_c'),('NADPH','nadph_c')]:
        # reactions that produce that cofactor
        prods = [r.id for r in cobra_model.reactions
                 if r.metabolites.get(cobra_model.metabolites.get_by_id(sign),0)>0]
        cof[cof_id] = sum(sol.get(r,0.0) for r in prods)
    return cof

if __name__=="__main__":
    os.environ['COMETS_HOME']='/Applications/COMETS'
    os.environ['GUROBI_COMETS_HOME'] = '/Library/gurobi1003/macos_universal2'
    os.environ['GRB_LICENSE_FILE']='/Library/gurobi1003/macos_universal2/gurobi.lic'
    cobra_model = load_model("iJO1366")
    #biomass_rxn = cobra_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_WT_53p95M")
    #cobra_model.objective = biomass_rxn    
    biomass_rxn = "BIOMASS_Ec_iJO1366_WT_53p95M"
    #substrates   = ["glc__D_e","fru_e","succ_e"]
    substrates = [
        "glc__D_e", "fru_e", "gal_e", "xyl__D_e", "succ_e", 
        "mal__L_e", "fum_e", "glyc_e", "cit_e"]
    substrates = [
        # Monosaccharides
        "glc__D_e",    # D-glucose
        "fru_e",       # D-fructose
        "gal_e",       # D-galactose
        "man_e",       # D-mannose
        "rib__D_e",    # D-ribose
        "xyl__D_e",    # D-xylose
    
        # Disaccharides & oligosaccharides
        "suc_e",       # sucrose
        "lac__D_e",    # lactose
        "mel_e",       # melibiose
        "tre_e",       # trehalose
        "malt_e",      # maltose
    
        # Sugar alcohols
        "glyc_e",      # glycerol
        "sbt__D_e",    # sorbitol
        "mann_e",      # mannitol
    
        # C2–C3 one-carbon/small acids
        "form_e",      # formate
        "ac_e",        # acetate
        "etoh_e",      # ethanol
        "pyr_e",       # pyruvate
    
        # C4–C6 organic acids
        "succ_e",      # succinate
        "fum_e",       # fumarate
        "mal__L_e",    # L-malate
        "cit_e",       # citrate
        "oxaloac_e",   # oxaloacetate
        "akg_e",       # alpha-ketoglutarate
    
        # Amino acids (as alternative C+N sources)
        "ala__L_e",    # L-alanine
        "asp__L_e",    # L-aspartate
        "glu__L_e",    # L-glutamate
        "ser__L_e",    # L-serine
        "gly__L_e",    # glycine
    
        # Aromatics & others
        #"phe__L_e",    # L-phenylalanine
        #"tgly_e",      # threonine (sometimes tgly_e)
    ]
    cobra_rxns = {r.id for r in cobra_model.reactions}
    rows = []
    for sub in substrates:
        ex_id = f"EX_{sub}"
        # 1) Skip if the model has no such exchange reaction
        if ex_id not in cobra_rxns:
            print(f"Skipping {sub}: no reaction {ex_id} in model")
            continue
        mu, sol = run_one_cycle(cobra_model, sub)
        wlen, bott, path = reaction_path_metrics(sol, cobra_model, sub, biomass_rxn)
        cof = cofactor_sums(sol, cobra_model)
        # carbon efficiency
        c_up = abs(sol.get(f"EX_{sub}",0.0))
        c_out= abs(sol.get("EX_co2_e",0.0))
        """
        # carbon output: all excreted C-containing metabolites
        c_out = 0.0
        for rxn in cobra_model.reactions:
            if rxn.id.startswith('EX_'):
                met = next(iter(rxn.metabolites.keys()))
                form = getattr(met, 'formula', '') or ''
                # parse C count (e.g. C6 -> 6, default to count of 'C')
                m = re.search(r'C(\d+)', form)
                count_C = int(m.group(1)) if m else form.count('C')
                flux = sol.get(rxn.id, 0.0)
                # positive flux = secretion
                if flux > 0 and count_C>0:
                    c_out += flux * count_C
        """
        # number of carbons in each
        C = {
            "glc__D_e": 6,   # glucose
            "fru_e":   6,   # fructose
            "gal_e":   6,   # galactose
            "xyl__D_e":5,   # xylose
            "succ_e":  4,   # succinate
            "mal__L_e":4,   # malate
            "fum_e":   4,   # fumarate
            "glyc_e":  3,   # glycerol (3C)
            "cit_e":   6    # citrate
        }
        C = {
        # Monosaccharides
        "glc__D_e":  6,   # D-glucose
        "fru_e":     6,   # D-fructose
        "gal_e":     6,   # D-galactose
        "man_e":     6,   # D-mannose
        "rib__D_e":  5,   # D-ribose
        "xyl__D_e":  5,   # D-xylose
    
        # Disaccharides & oligosaccharides
        "suc_e":    12,   # sucrose
        "lac__D_e": 12,   # lactose
        "mel_e":    12,   # melibiose
        "tre_e":    12,   # trehalose
        "malt_e":   12,   # maltose
    
        # Sugar alcohols
        "glyc_e":   3,    # glycerol
        "sbt__D_e": 6,    # sorbitol
        "mann_e":   6,    # mannitol
    
        # C1–C3 one-carbon/small acids
        "form_e":   1,    # formate
        "ac_e":     2,    # acetate
        "etoh_e":   2,    # ethanol
        "pyr_e":    3,    # pyruvate
    
        # C4–C6 organic acids
        "succ_e":   4,    # succinate
        "fum_e":    4,    # fumarate
        "mal__L_e": 4,    # L-malate
        "cit_e":    6,    # citrate
        "oxaloac_e":4,    # oxaloacetate
        "akg_e":    5,    # α-ketoglutarate
    
        # Amino acids (C+N sources)
        "ala__L_e": 3,    # L-alanine
        "asp__L_e": 4,    # L-aspartate
        "glu__L_e": 5,    # L-glutamate
        "ser__L_e": 3,    # L-serine
        "gly__L_e": 2,    # glycine
    
        # Aromatic & other
        #"phe__L_e": 9,    # L-phenylalanine
        #"tgly_e":   4     # L-threonine (if `tgly_e` is your thr exchange)
    }

        C = C[sub]       
        c_eff = (c_up*C - c_out)/(c_up*C) if c_up>0 else np.nan

        rows.append({
            "substrate": sub,
            "mu":         mu,
            "path_len":  wlen,
            "bottleneck_flux": bott,
            "C_eff":     c_eff,
            "Num_Cs":   C,
            **{f"{k}_prod":v for k,v in cof.items()}
        })

    df = pd.DataFrame(rows).set_index("substrate")
    print(df)
    

    # assume `df` is your DataFrame, indexed by substrate, with columns:
    # ['mu', 'path_len', 'bottleneck_flux', 'C_eff', 'ATP_prod', 'NADH_prod', 'NADPH_prod']
    
    # 1) compute the absolute Pearson‐R
    r = df.corr(method='spearman')['mu'].abs().drop('mu')
    
    # 2) turn that into R²
    r2 = r**2
    
    # 3) sort to see which predictor explains most variance
    r2.sort_values(ascending=False, inplace=True)
    print(r2)
    print("Best predictor:", r2.idxmax(), "explaining", r2.max()*100, "% of the variance in mu")
    
    # 1) Load the zero‐crossing CSV
    zc = pd.read_csv('zero_crossing_compare2NADH.csv')
    zc = pd.read_csv('zero_crossings_with_zeros.csv')
    # ensure the column names match exactly
    # zc.columns -> ['Carbon Pair','Zero Crossing',…] 

    # 2) Build a map: substrate -> NADH_prod
    #    (df is indexed by substrate)
    nadh_map = df['NADH_prod'].to_dict()
    mu_map = df['mu'].to_dict()
    ceff_map = df['C_eff'].to_dict()



    # 3) Compute the pairwise absolute NADH difference
    def strip_ex(src: str) -> str:
    # drop only the leading "EX_"
        return src[3:] if src.startswith("EX_") else src


    # 3b) Compute both diff-columns
    def abs_diff(pair, lookup):
        s1, s2 = pair.split(" + ")
        k1, k2 = strip_ex(s1), strip_ex(s2)
        return abs(lookup[k1] - lookup[k2])

    
    zc['NADH_diff'] = zc['Carbon Pair'].map(lambda p: abs_diff(p, nadh_map))
    zc['mu_diff']   = zc['Carbon Pair'].map(lambda p: abs_diff(p, mu_map))
    zc['Ceff_diff']   = zc['Carbon Pair'].map(lambda p: abs_diff(p, ceff_map))

    # 4) Scatter + regression line
    plt.figure(figsize=(5,4))
    sns.regplot(
        data=zc,
        x='mu_diff',
        y='Zero Crossing',
        ci=95,
        line_kws={'color':'red','linewidth':2},
        scatter_kws={'s':30}
    )
    plt.xlabel("|Δ Max growth rate|",fontsize='medium')
    plt.ylabel("Raw niche difference when coexistence occurs",fontsize='medium')
    #plt.title("Zero‐Crossing vs. max growth rate difference")
    plt.tight_layout()
    plt.savefig('limitingsim_maxgrowth5b.pdf')
    plt.show()

    # 5) Print R² 
    slope, intercept, r_val, p_val, std_err = linregress(
        zc['mu_diff'], zc['Zero Crossing']
            
    )
    print(f"NADH Diff vs Zero Crossing: R² = {r_val**2:.3f}, p = {p_val:.3g}")
    
    
    # 4) Scatter colored by mu_diff + regression line
    plt.figure(figsize=(5,4))
    
    # a) scatter with colormap
    sc = plt.scatter(
        zc['mu_diff'], zc['Zero Crossing'],
        c=zc['mu_diff'],           # color by mu_diff
        cmap='viridis_r',            # or 'plasma', 'inferno', etc.
        s=50,
        edgecolor='k',
        clip_on=False,
        alpha=0.8
    )
    
    # b) add regression line only
    sns.regplot(
        data=zc,
        x='mu_diff', y='Zero Crossing',
        scatter=False,             # no scatter here
        ci=95,
        line_kws={'color':'red','linewidth':2}
    )
    
    # c) labels, colorbar, layout
    plt.xlabel("|Δ Max growth rate|", fontsize='medium')
    plt.ylabel("Raw niche difference when coexistence occurs", fontsize='medium')
    
    #cbar = plt.colorbar(sc)
    #cbar.set_label("|Δ Max growth rate|", fontsize='medium')
    plt.xlim((-0.01,0.36))
    plt.tight_layout()
    plt.savefig('limitingsim_maxgrowth5b_colored.pdf', bbox_inches='tight')
    plt.show()

    # 4b) Plot NADH difference (x) vs. mu difference (y)
    plt.figure(figsize=(7,5))
    sns.scatterplot(
        data=df,
        x='C_eff',
        y='mu',

    )
    plt.xlabel(" Carbon efficiency ")
    plt.ylabel(" max growth rate ")
    plt.tight_layout()
    plt.show()

    # 4b) Plot NADH difference (x) vs. mu difference (y)
    plt.figure(figsize=(7,5))
    sns.scatterplot(
        data=df,
        x='NADH_prod',
        y='mu',

    )
    plt.xlabel(" NADH production ")
    plt.ylabel(" max growth rate  ")
    plt.tight_layout()
    plt.show()



    # 1) apply the Michaelis–Menten style transform (ND / (1 + ND))
    zc['ZeroCross_norm'] = zc['Zero Crossing'] #/ (1.0 + zc['Zero Crossing'])
    
    # (optional) if you really want a strict 0–1 stretch, you can then min–max scale:
    zc_min = 0
    zc_max = 4.07 #Niche Difference max
    zc['ZeroCross_norm_mm'] = (zc['ZeroCross_norm'] - zc_min) / (zc_max - zc_min)
    
    # 2) plot the transformed column
    plt.figure(figsize=(5,4))
    sns.regplot(
        data=zc,
        x='mu_diff',
        y='ZeroCross_norm_mm',       # or 'ZeroCross_norm' if you don’t need the extra min–max step
        ci=95,
        line_kws={'color':'red','linewidth':2},
        scatter_kws={'s':30}
    )
    plt.xlim(0.03,0.355)
    plt.xlabel("|Δ Max growth rate|", fontsize='medium')
    plt.ylabel("Niche Difference (ND) at coexistence", fontsize='medium')
    plt.tight_layout()
    plt.savefig('limitingsim_maxgrowth5b_norm.pdf')
    plt.show()