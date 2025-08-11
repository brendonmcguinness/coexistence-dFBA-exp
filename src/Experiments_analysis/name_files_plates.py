#!/usr/bin/env python3
import os
from pathlib import Path

# --- CONFIGURE THIS ---
folder = Path('/Users/brendonmcguinness/Documents/QLS/python_workspace/wetlab_analysis/colony_counting/exp_june28_2025/T1/heic')
# strains in order; the script will pick the 4th only if it sees 48 files
base_strains = ['MG_MGK', 'manX_dauA', 'ptsG_dctA']
extra_strain = 'ptsG_dctA_1G10S'
# ------------------------

# collect all .heic/.HEIC files, sorted by modification time
files = [p for p in folder.iterdir() if p.suffix.lower() == '.heic']
files.sort(key=lambda p: p.stat().st_mtime)

n = len(files)
print(f"Found {n} HEIC files in {folder!r}")
if n not in (36, 48):
    raise ValueError(f"Expected 36 or 48 files, but found {n}. Is this the right folder?")

# build the strain list
strains = base_strains + ([extra_strain] if n == 48 else [])

# each strain gets 12 files
if len(strains) * 12 != n:
    raise RuntimeError(f"Strains × 12 = {len(strains)*12}, but you have {n} files.")

# naming rules
ratios = ['5_95','95_5']
reps   = [1,2,3,1,2,3]
kans   = [False,False,False,True,True,True]

i = 0
for strain in strains:
    for ratio in ratios:
        for rep, is_kan in zip(reps, kans):
            old = files[i]
            new_name = f"{strain}_{ratio}_{rep}{'_kan' if is_kan else ''}.heic"
            new = folder / new_name
            print(f"Renaming {old.name} → {new_name}")
            os.rename(old, new)
            i += 1

print("All done!")
