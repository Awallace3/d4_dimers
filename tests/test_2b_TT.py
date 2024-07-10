import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools
import pandas as pd
import sys, os
from dispersion import disp
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ""))
import src

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
ang_to_bohr = src.constants.Constants().g_aatoau()
hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

# You will need to build https://github.com/Awallace3/dftd4 for pytest to pass
dftd4_bin = "/theoryfs2/ds/amwalla3/.local/bin/dftd4"
data_pkl = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/test.pkl"


def test_compute_2B_TT():
    df = pd.read_pickle(data_pkl)
    id_list = [0, 500, 2500, 2800, 2600, 6000]
    params_2B_BJ, params_ATM_BJ = src.paramsTable.param_lookup("sadz")
    params_2B, params_ATM = src.paramsTable.param_lookup("2B_TT_START4")
    print(params_2B)
    print(params_ATM)
    energies = np.zeros((len(id_list), 4))
    r4r2_ls = src.r4r2.r4r2_vals_ls()
    for n, i in enumerate(id_list):
        print(i)
        row = df.iloc[i]
        print(row['System'])
        print(f"{tools.closest_intermolecular_contact_dimer(row['Geometry'], row['monAs'], row['monBs']):.2f} Angstrom shortest contact distance")
        tools.print_cartesians(row['Geometry'])
        tt_disp = src.locald4.compute_disp_2B_TT_ATM_TT_dimer(
            row,
            params_2B,
            params_ATM,
        )
        bj_disp = src.locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B_BJ,
            params_ATM_BJ,
        )
        # TT dispersion energy
        energies[n, 0] = tt_disp
        # d4(BJ) from sapt0/adz parameters
        energies[n, 1] = bj_disp
        # sapt0 dispersion
        energies[n, 2] = row['SAPT0_adz'][-1]
        # optimization towards...
        energies[n, 3] = row['Benchmark'] - sum(row['SAPT0_adz'][1:-1])
    print(energies)
    assert np.allclose(energies[:, 0], energies[:, 1], atol=1e0)

test_compute_2B_TT()
