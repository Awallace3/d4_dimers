import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools
import pandas as pd
from psi4.driver.wrapper_database import database
import psi4
import sys, os
from dispersion import disp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ""))
import src

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
ang_to_bohr = src.constants.Constants().g_aatoau()
hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

# You will need to build https://github.com/Awallace3/dftd4 for pytest to pass
dftd4_bin = "/theoryfs2/ds/amwalla3/.local/bin/dftd4"
data_pkl = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/sr.pkl"

# y_p: -0.3581934534333338, y: 1.3651143133442858
def test_SR_4():
    df = pd.read_pickle(data_pkl)
    params = src.paramsTable.get_params("SAPT0_adz_3_IE_ATM")
    params_2B, params_ATM = src.paramsTable.generate_2B_ATM_param_subsets(params)
    y_targets = [-0.3581934534333338]
    for i in [0]:
        v = src.locald4.compute_disp_2B_BJ_ATM_SR_dimer(
            df.iloc[i],
            params_2B,
            params_ATM,
            mult_out=1.0,
            SR_func=disp.disp_SR_4,
        )
        assert np.isclose(v, y_targets[i], atol=1.0e-5)
