import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools
import pandas as pd
from psi4.driver.wrapper_database import database
import psi4
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ""))
import src

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
ang_to_bohr = src.constants.Constants().g_aatoau()
hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

# You will need to build https://github.com/Awallace3/dftd4 for pytest to pass
dftd4_bin = "/theoryfs2/ds/amwalla3/.local/bin/dftd4"
data_pkl = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/test.pkl"

def test_ATM_water() -> None:
    """
    compares dftd4 dispersion energies with damping parameters of HF
        HF params = [1.0, 1.61679827, 0.44959224, 3.35743605]
                  = [s6,  s8,         a1,         a2        ]
    """

    # TODO
    params = [1, 1.61679827, 0.44959224, 3.35743605]
    pos, carts = src.water_data.water_geom()
    charges = [0, 1]
    d4C6s, d4C8s, pairs, d4e_ATM, d4C6s_ATM = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges, dftd4_bin=dftd4_bin, p=params, s9=1.0, C6s_ATM=True
    )
    with open(".ATM", "r") as f:
        target_ATM = float(f.readline())

    cs = ang_to_bohr * np.array(carts, copy=True)
    print(params)
    params.append(1.0)
    compute_ATM = src.locald4.compute_bj_f90_ATM(
        pos, cs, d4C6s_ATM, params=params, ATM_only=True
    )

    print(f"{target_ATM= }\n{compute_ATM = }")
    assert abs(target_ATM - compute_ATM) < 1e-14


def test_compute_bj_dimer_f90_ATM():
    """
    Tests if the fortran DFTD4 ATM energy is the same as the python version
    """
    # TODO
    df = pd.read_pickle(data_pkl)
    id_list = [0, 500, 2700, 4926]
    params = src.paramsTable.paramsDict()["HF"]
    params.append(1.0)
    p = params[1:]
    print(params, p)
    energies = np.zeros((len(id_list), 2))
    r4r2_ls = src.r4r2.r4r2_vals_ls()
    for n, i in enumerate(id_list):
        print(n, i)
        row = df.iloc[i]
        d4_local_ATM = src.locald4.compute_bj_dimer_f90_ATM(
            p,
            row,
            r4r2_ls=r4r2_ls,
        )
        dftd4_ATM = src.locald4.compute_bj_dimer_DFTD4(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["charges"],
            s9=1.0,
        )
        assert abs(d4_local_ATM - dftd4_ATM) < 1e-12
