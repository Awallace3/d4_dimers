import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools
import pandas as pd
import sys, os
from dispersion import disp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ""))
import src

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
ang_to_bohr = src.constants.Constants().g_aatoau()
hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

# You will need to build https://github.com/Awallace3/dftd4 for pytest to pass
dftd4_bin = "/theoryfs2/ds/amwalla3/.local/bin/dftd4"
data_pkl = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/test.pkl"

# y_p: -0.3581934534333338, y: 1.3651143133442858


@pytest.fixture
def water1():
    return src.water_data.water_data1_ATM()


@pytest.fixture
def water2():
    return src.water_data.water_data2_ATM()


@pytest.mark.parametrize(
    "geom",
    [
        "water1",
        "water2",
    ],
)
def test_ATM_water(geom, request) -> None:
    """
    compares dftd4 dispersion energies with damping parameters of HF
        HF params = [1.0, 1.61679827, 0.44959224, 3.35743605]
                  = [s6,  s8,         a1,         a2        ]
    """

    (
        params,
        pos,
        carts,
        d4C6s,
        d4C8s,
        pairs,
        d4e,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C8s_A,
        pairs_A,
        d4e_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C8s_B,
        pairs_B,
        d4e_B,
        d4C6s_BTM_B,
    ) = request.getfixturevalue(geom)
    charges = [0, 1]
    target_ATM = d4e

    cs = ang_to_bohr * np.array(carts, copy=True)
    params.append(1.0)
    compute_ATM = src.locald4.compute_bj_f90_ATM(
        pos, cs, d4C6s_ATM, C6s=d4C6s, params=params, ATM_only=False
    )

    print(f"{target_ATM= }\n{compute_ATM = }")
    assert abs(target_ATM - compute_ATM) < 1e-14


@pytest.mark.parametrize(
    "geom",
    [
        ("water1"),
        ("water2"),
    ],
)
def test_ATM_water_IE(geom, request) -> None:
    charges = [0, 1]
    (
        params,
        pos,
        carts,
        d4C6s,
        d4C8s,
        pairs,
        d4e,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C8s_A,
        pairs_A,
        d4e_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C8s_B,
        pairs_B,
        d4e_B,
        d4C6s_ATM_B,
    ) = request.getfixturevalue(geom)
    tools.print_cartesians_pos_carts(pos, carts)
    params.append(1.0)
    print(params)

    carts *= ang_to_bohr
    carts_A *= ang_to_bohr
    carts_B *= ang_to_bohr

    e_d = src.locald4.compute_bj_f90_ATM(
        pos, carts, d4C6s_ATM, C6s=d4C6s, params=params, ATM_only=False
    )
    e_1 = src.locald4.compute_bj_f90_ATM(
        pos_A, carts_A, d4C6s_ATM_A, C6s=d4C6s_A, params=params, ATM_only=False
    )
    e_2 = src.locald4.compute_bj_f90_ATM(
        pos_B, carts_B, d4C6s_ATM_B, C6s=d4C6s_B, params=params, ATM_only=False
    )

    e_total = e_d - (e_1 + e_2)
    d4_e_total = d4e - (d4e_A + d4e_B)
    print(f"{e_d = }")
    print(f"{e_1 = }")
    print(f"{e_2 = }")
    print(f"{e_total = }")
    print(f"{d4e = }")
    print(f"{d4e_A = }")
    print(f"{d4e_B = }")
    print(f"{d4_e_total = }")
    assert abs(d4_e_total - e_total) < 1e-13


@pytest.mark.parametrize(
    "geom",
    [
        ("water1"),
        ("water2"),
    ],
)
def test_disp_ATM_CHG(geom, request):
    charges = [0, 1]
    (
        params,
        pos,
        carts,
        d4C6s,
        d4C8s,
        pairs,
        d4e,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C8s_A,
        pairs_A,
        d4e_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C8s_B,
        pairs_B,
        d4e_B,
        d4C6s_ATM_B,
    ) = request.getfixturevalue(geom)
    tools.print_cartesians_pos_carts(pos, carts)
    params.append(1.0)
    params = np.array(params, dtype=np.float64)
    pos = np.array(pos, dtype=np.int32)
    pos_A = np.array(pos_A, dtype=np.int32)
    pos_B = np.array(pos_B, dtype=np.int32)

    carts *= ang_to_bohr
    carts_A *= ang_to_bohr
    carts_B *= ang_to_bohr
    print(
        pos,
        carts,
        d4C6s,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C6s_ATM_B,
        params,
    )
    params_2B = params.copy()
    params_ATM = params.copy()
    print(params_2B)
    print(params_ATM)

    e_total = disp.disp_2B_BJ_ATM_CHG(
        pos,
        carts,
        d4C6s,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C6s_ATM_B,
        params_2B,
        params_ATM,
    )

    d4_e_total = d4e - (d4e_A + d4e_B)
    print(f"Target   d4= {d4_e_total}")
    print(f"Computed d4= {e_total}")
    assert abs(d4_e_total - e_total) < 1e-13


@pytest.mark.parametrize(
    "geom",
    [
        ("water1"),
    ],
)
def test_disp_ATM_TT_returns(geom, request):
    charges = [0, 1]
    (
        params,
        pos,
        carts,
        d4C6s,
        d4C8s,
        pairs,
        d4e,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C8s_A,
        pairs_A,
        d4e_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C8s_B,
        pairs_B,
        d4e_B,
        d4C6s_ATM_B,
    ) = request.getfixturevalue(geom)
    tools.print_cartesians_pos_carts(pos, carts)
    params.append(1.0)
    params = np.array(params, dtype=np.float64)
    pos = np.array(pos, dtype=np.int32)
    pos_A = np.array(pos_A, dtype=np.int32)
    pos_B = np.array(pos_B, dtype=np.int32)

    carts *= ang_to_bohr
    carts_A *= ang_to_bohr
    carts_B *= ang_to_bohr
    print(
        pos,
        carts,
        d4C6s,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C6s_ATM_B,
        params,
    )
    params_2B = params.copy()
    params_ATM = params.copy()
    print(params_2B)
    print(params_ATM)

    e_TT = disp.disp_2B_BJ_ATM_TT(
        pos,
        carts,
        d4C6s,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C6s_ATM_B,
        params_2B,
        params_ATM,
    )
    e_CHG  = disp.disp_2B_BJ_ATM_CHG(
        pos,
        carts,
        d4C6s,
        d4C6s_ATM,
        pos_A,
        carts_A,
        d4C6s_A,
        d4C6s_ATM_A,
        pos_B,
        carts_B,
        d4C6s_B,
        d4C6s_ATM_B,
        params_2B,
        params_ATM,
    )
    e_TT *= hartree_to_kcalmol
    e_CHG *= hartree_to_kcalmol
    print(f"Computed d4= {e_TT}")
    print(f"Computed d4= {e_CHG}")
    assert type(e_TT) == float
    assert e_TT != e_CHG
