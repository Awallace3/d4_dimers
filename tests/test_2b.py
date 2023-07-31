import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools
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
data_pkl = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/test.pkl"


@pytest.fixture
def water1():
    return src.water_data.water_data1()


@pytest.fixture
def water1_undamped():
    return src.water_data.water_data1_undamped()


def HBC6_data():
    return qcel.models.Molecule.from_data(
        """
0 1
C        1.69147262      -0.17006280       0.00000000
H        2.79500199      -0.28101305       0.00000000
O        1.02814129      -1.21720864       0.00000000
O        1.36966587       1.08860681       0.00000000
H        0.34380745       1.18798183       0.00000000
--
0 1
C       -1.69147262       0.17006280       0.00000000
H       -2.79500199       0.28101305       0.00000000
O       -1.02814129       1.21720864       0.00000000
O       -1.36966587      -1.08860681       0.00000000
H       -0.34380745      -1.18798183       0.00000000
units angstrom
"""
    )


def test_Qa_H() -> None:
    """
    test_Q_A_Q_Bs tests Q_A for correct value for H
    """
    Q_H = src.r4r2.r4r2_vals_eff(1)
    assert abs(Q_H - 2.0073489980568899) < 1e-14


def test_Qa_O() -> None:
    """
    test_Q_A_Q_Bs tests Q_A for correct value for H
    """
    Q_O = src.r4r2.r4r2_vals_eff(8)
    print(f"{Q_O = }\n      2.5936168242020377")
    assert abs(Q_O - 2.5936168242020377) < 1e-14


def test_d4_energies_damped(water1) -> None:
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
    ) = water1

    params.pop(0)  # compute_bj_f90 doesn't take in s6

    cs = ang_to_bohr * np.array(carts, copy=True)
    l_e = src.locald4.compute_bj_f90(pos, cs, d4C6s, params=params)

    print(f"{d4e = }\n{l_e = }")
    assert abs(d4e - l_e) < 1e-14


def test_d4_energies_undamped(water1_undamped) -> None:
    """
    compares dftd4 dispersion energies with damping parameters of HF
        HF params = [1.0, 1.0, 0.0, 0.0]
                  = [s6,  s8,  a1,  a2 ]
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
    ) = water1_undamped
    params.pop(0)  # compute_bj_f90 doesn't take in s6
    cs = ang_to_bohr * np.array(carts, copy=True)
    l_e = src.locald4.compute_bj_f90(pos, cs, d4C6s, params=params)

    print(f"{d4e = }\n{l_e = }")
    assert abs(d4e - l_e) < 1e-14


# # parametrize with fixture
# @pytest.mark.parametrize(
#     "filename,expected",
#     [
#         # 1. pass fixture name as a string
#         ("test_us_csv_file_name", ("us", "20220917", "csv")),
#         ("test_gb_csv_file_name", ("gb", "20220917", "csv")),
#     ],
# )
# def test_extract_ctry_date_and_ext_from_fileanme(filename, expected, request):
#     # 2. add 'request' fixture to the test's arguments ^^
#
#     # 3. convert string to fixture value
#     filename = request.getfixturevalue(filename)
#     assert extract_ctry_date_and_ext_from_filename(filename) == expected


@pytest.mark.parametrize(
    "geom",
    [
        ("water1"),
        ("water2"),
    ],
)
def test_dispersion_interaction_energy(geom, request) -> None:
    """
    test_dispersion_interaction_energy
    """
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
        d4C6s_BTM_B,
    ) = request.getfixturevalue(geom)
    p = params.copy()
    tools.print_cartesians_pos_carts(pos, carts)

    carts *= ang_to_bohr
    carts_A *= ang_to_bohr
    carts_B *= ang_to_bohr

    e_d = src.locald4.compute_bj_f90(pos, carts, d4C6s, params)
    e_1 = src.locald4.compute_bj_f90(pos_A, carts_A, d4C6s_A, params)
    e_2 = src.locald4.compute_bj_f90(pos_B, carts_B, d4C6s_B, params)

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
    assert abs(d4_e_total - e_total) < 1e-14

@pytest.mark.parametrize(
    "geom",
    [
        ("water1"),
        ("water2"),
    ],
)
def test_disp_module_2B(geom, request):
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
        d4C6s_BTM_B,
    ) = request.getfixturevalue(geom)
    p = params.copy()
    tools.print_cartesians_pos_carts(pos, carts)
    carts *= ang_to_bohr
    carts_A *= ang_to_bohr
    carts_B *= ang_to_bohr

    params = np.array(params, dtype=np.float64)
    pos = np.array(pos, dtype=np.int32)
    pos_A = np.array(pos_A, dtype=np.int32)
    pos_B = np.array(pos_B, dtype=np.int32)

    e_d = disp.disp_2b(pos, carts, d4C6s, params)
    e_1 = disp.disp_2b(pos_A, carts_A, d4C6s_A, params)
    e_2 = disp.disp_2b(pos_B, carts_B, d4C6s_B, params)

    e_total = e_d - (e_1 + e_2)
    d4_e_total = d4e - (d4e_A + d4e_B)
    assert abs(d4_e_total - e_total) < 1e-14

def test_charged_dftd4():
    """
    Ensures energy computed through dftd4 is different for neutral and charged
    """
    pos, carts = src.water_data.water_geom1()
    pos = pos[:3]
    carts = carts[:3, :]
    params = src.paramsTable.paramsDict()["HF"]
    _, _, _, neutral = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos,
        carts,
        [0],
        p=params,
    )
    _, _, _, cation = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos,
        carts,
        [1],
        p=params,
    )
    assert neutral != cation


def test_C6s_change_dimer_to_monomer_HBC6():
    # HBC6 - doubly hydrogen bonded, grab a single monomer and see if the C6's change
    # do dimer - monomer - monomer
    # do only interatomic pairs
    # These should not agree
    # HBC6 first dimer from /theoryfs2/ds/amwalla3/gits/psi4_amw/psi4/share/psi4/databases/HBC6.py
    params = src.paramsTable.paramsDict()["HF"]
    data = HBC6_data()

    gD = data.geometry
    pD = data.atomic_numbers
    ma = list(data.fragments[0])
    mb = list(data.fragments[1])
    print(data.fragment_charges)
    charges = np.array(
        [
            [int(sum(data.fragment_charges)), 1],
            [int(data.fragment_charges[0]), 1],
            [int(data.fragment_charges[1]), 1],
        ]
    )
    print(charges)

    C6s_dimer, _, _, df_c_e = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pD,
        gD,
        charges[0],
        p=params,
    )
    pA, cA = pD[ma], gD[ma, :]
    pB, cB = pD[mb], gD[mb, :]
    C6s_mA, _, _, _ = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pA,
        cA,
        charges[1],
        p=params,
    )
    C6s_mB, _, _, _ = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pB,
        cB,
        charges[2],
        p=params,
    )

    C6s_monA_from_dimer = src.locald4.get_monomer_C6s_from_dimer(C6s_dimer, ma)
    print(f"{C6s_monA_from_dimer = }")
    print(f"{C6s_mA = }")
    print(np.shape(C6s_monA_from_dimer))
    print(np.shape(C6s_mA))
    diff = np.abs(np.subtract(C6s_monA_from_dimer, C6s_mA))
    assert np.all(diff > 1e-6)


def test_C6s_change_Di_to_Mon_HBC6_IE():
    r4r2_ls = src.r4r2.r4r2_vals_ls()
    # HBC6 - doubly hydrogen bonded, grab a single monomer and see if the C6's change
    # do dimer - monomer - monomer
    # do only interatomic pairs
    # These should not agree
    # HBC6 first dimer from /theoryfs2/ds/amwalla3/gits/psi4_amw/psi4/share/psi4/databases/HBC6.py
    params = src.paramsTable.paramsDict()["HF"]
    data = HBC6_data()

    cD = data.geometry / ang_to_bohr
    print(cD)
    pD = data.atomic_numbers
    # tools.print_cartesians_pos_carts(pD, cD)
    ma = list(data.fragments[0])
    mb = list(data.fragments[1])
    charges = np.array(
        [
            [int(sum(data.fragment_charges)), 1],
            [int(data.fragment_charges[0]), 1],
            [int(data.fragment_charges[1]), 1],
        ]
    )
    pA, cA = pD[ma], cD[ma, :]
    pB, cB = pD[mb], cD[mb, :]
    tools.print_cartesians_pos_carts(pA, cA)
    tools.print_cartesians_pos_carts(pA, cA)
    C6s_dimer, C6s_mA, C6s_mB = src.locald4.calc_dftd4_c6_for_d_a_b(
        cD, pD, pA, cA, pB, cB, charges, p=params, s9=0.0
    )
    C6s_monA_from_dimer = src.locald4.get_monomer_C6s_from_dimer(C6s_dimer, ma)
    C6s_monB_from_dimer = src.locald4.get_monomer_C6s_from_dimer(C6s_dimer, mb)
    cD *= ang_to_bohr
    geom_bohr = np.hstack((np.reshape(pD, (-1, 1)), cD))
    row = {
        "Geometry_bohr": geom_bohr,
        "C6s": C6s_dimer,
        "charges": charges,
        "monAs": ma,
        "monBs": mb,
        "C6_A": C6s_mA,
        "C6_B": C6s_mB,
    }
    d4_mons_individually = src.locald4.compute_bj_dimer_f90(
        params,
        row,
        r4r2_ls=r4r2_ls,
    )
    row = {
        "Geometry_bohr": geom_bohr,
        "C6s": C6s_dimer,
        "charges": charges,
        "monAs": ma,
        "monBs": mb,
        "C6_A": C6s_monA_from_dimer,
        "C6_B": C6s_monB_from_dimer,
    }
    d4_dimer = src.locald4.compute_bj_dimer_f90(
        params,
        row,
        r4r2_ls=r4r2_ls,
    )
    assert np.all(abs(d4_mons_individually - d4_dimer) > 1e-6)


def test_C6s_Di_to_Mon_HBC6_IE_func_call():
    r4r2_ls = src.r4r2.r4r2_vals_ls()
    # HBC6 first dimer from /theoryfs2/ds/amwalla3/gits/psi4_amw/psi4/share/psi4/databases/HBC6.py
    params = src.paramsTable.paramsDict()["HF"]
    data = HBC6_data()

    cD = data.geometry / ang_to_bohr
    print(cD)
    pD = data.atomic_numbers
    # tools.print_cartesians_pos_carts(pD, cD)
    ma = list(data.fragments[0])
    mb = list(data.fragments[1])
    charges = np.array(
        [
            [int(sum(data.fragment_charges)), 1],
            [int(data.fragment_charges[0]), 1],
            [int(data.fragment_charges[1]), 1],
        ]
    )
    pA, cA = pD[ma], cD[ma, :]
    pB, cB = pD[mb], cD[mb, :]
    tools.print_cartesians_pos_carts(pA, cA)
    tools.print_cartesians_pos_carts(pA, cA)
    C6s_dimer, C6s_mA, C6s_mB = src.locald4.calc_dftd4_c6_for_d_a_b(
        cD, pD, pA, cA, pB, cB, charges, p=params, s9=0.0
    )
    cD *= ang_to_bohr
    geom_bohr = np.hstack((np.reshape(pD, (-1, 1)), cD))
    d4_dimer, d4_mons_individually = src.locald4.compute_bj_with_different_C6s(
        geom_bohr,
        ma,
        mb,
        charges,
        C6s_dimer,
        C6s_mA,
        C6s_mB,
        params,
    )
    assert np.all(abs(d4_mons_individually - d4_dimer) > 1e-6)

