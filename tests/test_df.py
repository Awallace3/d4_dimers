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


def test_stored_C6s():
    df = pd.read_pickle(data_pkl)
    id_list = [0, 500, 2700, 4992]
    params = src.paramsTable.paramsDict()["HF"]
    print(params)
    energies = np.zeros((len(id_list), 2))
    r4r2_ls = src.r4r2.r4r2_ls()

    for n, i in enumerate(id_list):
        row = df.iloc[i]
        pos = row["Geometry"][:, 0]
        carts = row["Geometry"][:, 1:]
        charges = row["charges"]
        Ma, Mb = row["monAs"], row["monBs"]
        C6s, _, _, d = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            pos,
            carts,
            charges[0],
            p=params,
        )
        print("gen:", C6s)
        print("row:", row["C6s"])
        assert np.allclose(C6s, row["C6s"], rtol=1e-14)

        mon_ca = carts[Ma]
        mon_pa = pos[Ma]
        C6_As, _, _, A = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pa,
            mon_ca,
            charges[1],
            p=params,
        )
        print("gen:", C6_As)
        print("row:", row["C6_A"])
        assert np.allclose(C6_As, row["C6_A"], rtol=1e-14)

        mon_cb = carts[Mb]
        mon_pb = pos[Mb]
        C6_Bs, _, _, B = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pb,
            mon_cb,
            charges[2],
            p=params,
        )
        assert np.allclose(C6_Bs, row["C6_B"], rtol=1e-14)


def test_dispersion_interaction_energy3() -> None:
    """
    test_dispersion_interaction_energy
    """
    params = [1, 1.61679827, 0.44959224, 3.35743605]
    p = params.copy()
    p.pop(0)
    df = pd.read_pickle(data_pkl)
    r = df.iloc[2700]
    num, coords = r["Geometry_bohr"][:, 0], r["Geometry_bohr"][:, 1:]
    print("angstrom")
    tools.print_cartesians(r["Geometry"], True)
    print()
    print("bohr")
    tools.print_cartesians(r["Geometry_bohr"], True)
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]
    print(num, coords)
    tools.print_cartesians_pos_carts(num, coords)

    e_d = src.locald4.compute_bj_f90(num, coords, r["C6s"], params=p)

    n1, p1 = num[monAs], coords[monAs, :]
    e_1 = src.locald4.compute_bj_f90(n1, p1, r["C6_A"], params=p)

    n2, p2 = num[monBs], coords[monBs, :]
    e_2 = src.locald4.compute_bj_f90(n2, p2, r["C6_B"], params=p)

    e_total = (e_d - (e_1 + e_2)) * hartree_to_kcalmol

    num, coords = r["Geometry"][:, 0], r["Geometry"][:, 1:]
    dftd4 = src.locald4.compute_bj_dimer_DFTD4(
        params, num, coords, monAs, monBs, charges
    )

    print(f"{e_d = }")
    print(f"{e_1 = }")
    print(f"{e_2 = }")
    print(f"{e_total = }")
    print(f"{dftd4 = }")
    assert abs(e_total - dftd4) < 1e-12


def test_compute_bj_f90():
    """
    ensures that the fortran and python versions of the bj dispersion energy are the same
    """
    df = pd.read_pickle(data_pkl)
    id_list = [0, 500, 2700, 4926]
    params = src.paramsTable.paramsDict()["HF"]
    p = params[1:]
    print(params, p)
    energies = np.zeros((len(id_list), 2))
    r4r2_ls = src.r4r2.r4r2_vals_ls()
    for n, i in enumerate(id_list):
        print(i)
        row = df.iloc[i]
        print(row["Geometry_bohr"])
        d4_local = src.locald4.compute_bj_dimer_f90(
            p,
            row,
            r4r2_ls=r4r2_ls,
        )
        dftd4 = src.locald4.compute_bj_dimer_DFTD4(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["charges"],
        )
        energies[n, 0] = d4_local
        energies[n, 1] = dftd4
    print(energies)
    assert np.allclose(energies[:, 0], energies[:, 1], atol=1e-14)


def test_charged_C6s_in_df():
    """
    Ensures that the C6's pre-computed in df are correctly charged
    """
    params = src.paramsTable.paramsDict()["HF"]
    df = pd.read_pickle(data_pkl)
    id_list = [4926]
    for n, i in enumerate(id_list):
        print(i)
        row = df.iloc[i]
        charges = row["charges"]
        print(charges)
        C6s_df_c, _, _, df_c_e = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            row["Geometry"][:, 0],
            row["Geometry"][:, 1:],
            charges[0],
            p=params,
        )
        C6s_c, _, _, df = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            row["Geometry"][:, 0],
            row["Geometry"][:, 1:],
            [0, 1],
            p=params,
        )
        assert not np.all(C6s_df_c == C6s_c)


def test_pairwise_AB_versus_classic_IE():
    """
    test_pairwise_AB_versus_classic_IE ensures that the pairwise AB energy is
    DIFFERENT from the classic IE
    """
    df = pd.read_pickle(data_pkl)
    id_list = [0, 500, 2700, 4926]
    params = src.paramsTable.paramsDict()["HF"]
    p = params[1:]
    energies = np.zeros((len(id_list), 2))
    r4r2_ls = src.r4r2.r4r2_vals_ls()
    for n, i in enumerate(id_list):
        print(n, i)
        row = df.iloc[i]
        d4_ie = src.locald4.compute_bj_dimer_f90(
            p,
            row,
            r4r2_ls=r4r2_ls,
        )
        d4_pairs = src.locald4.compute_bj_pairs_DIMER(
            params,
            row,
            r4r2_ls=r4r2_ls,
        )
        assert abs(d4_ie - d4_pairs) > 1e-6


def test_C6s_change_dimer_to_monomer():
    """
    Checks if the C6's change when a dimer is changed to a monomer
    AND ensures that C6's stored in dataframe are correct for
    each monomer
    """
    params = src.paramsTable.paramsDict()["HF"]
    df = pd.read_pickle(data_pkl)
    id_list = [0, 2600, 4000, 4926, 7000]
    print(id_list)
    for n, i in enumerate(id_list):
        print(i)

        row = df.iloc[i]
        ma = row["monAs"]
        mb = row["monBs"]
        charges = row["charges"]
        geom = row["Geometry"]

        C6s_dimer, _, _, df_c_e = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            geom[:, 0],
            geom[:, 1:],
            charges[0],
            p=params,
        )
        C6_diff = np.abs(np.subtract(C6s_dimer, row["C6s"]))
        assert np.all(C6_diff < 1e-12)
        pA, cA = geom[ma, 0], geom[ma, 1:]
        pB, cB = geom[mb, 0], geom[mb, 1:]
        C6s_mA, _, _, _ = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            pA,
            cA,
            charges[1],
            p=params,
        )
        C6_diff = np.abs(np.subtract(C6s_mA, row["C6_A"]))
        assert np.all(C6_diff < 1e-12)
        C6s_mB, _, _, _ = src.locald4.calc_dftd4_c6_c8_pairDisp2(
            pB,
            cB,
            charges[2],
            p=params,
        )
        C6_diff = np.abs(np.subtract(C6s_mB, row["C6_B"]))
        assert np.all(C6_diff < 1e-12)

        C6s_monA_from_dimer = src.locald4.get_monomer_C6s_from_dimer(C6s_dimer, ma)
        diff = np.abs(np.subtract(C6s_monA_from_dimer, C6s_mA))
        assert not np.all(diff < 1e-12)


def test_C6s_change_dimer_to_monomer_IE():
    params = src.paramsTable.paramsDict()["HF"]
    df = pd.read_pickle(data_pkl)
    id_list = [0, 2600, 4000, 4926, 7000]
    print(id_list)
    for n, i in enumerate(id_list):
        print(i)

        row = df.iloc[i]
        ma = row["monAs"]
        mb = row["monBs"]
        charges = row["charges"]
        geom = row["Geometry"]

        cD = geom[:, 1:]
        print(cD)
        pD = geom[:, 0]
        # tools.print_cartesians_pos_carts(pD, cD)
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


def test_C6s_change_dimer_to_monomer_IE_2():
    """
    Checks if IE changes between selecting monomer or dimer C6s
    """
    params = src.paramsTable.paramsDict()["HF"]
    df = pd.read_pickle(data_pkl)
    id_list = [515, 0, 2600, 4000, 4926, 7000]
    print(id_list)
    for n, i in enumerate(id_list):
        print(i)
        row = df.iloc[i]
        ma = row["monAs"]
        mb = row["monBs"]
        charges = row["charges"]
        geom_bohr = row["Geometry_bohr"]
        C6s_dimer = row["C6s"]
        C6s_mA = row["C6_A"]
        C6s_mB = row["C6_B"]

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
        diff = d4_dimer - d4_mons_individually
        print(diff)
        assert abs(d4_mons_individually - d4_dimer) > 1e-12
