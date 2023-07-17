import pytest
import src
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools
import pandas as pd
from psi4.driver.wrapper_database import database
import psi4

# psi4.set_options({"basis": "STO-3g"})
# v = database("HF", "HBC6", subset='small')
# print(f"{v = }")

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
print(f"{ang_to_bohr_qcel = }")
ang_to_bohr = src.constants.Constants().g_aatoau()
print(f"{ang_to_bohr = }")
print(f"difference: {ang_to_bohr_qcel - ang_to_bohr}")

hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")


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


# You will need to build https://github.com/Awallace3/dftd4 for pytest to pass
dftd4_bin = "/theoryfs2/ds/amwalla3/.local/bin/dftd4"


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


def test_d4_energies_damped() -> None:
    """
    compares dftd4 dispersion energies with damping parameters of HF
        HF params = [1.0, 1.61679827, 0.44959224, 3.35743605]
                  = [s6,  s8,         a1,         a2        ]
    """

    params = [1, 1.61679827, 0.44959224, 3.35743605]
    pos, carts = src.water_data.water_geom()
    charges = [0, 1]
    d4C6s, d4C8s, pairs, d4e = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges, dftd4_bin=dftd4_bin, p=params
    )

    params.pop(0)  # compute_bj_f90 doesn't take in s6

    cs = ang_to_bohr * np.array(carts, copy=True)
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(pos, cs)

    # l_e = src.locald4.compute_bj_f90(pos, carts, C6s, params=params)
    l_e = src.locald4.compute_bj_f90(pos, cs, d4C6s, params=params)

    print(f"{d4e = }\n{l_e = }")
    assert abs(d4e - l_e) < 1e-14


def test_d4_energies_undamped() -> None:
    """
    compares dftd4 dispersion energies with damping parameters of HF
        HF params = [1.0, 1.0, 0.0, 0.0]
                  = [s6,  s8,  a1,  a2 ]
    """

    params = [1.0, 1.0, 0.0, 0.0]
    pos, carts = src.water_data.water_geom()
    charges = [0, 1]
    d4C6s, d4C8s, pairs, d4e = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges, dftd4_bin=dftd4_bin, p=params
    )

    params.pop(0)  # compute_bj_f90 doesn't take in s6

    cs = ang_to_bohr * np.array(carts, copy=True)
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(pos, cs)

    # l_e = src.locald4.compute_bj_f90(pos, carts, C6s, params=params)
    l_e = src.locald4.compute_bj_f90(pos, cs, d4C6s, params=params)

    print(f"{d4e = }\n{l_e = }")
    assert abs(d4e - l_e) < 1e-14


def test_d4_energies_damped() -> None:
    """
    compares dftd4 dispersion energies with damping parameters of HF
        HF params = [1.0, 1.61679827, 0.44959224, 3.35743605]
                  = [s6,  s8,         a1,         a2        ]
    """

    params = [1, 1.61679827, 0.44959224, 3.35743605]
    pos, carts = src.water_data.water_geom()
    charges = [0, 1]
    d4C6s, d4C8s, pairs, d4e = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges, dftd4_bin=dftd4_bin, p=params
    )

    params.pop(0)  # compute_bj_f90 doesn't take in s6

    cs = ang_to_bohr * np.array(carts, copy=True)
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(pos, cs)

    # l_e = src.locald4.compute_bj_f90(pos, carts, C6s, params=params)
    l_e = src.locald4.compute_bj_f90(pos, cs, d4C6s, params=params)

    print(f"{d4e = }\n{l_e = }")
    assert abs(d4e - l_e) < 1e-14


def test_dispersion_interaction_energy() -> None:
    """
    test_dispersion_interaction_energy
    """
    # TODO: make dftd4 api use correct params
    params = [1, 1.61679827, 0.44959224, 3.35743605]
    p = params.copy()
    p.pop(0)

    num, coords = src.water_data.water_geom()
    tools.print_cartesians_pos_carts(num, coords)
    coords = ang_to_bohr * np.array(coords, copy=True)
    charges = [0, 1]

    # C6s, a, cn, pcharges = src.locald4.dftd4_api(num, coords)
    d4C6s, d4C8s, pairs, d4_e_d = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        num, coords / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )

    e_d = src.locald4.compute_bj_f90(num, coords, d4C6s, params=p)

    n1, p1 = num[:3], coords[:3, :]
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(n1, p1)
    d4C6s, d4C8s, pairs, d4_e_1 = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        n1, p1 / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )
    e_1 = src.locald4.compute_bj_f90(n1, p1, d4C6s, params=p)

    n2, p2 = num[3:], coords[3:, :]
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(n2, p2)
    d4C6s, d4C8s, pairs, d4_e_2 = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        n2, p2 / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )
    e_2 = src.locald4.compute_bj_f90(n2, p2, d4C6s, params=p)

    e_total = e_d - (e_1 + e_2)
    d4_e_total = d4_e_d - (d4_e_1 + d4_e_2)
    print(f"{e_d = }")
    print(f"{e_1 = }")
    print(f"{e_2 = }")
    print(f"{e_total = }")
    print(f"{d4_e_d = }")
    print(f"{d4_e_1 = }")
    print(f"{d4_e_2 = }")
    print(f"{d4_e_total = }")
    assert abs(d4_e_total - e_total) < 1e-14


def test_dispersion_interaction_energy2() -> None:
    """
    test_dispersion_interaction_energy
    """
    # TODO: make dftd4 api use correct params
    params = [1, 1.61679827, 0.44959224, 3.35743605]
    p = params.copy()
    p.pop(0)

    num, coords = src.water_data.water_geom2()
    tools.print_cartesians_pos_carts(num, coords)
    coords = ang_to_bohr * np.array(coords, copy=True)
    charges = [0, 1]

    # C6s, a, cn, pcharges = src.locald4.dftd4_api(num, coords)
    d4C6s, d4C8s, pairs, d4_e_d = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        num, coords / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )

    e_d = src.locald4.compute_bj_f90(num, coords, d4C6s, params=p)

    n1, p1 = num[:3], coords[:3, :]
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(n1, p1)
    d4C6s, d4C8s, pairs, d4_e_1 = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        n1, p1 / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )
    e_1 = src.locald4.compute_bj_f90(n1, p1, d4C6s, params=p)

    n2, p2 = num[3:], coords[3:, :]
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(n2, p2)
    d4C6s, d4C8s, pairs, d4_e_2 = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        n2, p2 / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )
    e_2 = src.locald4.compute_bj_f90(n2, p2, d4C6s, params=p)

    e_total = e_d - (e_1 + e_2)
    d4_e_total = d4_e_d - (d4_e_1 + d4_e_2)
    print(f"{e_d = }")
    print(f"{e_1 = }")
    print(f"{e_2 = }")
    print(f"{e_total = }")
    print(f"{d4_e_d = }")
    print(f"{d4_e_1 = }")
    print(f"{d4_e_2 = }")
    print(f"{d4_e_total = }")
    assert abs(d4_e_total - e_total) < 1e-14


def test_stored_C6s():
    df = pd.read_pickle("data/d4.pkl")
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
    # TODO: make dftd4 api use correct params
    params = [1, 1.61679827, 0.44959224, 3.35743605]
    p = params.copy()
    p.pop(0)
    df = pd.read_pickle("data/d4.pkl")
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
    df = pd.read_pickle("data/d4.pkl")
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


def test_charged_dftd4():
    """
    Ensures energy computed through dftd4 is different for neutral and charged
    """
    pos, carts = src.water_data.water_geom()
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


def test_charged_C6s_in_df():
    """
    Ensures that the C6's pre-computed in df are correctly charged
    """
    params = src.paramsTable.paramsDict()["HF"]
    df = pd.read_pickle("data/d4.pkl")
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
    df = pd.read_pickle("data/d4.pkl")
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


"""
"""
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

    print("geom")
    tools.print_cartesians_pos_carts(pos, carts)

    cs = ang_to_bohr * np.array(carts, copy=True)
    print(params)
    params.append(1.0)
    compute_ATM = src.locald4.compute_bj_f90_ATM(pos, cs, d4C6s_ATM, params=params)

    print(f"{target_ATM= }\n{compute_ATM = }")
    assert abs(target_ATM - compute_ATM) < 1e-14


def test_compute_bj_dimer_f90_ATM():
    """
    Tests if the fortran DFTD4 ATM energy is the same as the python version
    """
    # TODO
    df = pd.read_pickle("data/d4.pkl")
    id_list = [0, 500, 2700, 4926]
    params = src.paramsTable.paramsDict()["HF"]
    params.extend(1.0)
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


def test_C6s_change_dimer_to_monomer():
    """
    Checks if the C6's change when a dimer is changed to a monomer
    AND ensures that C6's stored in dataframe are correct for
    each monomer
    """
    params = src.paramsTable.paramsDict()["HF"]
    df = pd.read_pickle("data/d4.pkl")
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
    df = pd.read_pickle("data/d4.pkl")
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


def test_C6s_change_dimer_to_monomer_HBC6_IE():
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


def test_C6s_change_dimer_to_monomer_HBC6_IE_func_call():
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


def test_C6s_change_dimer_to_monomer_IE():
    """
    Checks if IE changes between selecting monomer or dimer C6s
    """
    params = src.paramsTable.paramsDict()["HF"]
    df = pd.read_pickle("data/d4.pkl")
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
        assert abs(d4_mons_individually - d4_dimer) < 1e-12


def test_water_dftd4_2_body_and_ATM():
    params = src.paramsTable.paramsDict()["pbe"]
    print(params)
    df = pd.read_pickle("data/d4.pkl")
    row = df.iloc[3014]
    charges = row["charges"]
    geom = row["Geometry"]
    ma = row['monAs']
    mb = row['monBs']
    pos, carts = geom[:, 0], geom[:, 1:]
    d4C6s, d4C8s, pairs, d4e_dimer = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_dimer = }")
    d4C6s, d4C8s, pairs, d4e_monA = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[ma], carts[ma], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monA = }")
    d4C6s, d4C8s, pairs, d4e_monB = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[mb], carts[mb], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monB = }")
    IE = d4e_dimer - d4e_monA - d4e_monB
    # print(f"{IE = }")
    ed4_2_body_IE = src.locald4.compute_bj_dimer_f90(params, row)
    ed4_2_body_IE /= hartree_to_kcalmol
    print(f"{ed4_2_body_IE = }")
    d4C6s, d4C8s, pairs, d4e_dimer_ATM = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges[0], dftd4_bin=dftd4_bin, p=params, s9=1.0
    )
    d4C6s, d4C8s, pairs, d4e_monA_ATM = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[ma], carts[ma], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monA_ATM = }")
    d4C6s, d4C8s, pairs, d4e_monB_ATM = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[mb], carts[mb], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monB_ATM = }")
    IE_ATM = d4e_dimer_ATM - d4e_monA_ATM - d4e_monB_ATM
    print(f"{IE_ATM = }")
    print(f"{d4e_dimer_ATM = }")
    print(f"{d4e_dimer_ATM - d4e_dimer = }")
    IE_diff_ATM_2_body = IE_ATM - ed4_2_body_IE
    print(f"{IE_diff_ATM_2_body = }")

    assert False


"""
/theoryfs2/ds/amwalla3/.local/bin/dftd4 dat.xyz --property --param 1.0 1.61679827 0.44959224 3.35743605 --mbdscale 0.0 -c [0 1] --pair-resolved
"""
