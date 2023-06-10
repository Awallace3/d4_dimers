import pytest
import src
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools
import pandas as pd

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
print(f"{ang_to_bohr_qcel = }")
ang_to_bohr = src.constants.Constants().g_aatoau()
print(f"{ang_to_bohr = }")
print(f"difference: {ang_to_bohr_qcel - ang_to_bohr}")

hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

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
    df = pd.read_pickle("data/d4.pkl")
    id_list = [0, 500, 2700, 4992]
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
