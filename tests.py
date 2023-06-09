import pytest
import src
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools

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

    # l_e = src.locald4.compute_bj_f90_eff(pos, carts, C6s, params=params)
    l_e = src.locald4.compute_bj_f90_eff(pos, cs, d4C6s, params=params)

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

    # l_e = src.locald4.compute_bj_f90_eff(pos, carts, C6s, params=params)
    l_e = src.locald4.compute_bj_f90_eff(pos, cs, d4C6s, params=params)

    print(f"{d4e = }\n{l_e = }")
    assert abs(d4e - l_e) < 1e-14


def test_d4_energies_damped_simplified() -> None:
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

    # l_e = src.locald4.compute_bj_f90_eff(pos, carts, C6s, params=params)
    l_e = src.locald4.compute_bj_f90_simplified(pos, cs, d4C6s, params=params)

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

    e_d = src.locald4.compute_bj_f90_simplified(num, coords, d4C6s, params=p)

    n1, p1 = num[:3], coords[:3, :]
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(n1, p1)
    d4C6s, d4C8s, pairs, d4_e_1 = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        n1, p1 / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )
    e_1 = src.locald4.compute_bj_f90_simplified(n1, p1, d4C6s, params=p)

    n2, p2 = num[3:], coords[3:, :]
    # C6s, a, cn, pcharges = src.locald4.dftd4_api(n2, p2)
    d4C6s, d4C8s, pairs, d4_e_2 = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        n2, p2 / ang_to_bohr, charges, dftd4_bin=dftd4_bin, p=params
    )
    e_2 = src.locald4.compute_bj_f90_simplified(n2, p2, d4C6s, params=p)

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
