import pytest
from src.tools import print_cartesians, print_cartesians_pos_carts, convert_geom_to_bohr
from src.constants import Constants
from src.setup import (
    calc_dftd4_props,
    compute_bj_f90,
    compute_bj_pairs,
    calc_dftd4_props_params,
    create_mon_geom,
    calc_dftd4_pair_resolved,
    compute_pairwise_dispersion,

)
import numpy as np
import math
from src.r4r2 import r4r2_vals
import json
from src.constants import Constants
from src.setup import (
    convert_str_carts_np_carts,
    write_xyz_from_np,
    calc_dftd4_props,
    gather_data3_dimer_splits,
    compute_bj_from_dimer_AB,
    compute_bj_from_dimer_AB_with_C6s,
    compute_bj_from_dimer_AB_all_C6s,
)
import subprocess
import os
import pandas as pd
from qcelemental import constants
from src.optimization import compute_int_energy_stats_dftd4_key


def carts1():
    return """
7    -0.2143860000000000    -1.9930209999999999    0.0001580000000000
6    -1.4834579999999999    -1.5376490000000000    0.0000390000000000
7    -1.8969870000000000    -0.2697650000000000    -0.0001260000000000
6    -0.8590920000000000    0.5833320000000000    -0.0001300000000000
6    0.5023510000000000    0.2582240000000000    0.0000250000000000
6    0.8189870000000000    -1.1204149999999999    0.0001690000000000
7    1.2930250000000001    1.3953620000000000    -0.0000180000000000
6    0.4308840000000000    2.3855279999999999    -0.0001910000000000
7    -0.8846300000000000    1.9602990000000000    -0.0002680000000000
1    -2.2473350000000001    -2.3170039999999998    0.0000550000000000
7    2.0781529999999999    -1.5902790000000000    0.0003130000000000
1    2.8393370000000000    -0.9285350000000000    0.0003540000000000
1    2.2576250000000000    -2.5992259999999998    0.0005070000000000
1    0.6864280000000000    3.4404690000000002    -0.0002720000000000
1    -1.7150930000000000    2.5358360000000002    -0.0004080000000000
7    0.9890610000000000    0.7972580000000000    3.4000580000000000
6    1.2176640000000001    -0.5930370000000000    3.4000040000000000
7    0.1323910000000000    -1.4089719999999999    3.3999649999999999
6    -1.1109620000000000    -0.9038440000000000    3.3999709999999999
6    -1.3524490000000000    0.5216270000000000    3.4000230000000000
6    -0.2635560000000000    1.3337810000000001    3.4000629999999998
1    1.8151999999999999    1.3809540000000000    3.4000919999999999
8    2.3888159999999998    -0.9949650000000000    3.3999910000000000
7    -2.1304249999999998    -1.7687059999999999    3.3999290000000002
1    -1.9560750000000000    -2.7965740000000001    3.3998710000000001
1    -3.0781360000000002    -1.4251389999999999    3.3999280000000001
1    -2.3598170000000001    0.9292250000000000    3.4000319999999999
1    -0.3321890000000000    2.4208530000000001    3.4001060000000001
    """


def carts2() -> str:
    """
    carts2
    """
    return """
6       -1.0617092040   1.2971405720    0.2920600030
8       -0.3581611160   2.2704586130    0.5318126680
8       -0.5893035160   0.0949177580    0.0037888130
6       -2.5584277980   1.3425498230    0.2962573200
1       0.4044356590    0.1277226210    0.0184118380
1       -2.8959979780   2.3474640020    0.5183163400
1       -2.9328892780   1.0223904510    -0.6729955510
1       -2.9372119600   0.6449104330    1.0395570840
6       2.2924117880    1.1062202300    0.2673807780
8       1.5887934230    0.1329057420    0.0275931760
8       1.8199854540    2.3086556180    0.5551592490
6       3.7891242360    1.0603174840    0.2654163830
1       0.8261969140    2.2757573840    0.5407587450
1       4.1267038000    0.0589982910    0.0278908880
1       4.1706227810    1.7706704230    -0.4640525150
1       4.1606405470    1.3630120010    1.2414717440
"""


# def test_dftd4_commandline_C6s():
#     test_carts = carts1()
#     c = convert_str_carts_np_carts(test_carts)
#     write_xyz_from_np(c[:, 0], c[:, 1:], outfile="test.xyz")
#     C6s, C8s = calc_dftd4_props(
#         c[:, 0],
#         c[:, 1:],
#         input_xyz="test.xyz",
#         output_json="test.json",
#     )
#     print(C6s)
#
#     with open("test.json") as f:
#         C6s_json = json.load(f)["c6 coefficients"]
#         print(C6s_json)
#     C6s_json = np.array(C6s_json).reshape((len(C6s), len(C6s)))
#     print(C6s[0], C6s_json[0])
#     t = np.abs(np.subtract(C6s, C6s_json))
#     # os.remove("test.xyz")
#     # os.remove("test.json")
#     # os.remove("C_n.json")
#     assert np.all(t < 1e-13)


def compute_bj_f90_test_setup():
    carts = convert_str_carts_np_carts(carts1())
    params = [1.61679827, 0.44959224, 3.35743605]
    pos = carts[:, 0]
    carts = carts[:, 1:]
    Ma = 15
    Mb = 13
    C6s, C8s = calc_dftd4_props(pos, carts)
    return params, pos, carts, Ma, Mb, C6s, C8s


def read_dftd4_vals():
    with open("vals_test.json") as f:
        d = json.load(f)
    for k, v in d.items():
        d[k] = np.array(v)
    return d


def compute_bj_f90_pieces():
    params, pos, carts, Ma, Mb, C6s, C8s = compute_bj_f90_test_setup()
    rrijs, r2s, t6s, t8s, edisps, des = [], [], [], [], [], []

    electron_mass = 9.1093837015e-31
    c = 299792458
    fine_structure_constant = 7.2973525693e-3
    hbar = 6.62607015e-34 / (2 * math.pi)
    bohr = hbar / (electron_mass * c * fine_structure_constant)
    autoaa = bohr * 1e10
    aatoau = 1 / autoaa

    energy = 0
    s8, a1, a2 = params
    s6 = 1.0
    C8s = np.zeros(np.shape(C6s))
    N_tot = len(carts)
    energies = np.zeros(Ma + Mb)
    M_tot = len(carts)
    lattice_points = 1

    cs = aatoau * np.array(carts, copy=True)
    for i in range(M_tot):
        el1 = int(pos[i])
        el1_r4r2 = r4r2_vals(el1)
        Q_A = np.sqrt(np.sqrt(el1) * el1_r4r2)

        for j in range(i):
            if i == j:
                continue
            for k in range(lattice_points):
                el2 = int(pos[j])
                el2_r4r2 = r4r2_vals(el2)
                Q_B = np.sqrt(np.sqrt(el2) * el2_r4r2)

                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6 = C6s[i, j]
                r1, r2 = cs[i, :], cs[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))
                # R value is not a match since dftd4 converts input carts
                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)
                edisp = s6 * t6 + s8 * rrij * t8
                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6 * edisp * 0.5
                energies[i] += de
                if i != j:
                    energies[j] += de
                rrijs.append(rrij)
                r2s.append(r2)
                t6s.append(t6)
                t8s.append(t8)
                edisps.append(edisp)
                des.append(de)

    # print(len(rrijs))
    # print(M_tot * (M_tot - 1) / 2)
    energy = np.sum(energies)
    return rrijs, r2s, t6s, t8s, edisps, des, energies, energy


# rrijs, r2s, t6s, t8s, edisps, des, energies, energy = compute_bj_f90_pieces()
# d = read_dftd4_vals()
# rrijs_t, r2s_t, t6s_t, t8s_t, edisps_t, des_t, energies_t = (
#     d["rrijs"],
#     d["r2s"],
#     d["t6s"],
#     d["t8s"],
#     d["edisps"],
#     d["des"],
#     d["energy"],
# )
# @pytest.mark.parametrize("rrijs", rrijs)
# @pytest.mark.parametrize("rrijs_t", rrijs_t)
# def test_bj_rrij():
#     """
#     dispersion2
#     _t are reported up to 6 decimal places
#     """
#     t = np.abs(np.subtract(rrijs, rrijs_t))
#     assert np.all(t < 1e-6)
#
#
# def test_bj_r2s():
#     """
#     dispersion2
#     _t are reported up to 6 decimal places
#     """
#     t = np.abs(np.subtract(r2s, r2s_t))
#     assert np.all(t < 1e-5)
#
#
# def test_bj_t6s():
#     """
#     dispersion2
#     _t are reported up to 16 decimal places
#     """
#     t = np.abs(np.subtract(t6s, t6s_t))
#     assert np.all(t < 1e-16)
#
#
# def test_bj_t8s():
#     """
#     dispersion2
#     _t are reported up to 16 decimal places
#     """
#     t = np.abs(np.subtract(t8s, t8s_t))
#     assert np.all(t < 1e-16)
#
#
# def test_bj_edisps():
#     """
#     dispersion2
#     _t are reported up to 16 decimal places
#     """
#     t = np.abs(np.subtract(edisps, edisps_t))
#     assert np.all(t < 1e-16)
#
#
# def test_bj_des():
#     """
#     dispersion2
#     _t are reported up to 16 decimal places
#     """
#     t = np.abs(np.subtract(des, des_t))
#     assert np.all(t < 1e-16)
#
#
# def test_bj_energies():
#     """
#     dispersion2
#     _t are reported up to 16 decimal places
#     """
#     t = np.abs(np.subtract(energies, energies_t))
#     assert np.all(t < 1e-16)
#
#
# def test_bj_energies():
#     """
#     dispersion2
#     _t are reported up to 16 decimal places
#     """
#     e = np.sum(energies)
#     e_t = np.sum(energies_t)
#     t = np.abs(e - e_t)
#     assert t < 1e-16


def compute_parameters():
    # rrijs, r2s, t6s, t8s, edisps, des, energies, energy = compute_bj_f90_pieces()
    # d = read_dftd4_vals()
    # rrijs_t, r2s_t, t6s_t, t8s_t, edisps_t, des_t = (
    #     d["rrijs"],
    #     d["r2s"],
    #     d["t6s"],
    #     d["t8s"],
    #     d["edisps"],
    #     d["des"],
    # )
    # test_bj_rrij(rrijs, rrijs_t)
    # print(edisps[16], edisps_t[16], np.subtract(edisps[16], edisps_t[16]))
    t = np.zeros((len(r2s), 3))
    t[:, 0] = r2s
    t[:, 1] = r2s_t
    t[:, 2] = t[:, 0] - t[:, 1]
    for i in t:
        print(i)

    return


def test_Hs_splits_size_match_false():
    df = pd.read_pickle("opt_test.pkl")
    df1 = df.reset_index(drop=True)
    df2, ind = gather_data3_dimer_splits(df1)
    correct = [False for i in ind]
    print(correct)
    for n, i in enumerate(ind):
        c1 = df1.loc[i, "Geometry"]
        c2 = df2.loc[i, "Geometry"][:-2, :]
        mol = [False for i in c1]
        for n1, r1 in enumerate(c1):
            for n2, r2 in enumerate(c2):
                if np.array_equal(r1, r2):
                    mol[n1] = True
                    break
        if sum(mol) == len(mol):
            print(n, correct)
            correct[n] = True
    assert sum(correct) != len(correct)


def test_Hs_splits_size_match_true():
    df = pd.read_pickle("opt_test.pkl")
    df1 = df.reset_index(drop=True)
    df2, ind = gather_data3_dimer_splits(df1)
    correct = [False for i in ind]
    for n, i in enumerate(ind):
        c1 = df1.loc[i, "Geometry"]
        c2 = df2.loc[i, "Geometry"]
        mol = [False for i in c1]
        for n1, r1 in enumerate(c1):
            for n2, r2 in enumerate(c2):
                if np.array_equal(r1, r2):
                    mol[n1] = True
                    break
        if sum(mol) == len(mol):
            print(n, correct)
            correct[n] = True
    assert sum(correct) == len(correct) and sum(correct) > 0


def compare_opt_jeff() -> None:
    """
    compare_opt_jeff ...
    """
    df2 = pd.read_pickle("opt.pkl")
    ms = pd.read_pickle("jeff.pkl")
    df2 = df2.sort_values(by="Benchmark")
    ms = ms.sort_values(by="Benchmark")
    g1 = df2["Geometry"].to_list()
    g3 = ms["Geometry"].to_list()
    b1 = df2["Benchmark"].to_list()
    b3 = ms["Benchmark"].to_list()
    g = 0
    b = 0
    for i in range(len(g1)):
        if np.array_equal(g1[i], g3[i]):
            g += 1
        if b1[i] == b2[i]:
            b += 1
    assert g == b == len(g1)



def sum_ma_mb(
    ma,
    mb,
    m: np.array,
) -> float:
    """
    sum_ma_mb
    """
    e = 0
    for i in ma:
        for j in mb:
            e += m[i, j]
    return e


def create_test_opt_t_dm() -> None:
    """
    create_test_opt_t_dm produces data for test_dm
    """
    df = pd.read_pickle("opt4.pkl")
    test_df = df.loc[[1466, 7265]]
    test_df["psi4_IE"] = [14.829109041411, 27.231587582085]
    return test_df


def create_test_opt_t_dm2() -> None:
    """
    create_test_opt_t_dm2 produces data for test_dm
    """
    df = pd.read_pickle("opt5.pkl")
    test_df = df.loc[[1466, 7265]]
    test_df["psi4_IE"] = [14.829109041411, 27.231587582085]
    return test_df


# TODO: analyze monomers for dftd4

# def test_dm_vs_dftd4_disp():
#     df = create_test_opt_t_dm()
#     params=[1.61679827, 0.44959224, 3.35743605]
#     df["d4_dm_C6s"] = df.apply(
#         lambda row: compute_bj_from_dimer_AB_with_C6s(
#             params,
#             row["Geometry"][:, 0],  # pos
#             row["Geometry"][:, 1:],  # carts
#             row["monAs"],
#             row["monBs"],
#             row["C6s"],
#             mult_out=1.0,
#         ),
#         axis=1,
#     )
#     df["d4_dm"] = df.apply(
#         lambda row: compute_bj_from_dimer_AB(
#             params,
#             row["Geometry"][:, 0],  # pos
#             row["Geometry"][:, 1:],  # carts
#             row["monAs"],
#             row["monBs"],
#             row["C6s"],
#             mult_out=1.0,
#         ),
#         axis=1,
#     )
#     df["d4_dm_diff_psi4_DISP_dftd4"] = (df["d4_dm_C6s"] - df["psi4_DISP"])
#     df["d4_dm_diff_psi4_DISP_self"] = (df["d4_dm"] - df["psi4_DISP"])
#
#     print(df["d4_dm_C6s"])
#     print(df["d4_dm_diff_psi4_DISP_dftd4"])
#     print(df["psi4_DISP"])
#     print(df["d4_dm_diff_psi4_DISP_self"])
#     assert (df["d4_dm_diff_psi4_DISP_self"] == df["d4_dm_diff_psi4_DISP_dftd4"]).all()


# def test_dm_vs_dftd4():
#     df = create_test_opt_t_dm()
#     params = [1.61679827, 0.44959224, 3.35743605]
#     df["d4_dm_C6s"] = df.apply(
#         lambda row: compute_bj_from_dimer_AB_with_C6s(
#             params,
#             row["Geometry"][:, 0],  # pos
#             row["Geometry"][:, 1:],  # carts
#             row["monAs"],
#             row["monBs"],
#             row["C6s"],
#             mult_out=1.0,
#         ),
#         axis=1,
#     )
#     df["d4_dm"] = df.apply(
#         lambda row: compute_bj_from_dimer_AB(
#             params,
#             row["Geometry"][:, 0],  # pos
#             row["Geometry"][:, 1:],  # carts
#             row["monAs"],
#             row["monBs"],
#             row["C6s"],
#             mult_out=1.0,
#         ),
#         axis=1,
#     )
#
#     print(df["d4_dm_C6s"])
# assert (df["d4_dm_diff_psi4_DISP_self"] - df["d4_dm_diff_psi4_DISP_dftd4"] < 1).all()


def test_dm_all_C6s():
    df = create_test_opt_t_dm2()
    params = [1.61679827, 0.44959224, 3.35743605]
    df["d4_dm_C6s"] = df.apply(
        lambda row: compute_bj_from_dimer_AB_all_C6s(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            C6_A=row["C6_A"],
            C6_B=row["C6_B"],
        ),
        axis=1,
    )
    df["IE_d4_all_C6s"] = df["d4_dm_C6s"] + df["HF_jdz"]
    df["IE_diff"] = df["IE_d4_all_C6s"] - df["psi4_IE"]
    print(df[["IE_d4_all_C6s", "psi4_IE", "IE_diff"]])
    assert (df["IE_diff"].abs() < 1e-4).all()


def ttest_pairs() -> None:
    """
    test_pairs
    """
    df = create_test_opt_t_dm2()
    params = [1.61679827, 0.44959224, 3.35743605]
    df["d4_dm_pairs"] = df.apply(
        lambda row: compute_bj_pairs(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            mult_out=constants.conversion_factor("hartree", "kcal / mol"),
        ),
        axis=1,
    )
    df["IE_pairs"] = df["HF_jdz"] + df["d4_dm_pairs"]
    print(df[["IE_pairs", "psi4_IE"]])
    assert ((df["IE_pairs"] - df["psi4_IE"]) < 1e-4).all()


def version_tests():
    """
    docstring
    """
    df2 = create_test_opt_t_dm2()
    params = [1.61679827, 0.44959224, 3.35743605]
    df2["d4_dm_C6s"] = df2.apply(
        lambda row: compute_bj_from_dimer_AB_all_C6s(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            C6_A=row["C6_A"],
            C6_B=row["C6_B"],
        ),
        axis=1,
    )
    df2["IE_d4_all_C6s"] = df2["d4_dm_C6s"] + df2["HF_jdz"]
    df2["IE_diff"] = df2["IE_d4_all_C6s"] - df2["psi4_IE"]

    df = create_test_opt_t_dm()
    params = [1.61679827, 0.44959224, 3.35743605]
    df["d4_dm_C6s"] = df.apply(
        lambda row: compute_bj_from_dimer_AB_with_C6s(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            mult_out=1.0,
        ),
        axis=1,
    )
    df["d4_dm"] = df.apply(
        lambda row: compute_bj_from_dimer_AB(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            mult_out=1.0,
        ),
        axis=1,
    )
    df["d4_dm_pairs"] = df.apply(
        lambda row: compute_bj_pairs(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            mult_out=1.0,
        ),
        axis=1,
    )
    df["IE_c6s"] = df["HF_jdz"] + df["d4_dm_C6s"] * 627.509
    df["IE_dm"] = df["HF_jdz"] + df["d4_dm"] * 627.509
    df["IE_pairs"] = df["HF_jdz"] + df["d4_dm_pairs"] * 627.509
    print(df[["d4_dm_C6s", "d4_dm", "d4_dm_pairs"]])
    print(df[["IE_c6s", "IE_dm", "IE_pairs", "psi4_IE"]])
    print(df2[["IE_d4_all_C6s", "psi4_IE", "IE_diff"]])
    return


def calc_dftd4_disp_pieces(atoms, geom, ma, mb, params):
    x, y, d = calc_dftd4_props_params(atoms, geom, p=params)
    print(x[0][0])
    x, y, a = calc_dftd4_props_params(atoms[ma], geom[ma, :], p=params)
    x, y, b = calc_dftd4_props_params(atoms[mb], geom[mb, :], p=params)
    return d - (a + b)


def compute_difference_cbjp_vs_dftd4(ind, df):
    mol = df.iloc[ind]
    print(mol)
    params = [1.61679827, 0.44959224, 3.35743605]
    cbjp = compute_bj_from_dimer_AB_all_C6s(
        params,
        mol["Geometry"][:, 0],  # pos
        mol["Geometry"][:, 1:],  # carts
        mol["monAs"],
        mol["monBs"],
        mol["C6s"],
        mol["C6_A"],
        mol["C6_B"],
        mult_out=1.0,
    )
    e = calc_dftd4_disp_pieces(
        mol["Geometry"][:, 0],
        mol["Geometry"][:, 1:],
        mol["monAs"],
        mol["monBs"],
        params,
    )
    return cbjp, e


def test_dftd4_versions():
    m_id = 4756
    df = pd.read_pickle("tests/diffs.pkl")
    mol = df.iloc[m_id]
    atom_numbers = mol["Geometry"][:, 0]
    carts = mol["Geometry"][:, 1:]
    x, y, e1 = calc_dftd4_props_params(atom_numbers, carts, dftd4_p="dftd4")
    x, y, e2 = calc_dftd4_props_params(
        atom_numbers, carts, dftd4_p="/theoryfs2/ds/amwalla3/miniconda3/bin/dftd4"
    )
    h_to_kcal_mol = constants.conversion_factor("hartree", "kcal / mol")
    e1 *= h_to_kcal_mol
    e2 *= h_to_kcal_mol
    assert abs(e1 - e2) < 1e-4


def test_grimme_dftd4_psi4():
    df = pd.read_pickle("tests/diffs_grimme.pkl")
    compute_int_energy_stats_dftd4_key(df, hf_key="HF_jdz")
    df["HF_diff_abs"] = df["HF_diff"].abs()
    assert (df["HF_diff_abs"] < 1e-4).sum() == len(df)


# def test_dftd4_all():
#     """ """
#     df = pd.read_pickle("tests/diffs.pkl")
#     cs, es = [], []
#     for i in range(len(df)):
#         cbjp, e = compute_difference_cbjp_vs_dftd4(i, df)
#         cs.append(cbjp)
#         es.append(e)
#
#     h_to_kcal_mol = constants.conversion_factor("hartree", "kcal / mol")
#     cs = np.array(cs) * h_to_kcal_mol
#     es = np.array(es) * h_to_kcal_mol
#     diff = np.abs(np.subtract(cs, es))
#     assert np.all((diff < 1e-6))


def test_dftd4_0():
    """
    test difference between dftd4 and self C6s w/ damping on ind=0
    """
    df = pd.read_pickle("tests/diffs.pkl")
    cbjp, e = compute_difference_cbjp_vs_dftd4(0, df)

    h_to_kcal_mol = constants.conversion_factor("hartree", "kcal / mol")
    cbjp *= h_to_kcal_mol
    e *= h_to_kcal_mol
    print(cbjp, e)
    assert abs(cbjp - e) < 1e-8


def test_dftd4_SSI_5555():
    """
    test difference between dftd4 and self C6s w/ damping on ind=5555
    """
    df = pd.read_pickle("tests/diffs.pkl")
    cbjp, e = compute_difference_cbjp_vs_dftd4(5555, df)
    h_to_kcal_mol = constants.conversion_factor("hartree", "kcal / mol")
    cbjp *= h_to_kcal_mol
    e *= h_to_kcal_mol
    print(cbjp, e)
    assert abs(cbjp - e) < 1e-8


def test_dftd4_X10by20_4756():
    """
    test difference between dftd4 and self C6s w/ damping on ind=4756
    """
    df = pd.read_pickle("tests/diffs.pkl")
    cbjp, e = compute_difference_cbjp_vs_dftd4(4756, df)

    h_to_kcal_mol = constants.conversion_factor("hartree", "kcal / mol")
    cbjp *= h_to_kcal_mol
    e *= h_to_kcal_mol
    print(cbjp, e)
    assert abs(cbjp - e) < 1e-8


def test_dftd4_calc_psi4() -> None:
    """
    test_dftd4_calc_psi4
    params = [1.61679827, 0.44959224, 3.35743605]
    """
    df = pd.read_pickle("tests/diffs.pkl")
    df.dropna(subset=["HF_jdz_dftd4"], how="all", inplace=True)
    print(df.columns)
    df["diff"] = df.apply(lambda r: r["HF_jdz_d4_sum"] - r["HF_jdz_dftd4"], axis=1)

    v = (df["HF_diff"].abs() < 1e-1).all()
    print(sorted(df["HF_diff"].abs().to_list(), reverse=True)[:50])
    assert v




def main():
    # TODO: attempt pair-resolved dispersion for C6s
    df = pd.read_pickle("tests/diffs_grimme.pkl")
    m = df.iloc[1]
    *_, disp = compute_pairwise_dispersion(m)
    # pairs is tabulated in kcal/mol already
    # disp = 0
    # for a in m["monAs"]:
    #     for b in m["monBs"]:
    #         disp += pairs[a, b]
    print(disp, m["dftd4_disp_ie_grimme_params"], m["HF_qz"], m["Benchmark"])
    mult_out = constants.conversion_factor("hartree", "kcal / mol")
    # print(e * mult_out)
    return


if __name__ == "__main__":
    main()
