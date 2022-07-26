import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
import dftd4
import pprint as pp
import pickle
from dftd4.interface import DispersionModel, DampingParam
from src.tools import write_pickle, read_pickle
from tqdm import tqdm
from src.setup import (
    construct_xyz_lookup,
    gather_data2,
    build_dummy,
    write_xyz_from_np,
    convert_str_carts_np_carts,
)
from src.structs import d4_values, mol_i, mols_data
import subprocess
import json

path_db = "data/Databases/"


def gather_data(
    pkl_path: str = "master-regen.pkl",
    csv_path: str = "SAPT-D3-Refit-Data.csv",
    out_pkl: str = "data.pkl",
):
    """
    gather_data collects pkl data
    """
    ms = pd.read_pickle(pkl_path)
    ms = ms[ms["DB"] != "PCONF"]
    ms = ms[ms["DB"] != "SCONF"]
    ms = ms[ms["DB"] != "ACONF"]
    ms = ms[ms["DB"] != "CYCONF"]

    df = pd.read_csv("SAPT-D3-Refit-Data.csv")
    out_df = ms[["Benchmark", "HF INTERACTION ENERGY", "Geometry"]]
    energies = ms[["Benchmark", "HF INTERACTION ENERGY"]].to_numpy()
    carts = ms["Geometry"].to_list()
    atom_order = [np.array(i[:, 0]) for i in carts]
    carts = [i[:, 1:] for i in carts]

    data = [energies, atom_order, carts]
    C6s, C8s, Qs = [], [], []
    for i in tqdm(range(len(data[0])), desc="DFTD4 Props", ascii=True):
        C6, Q = calc_dftd4_props(data[1][i], data[2][i])
        C6s.append(C6)
        Qs.append(Q)
    C6s = C6s
    Qs = Qs
    d4_vals = d4_values(C6s, Qs, C8s)
    data_out = mols_data(energies, atom_order, carts, d4_vals)
    write_pickle(data_out, out_pkl)
    return data_out


def compute_bj_dftd4(
    params,
    atom_numbers: np.array,
    carts: np.array,
):
    s8, a1, a2 = params
    energy = 0.0
    disp = DispersionModel(
        numbers=atom_numbers,
        positions=carts,
    )
    param = DampingParam(s6=1, s8=s8, a1=a1, a2=a2)
    res = disp.get_dispersion(param, grad=False)
    energy = res.get("energy")
    energy *= -1.0 * 627.5095
    return energy


def distance_3d(r1, r2):
    return np.linalg.norm(r1 - r2)


def compute_bj(params, mol: mol_i):
    s8, a1, a2 = params
    s6 = 1
    energy = 0.0
    Rs = np.zeros((len(mol.carts), len(mol.carts)))

    for n1, r1 in enumerate(mol.carts):
        for n2, r2 in enumerate(mol.carts):
            R = distance_3d(r1, r2)
            Rs[n1, n2] = R
    C8s = np.zeros(np.shape(mol.C6s))
    for i in range(len(mol.C6s)):
        for j in range(len(mol.C6s[i])):
            v = np.sqrt(mol.Qs[i] * mol.Qs[j])
            if not np.isnan(v):
                C8s[i, j] = 3 * mol.C6s[i, j] * v
    mol.C8s = C8s

    for i in range(len(mol.C6s)):
        for j in range(len(mol.C6s[i])):
            C6 = mol.C6s[i, j]
            C8 = mol.C8s[i, j]
            R = Rs[i, j]
            R_0 = np.sqrt(C6 * C8)
            energy += s6 * C6 / (R**6.0 + (a1 * R_0 + a2) ** 6.0)
            energy += s8 * C8 / (R**8.0 + (a1 * R_0 + a2) ** 8.0)

    # for pair in d3data:
    #     atom1, atom2, R, R0, C6, C8 = pair

    #     R0 = np.sqrt(C8 / C6)
    #     energy += C6 / (R**6.0 + (a1 * R0 + a2) ** 6.0)
    #     energy += s8 * C8 / (R**8.0 + (a1 * R0 + a2) ** 8.0)

    energy *= -1.0 * 627.5095
    return energy


def get_db(reffile):
    # Get the master database
    master = pd.read_pickle(reffile)

    master = master[master["DB"] != "PCONF"]
    master = master[master["DB"] != "SCONF"]
    master = master[master["DB"] != "ACONF"]
    master = master[master["DB"] != "CYCONF"]
    weights = []
    dref = []
    bm = []
    df = pd.read_csv("SAPT-D3-Refit-Data.csv")
    for idx, item in master.iterrows():
        db = item["DB"]
        sys = item["System"]
        r = item["R"]

        try:
            # ret = df.loc[(df['DB'] == db) & (df['System'] == sys) & (df['R'] == r), ['Benchmark', 'Weight', 'HF-CP-qzvp-BJ-F']].to_numpy()[0]
            ret = df.loc[
                (df["System"] == sys)
                & (abs(df["Benchmark"] - item["Benchmark"]) < 1e-6),
                ["Benchmark", "Weight", "HF-CP-qzvp-BJ-F", "HF-CP-qzvp-Zero-F"],
            ].to_numpy()[0]
        except:
            print(db, sys, r, item["Benchmark"])
            print("Exiting")
            exit()

        bm.append(ret[0])
        weights.append(ret[1])
        dref.append(ret[2])

    master["weights"] = weights
    master["dref"] = dref
    master["bm"] = bm

    # training = master[master['Fitset']==True]
    training = master
    ntrain = len(training)
    print(ntrain)
    print(training)
    return training


def calc_dftd4_props(
    atom_numbers: np.array,
    carts: np.array,
):
    disp = DispersionModel(
        numbers=atom_numbers,
        positions=carts,
    )
    # res = disp.get_pairwise_dispersion(DampingParam(method='tpss'))
    props = disp.get_properties()
    C6s = props.get("c6 coefficients")
    Qs = props.get("partial charges")
    return C6s, Qs


def compute_int_energy(params: [float], data: mols_data):
    """
    compute_int_energy is used to optimize paramaters for damping function in dftd4
    """
    rmse = 0
    n_mols = len(data.carts)
    for i in range(n_mols):
        mol = data.get_mol(i)
        mol.print_cartesians()
        print(len(mol.carts), len(mol.C6s), len(mol.C6s[0]))
        print(mol.C6s)
        en = compute_bj(params, mol)
        print(en)
        en = compute_bj_dftd4(params, data.atom_orders[i], data.carts[i])
        print(en)

    return 0


# @dataclass
# class mols_data:
#     energies: np.array
#     atom_orders: [np.array]
#     carts: [np.array]
#     d4_vals: d4_values


def shorten_data(data, start, stop):
    """
    shorten_data
    """
    data.energies = data.energies[start:stop]
    data.atom_orders = data.atom_orders[start:stop]
    data.carts = data.carts[start:stop]
    data.d4_vals.C6s = data.d4_vals.C6s[start:stop]
    data.d4_vals.Qs = data.d4_vals.Qs[start:stop]

    return data


def test_size(n=15):
    v = range(n)
    # v = [str(i) for i in range(28)]
    print(v)
    pairs = []
    for i in v:
        for j in v:
            if i > j:
                pairs.append("%d%d" % (i, j))
    print(pairs)
    print(len(v), len(pairs), n**2 / 2 - n / 2)


"""
   polarizabilites are dependent on basis-set ** ask later
   used aug-QZ
   need to figure out what using with TD-DFT

need N_a by N_b for indexing
1. split mon_A and mon_B
2. pick block C6 that is N_aXN_b
3. split charges into Q_A and Q_B
4. remove rest
5. make N_A by N_B distance matrix
       # carts distance

"""

# check units on C6 outputs, Energy/units
# try
def test(
    atom_numbers: np.array,
    carts: np.array,
):
    disp = DispersionModel(
        numbers=atom_numbers,
        positions=carts,
    )
    # res = disp.get_pairwise_dispersion(DampingParam(method='tpss'))
    # param = DampingParam(s6=s6, s8=s8, a1=a1, a2=a2)
    props = disp.get_properties()
    C6s = props.get("c6 coefficients")
    # print("P:", props.get("polarizibilities"))

    write_xyz_from_np(atom_numbers, carts)
    # print(atom_numbers, carts)
    subprocess.call(
        "dftd4 dat.xyz --mbdscale 0.0 --func hf --json t.json --property > setup.txt",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    with open("t.json") as f:
        C6s_json = json.load(f)["c6 coefficients"]
    C6s_json = np.array(C6s_json).reshape((len(C6s), len(C6s)))
    print(C6s[0])
    print(C6s_json[0])
    # print(props)
    return C6s_json


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # construct_xyz_lookup()
    # build_dummy()
    test_carts = """
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
    # test_carts = convert_str_carts_np_carts(test_carts)
    # test(
    #     # np.array([1.0, 1.0]),
    #     # np.array(
    #     #     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    #     # ),
    #     test_carts[:, 0],
    #     test_carts[:, 1:],
    # )

    gather_data2()

    # data = gather_data()
    # read_master_regen()
    # data = read_pickle("data.pkl")
    # data = shorten_data(data, 3368, 3369)
    # # print(len(data.energies), len(data.d4_vals.C6s))

    # # print(len(data.carts[0]))
    # # print(len(data.d4_vals.C6s[0]), len(data.d4_vals.C6s[0][0]))
    # # print(len(data.d4_vals.Qs[0]))
    # # print(data.carts[0])
    # # print(data.d4_vals.C6s[0])
    # # print(data.d4_vals.Qs[0])
    # # (s6=1.0, s8, s9=1.0, a1, a2, alp=16.0)
    # BJ_init = [0.9171, 0.3385, 2.883]
    # print(f"Damping: BJ")
    # print(f"Metric: RMSE")
    # compute_int_energy(BJ_init, data)
    # ret = opt.minimize(compute_int_energy, init, method='powell')

    return


if __name__ == "__main__":
    main()
