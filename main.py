import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
import pprint as pp
import pickle
from src.tools import write_pickle, read_pickle
from tqdm import tqdm
from src.setup import (
    construct_xyz_lookup,
    gather_data2,
    build_dummy,
    write_xyz_from_np,
    convert_str_carts_np_carts,
    create_data_csv,
    compute_bj_self,
    read_xyz,
    create_pt_dict,
    compute_bj_opt,
    gather_data3,
)
from src.structs import d4_values, mol_i, mols_data
import subprocess
import json
import psi4

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


# def compute_bj_dftd4(
#     params,
#     atom_numbers: np.array,
#     carts: np.array,
# ):
#     s8, a1, a2 = params
#     energy = 0.0
#     disp = DispersionModel(
#         numbers=atom_numbers,
#         positions=carts,
#     )
#     param = DampingParam(s6=1, s8=s8, a1=a1, a2=a2)
#     res = disp.get_dispersion(param, grad=False)
#     energy = res.get("energy")
#     energy *= -1.0 * 627.5095
#     return energy


def distance_3d(r1, r2):
    return np.linalg.norm(r1 - r2)


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


# def calc_dftd4_props(
#     atom_numbers: np.array,
#     carts: np.array,
# ):
#     disp = DispersionModel(
#         numbers=atom_numbers,
#         positions=carts,
#     )
#     # res = disp.get_pairwise_dispersion(DampingParam(method='tpss'))
#     props = disp.get_properties()
#     C6s = props.get("c6 coefficients")
#     Qs = props.get("partial charges")
#     return C6s, Qs


def read_carts_compute_bj(
    params: [float],
    # xyz: str,
    xyz: np.array,
    C6s: np.array,
    C8s: np.array,
    el_dc: dict,
) -> float:
    """
    read_carts_compute_bj reads xyz path to compute bj d4 energy
    """
    # geom = read_xyz(xyz, el_dc)
    # energy = compute_bj_opt(params, geom[:, 0], geom[:, 1:], C6s, C8s)
    energy = compute_bj_opt(params, xyz[:, 0], xyz[:, 1:], C6s, C8s)
    return energy


def compute_int_energy(params: [float], df: pd.DataFrame):
    """
    compute_int_energy is used to optimize paramaters for damping function in dftd4
    """
    rmse = 0
    diff = np.zeros(len(df))
    el_dc = create_pt_dict()
    # res = np.zeros(len(df))
    # for n, (idx, item) in enumerate(df.iterrows()):
    #     xyz = item["Geometry"]
    #     C6s = item["C6s"]
    #     C8s = item["C8s"]
    #     d4 = read_carts_compute_bj(abs(params), xyz, C6s, C8s, el_dc)
    #     dhf = item["HF INTERACTION ENERGY"]
    #     sdd = d4 + dhf
    #     ref = item["Benchmark"]
    #     res[n] = ref - sdd
    # rmse = np.sqrt(np.mean(np.square(res)))
    # print(rmse, params)
    # return rmse

    df["d4"] = df.apply(
        lambda row: read_carts_compute_bj(
            params,
            # row["xyz_d"],
            row["Geometry"],
            row["C6s"],
            row["C8s"],
            el_dc,
        ),
        axis=1,
    )
    df["diff"] = df.apply(
        lambda r: r["Benchmark"] - (r["HF INTERACTION ENERGY"] + r["d4"]), axis=1
    )
    print(df[["Benchmark", "HF INTERACTION ENERGY", "d4", "diff"]])

    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, df["d4"].mean(), params)
    df["diff"] = 0
    return rmse


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
"""


def optimization():
    params = [0.9171, 0.3385, 2.883]
    print("RMSE\t\tparams")
    df = pd.read_pickle("opt3.pkl")
    # df = df.iloc[[0, 1]]
    # df["HF INTERACTION ENERGY"] = -df["HF INTERACTION ENERGY"]
    ret = opt.minimize(compute_int_energy, args=(df), x0=params, method="powell")
    print("Results =", ret)
    return


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # create_data_csv()
    # gather_data2()
    gather_data3()
    optimization()

    # compute_int_energy(params)

    return


if __name__ == "__main__":
    main()
