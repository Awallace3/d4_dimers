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
    gather_data3_dimer_splits,
)
from src.structs import d4_values, mol_i, mols_data
import subprocess
import json
import psi4


def compute_int_energy(params: [float], df: pd.DataFrame):
    """
    compute_int_energy is used to optimize paramaters for damping function in dftd4
    """
    rmse = 0
    diff = np.zeros(len(df))
    el_dc = create_pt_dict()
    df["d4"] = df.apply(
        lambda row: compute_bj_opt(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["C6s"],
            row["C8s"],
            row["monAs"],
            row["monBs"],
        ),
        axis=1,
    )
    df["diff"] = df.apply(
        lambda r: r["Benchmark"] - (r["HF INTERACTION ENERGY"] + r["d4"]), axis=1
    )
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params)
    df["diff"] = 0
    return rmse


"""
   polarizabilites are dependent on basis-set ** ask later
   used aug-QZ
   need to figure out what using with TD-DFT
"""


def optimization(df):
    # &  s6=1.0000_wp, s8=3.02227550_wp, a1=0.47396846_wp, a2=4.49845309_wp )
    params = [3.02227550, 0.47396846, 4.49845309]
    print("RMSE\t\tparams")
    ret = opt.minimize(compute_int_energy, args=(df), x0=params, method="powell")
    print("\nResults\n", ret)
    return


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # gather_data3(output_path="opt5.pkl")
    df = pd.read_pickle("opt5.pkl")
    optimization(df)
    return


if __name__ == "__main__":
    main()
