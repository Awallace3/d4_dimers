from .setup import (
    create_pt_dict,
    compute_bj_opt,
    gather_data3,
)
import scipy.optimize as opt

import pandas as pd
import numpy as np


def compute_int_energy_stats(
    params: [float], df: pd.DataFrame
) -> (float, float, float,):
    """
    compute_int_energy is used to optimize paramaters for damping function in dftd4
    """
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
    mae = df["diff"].abs().sum() / len(df["diff"])
    rmse = (df["diff"] ** 2).mean() ** 0.5
    max_e = df["diff"].max()
    return mae, rmse, max_e


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


def get_folds(nfold, ntrain):
    """
    computes labels for fitset
    """
    folds = []
    for f in range(nfold):
        f_def = []
        for n in range(ntrain):
            if (n % nfold) == f:
                f_def.append(False)
            else:
                f_def.append(True)
        folds.append(f_def)
    return folds


def optimization(
    df: pd.DataFrame,
    params: [] = [3.02227550, 0.47396846, 4.49845309],
):
    # &  s6=1.0000_wp, s8=3.02227550_wp, a1=0.47396846_wp, a2=4.49845309_wp )
    print("RMSE\t\tparams")
    ret = opt.minimize(compute_int_energy, args=(df), x0=params, method="powell")
    print("\nResults\n")
    out_params = ret.x
    mae, rmse, max_e = compute_int_energy_stats(out_params, df)
    print("1. MAE = %.4f\n2. RMSE = %.4f\n3. MAX = %.4f" % (mae, rmse, max_e))
    return out_params, mae, rmse, max_e


def opt_cross_val(
    df: pd.DataFrame,
    nfolds: int = 5,
) -> None:
    """
    opt_cross_val performs n-fold cross validation on opt*.pkl df from
    gather_data3
    """
    folds = get_folds(5, len(df))

    for n, fold in enumerate(folds):
        print(f"Fold {n}")
        df["Fitset"] = fold
        training = df[df["Fitset"] == True]
        testing = df[df["Fitset"] == False]
        print(f"Training: {len(training)}")
        print(f"Testing: {len(testing)}")
        # ret = opt.minimize(compute_int_energy, init, training, method='powell')
        optimization(training)

    return
