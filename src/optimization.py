from .setup import (
    create_pt_dict,
    compute_bj_opt,
    gather_data3,
    compute_bj_pairs,
    compute_bj_from_dimer_AB,
    calc_dftd4_props,
    compute_bj_from_dimer_AB_all_C6s,
)
import scipy.optimize as opt
import time
import pandas as pd
import numpy as np
from .tools import print_cartesians


def HF_only() -> (float, float, float):
    """
    HF_only ...
    """
    df = pd.read_pickle("base.pkl")
    basis_set = "adz"
    HF_vals = [1.61679827, 0.44959224, 3.35743605]
    mae, rmse, max_e = compute_int_energy_stats(HF_vals, df, "HF_jdz")
    print("        1. MAE  = %.4f" % mae)
    print("        2. RMSE = %.4f" % rmse)
    print("        3. MAX  = %.4f" % max_e)
    return mae, rmse, max_e


def SAPT0_only() -> (float, float, float):
    """
    SAPT0_only returns stats on just SAPT0
    """
    # df = pd.read_pickle("base.pkl")
    # basis_set = "adz"
    # HF_vals = [1.61679827, 0.44959224, 3.35743605]
    # mae, rmse, max_e = compute_int_energy_stats(HF_vals, df, "HF_jdz")
    # print("        1. MAE  = %.4f" % mae)
    # print("        2. RMSE = %.4f" % rmse)
    # print("        3. MAX  = %.4f" % max_e)
    return


def find_max_e(
    df: pd.DataFrame,
    params: [] = [0.5971246, 0.64577916, 1.17106229],
    hf_key: str = "HF_adz",
    count: int = 5,
) -> list:
    """
    find_max_e finds maximum errors
    """
    print(f"\nhf_key = {hf_key}, Params = {params}")
    diff = np.zeros(len(df))
    el_dc = create_pt_dict()
    df["d4"] = df.apply(
        lambda row: compute_bj_from_dimer_AB_all_C6s(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            row["C6_A"],
            row["C6_B"],
        ),
        # lambda row: compute_bj_from_dimer_AB(
        #     params,
        #     row["Geometry"][:, 0],  # pos
        #     row["Geometry"][:, 1:],  # carts
        #     row["monAs"],
        #     row["monBs"],
        #     row["C6s"],
        # ),
        # lambda row: compute_bj_pairs(
        #     params,
        #     row["Geometry"][:, 0],  # pos
        #     row["Geometry"][:, 1:],  # carts
        #     row["monAs"],
        #     row["monBs"],
        #     row["C6s"],
        #     mult_out=627.509,
        # ),
        # lambda row: compute_bj_opt(
        #     params,
        #     row["Geometry"][:, 0],  # pos
        #     row["Geometry"][:, 1:],  # carts
        #     row["C6s"],
        #     row["C8s"],
        #     row["monAs"],
        #     row["monBs"],
        # ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    mae = df["diff"].abs().sum() / len(df["diff"])
    rmse = (df["diff"] ** 2).mean() ** 0.5
    df["diff_abs"] = df["diff"].abs()
    max_e = df["diff_abs"].max()
    df = df.sort_values(by=["diff_abs"], ascending=False)
    df = df.reset_index(drop=False)
    print("        1. MAE  = %.4f" % mae)
    print("        2. RMSE = %.4f" % rmse)
    print("        3. MAX  = %.4f" % max_e)
    print(
        df[
            [
                "index",
                "DB",
                "Benchmark",
                hf_key,
                "d4",
                "diff",
                "diff_abs",
            ]
        ].head(30)
    )
    for i in range(count):
        print(f"\nMol {i}")
        print(df.iloc[i])
        print("\nCartesians")
        print_cartesians(df.iloc[i]["Geometry"])
    return mae, rmse, max_e, df


def compute_int_energy_stats(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
) -> (float, float, float,):
    """
    compute_int_energy is used to optimize paramaters for damping function in dftd4
    """
    diff = np.zeros(len(df))
    el_dc = create_pt_dict()
    df["d4"] = df.apply(
        lambda row: compute_bj_pairs(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            mult_out=627.509,
        ),
        # lambda row: compute_bj_from_dimer_AB(
        #     params,
        #     row["Geometry"][:, 0],  # pos
        #     row["Geometry"][:, 1:],  # carts
        #     row["monAs"],
        #     row["monBs"],
        #     row["C6s"],
        # ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    mae = df["diff"].abs().sum() / len(df["diff"])
    rmse = (df["diff"] ** 2).mean() ** 0.5
    max_e = df["diff"].abs().max()
    return mae, rmse, max_e


def compute_int_energy(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
):
    """
    compute_int_energy is used to optimize paramaters for damping function in dftd4
    """
    rmse = 0
    diff = np.zeros(len(df))
    el_dc = create_pt_dict()
    df["d4"] = df.apply(
        lambda row: compute_bj_pairs(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["monAs"],
            row["monBs"],
            row["C6s"],
            mult_out=627.509,
        ),
        # lambda row: compute_bj_from_dimer_AB(
        #     params,
        #     row["Geometry"][:, 0],  # pos
        #     row["Geometry"][:, 1:],  # carts
        #     row["monAs"],
        #     row["monBs"],
        #     row["C6s"],
        # ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
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
    hf_key: str = "HF INTERACTION ENERGY",
):
    # &  s6=1.0000_wp, s8=3.02227550_wp, a1=0.47396846_wp, a2=4.49845309_wp )
    print("RMSE\t\tparams")
    ret = opt.minimize(
        # compute_int_energy, args=(df, hf_key), x0=params, method="powell"
        compute_int_energy, args=(df, hf_key), x0=params, method="powell"
    )
    print("\nResults\n")
    out_params = ret.x
    mae, rmse, max_e = compute_int_energy_stats(out_params, df, hf_key)
    # print("1. MAE = %.4f\n2. RMSE = %.4f\n3. MAX = %.4f" % (mae, rmse, max_e))
    return out_params, mae, rmse, max_e


def avg_matrix(
    arr: np.array,
) -> np.array:
    """
    avg_matrix computes average of each column
    """
    s = len(arr[0, :])
    out = np.zeros(s)
    for i in range(s):
        out[i] = arr[:, i].sum() / len(arr[:, 0])
    return out


def stats_testing_set(
    params,
    test,
) -> (float, float, float):
    """
    stats_testing_set ...
    """


def opt_cross_val(
    df: pd.DataFrame,
    nfolds: int = 5,
    start_params: [] = [3.02227550, 0.47396846, 4.49845309],
    hf_key: str = "HF INTERACTION ENERGY",
) -> None:
    """
    opt_cross_val performs n-fold cross validation on opt*.pkl df from
    gather_data3
    """
    start = time.time()
    folds = get_folds(nfolds, len(df))
    stats = np.zeros((nfolds, 3))
    p_out = np.zeros((nfolds, len(start_params)))
    mp, mmae, mrmse, mmax_e = optimization(df, start_params, hf_key)
    print("1. MAE = %.4f\n2. RMSE = %.4f\n3. MAX = %.4f" % (mmae, mrmse, mmax_e))
    for n, fold in enumerate(folds):
        print(f"Fold {n} Start")
        df["Fitset"] = fold
        training = df[df["Fitset"] == True]
        training = training.reset_index(drop=True)
        testing = df[df["Fitset"] == False]
        testing = testing.reset_index(drop=True)
        print(f"Training: {len(training)}")
        print(f"Testing: {len(testing)}")

        o_params, omae, ormse, omax_e = optimization(training, mp, hf_key)
        mae, rmse, max_e = compute_int_energy_stats(o_params, testing, hf_key)

        stats[n] = np.array([mae, rmse, max_e])
        p_out[n] = o_params
        print(f"Fold {n} End")

    avg = avg_matrix(stats)
    mae, rmse, max_e = avg

    total_time = (time.time() - start) / 60
    print("\nTime = %.2f Minutes\n" % total_time)
    print("\n\t%d Fold Procedure" % nfolds)
    print("\nParameters:\n", p_out)
    print("\nStats:\n", stats)
    print("\nFinal Results")
    print("\n\tFull Optimization")
    print("\nParameters\n")
    print("        1. s6 = %.6f" % mp[0])
    print("        2. a1 = %.6f" % mp[1])
    print("        3. a2 = %.6f" % mp[2])
    print("\nStats\n")
    print("        1. MAE  = %.4f" % mmae)
    print("        2. RMSE = %.4f" % mrmse)
    print("        3. MAX  = %.4f" % mmax_e)
    print("\nFinal %d-Fold Averaged Stats\n" % (nfolds))
    print("        1. MAE  = %.4f" % mae)
    print("        2. RMSE = %.4f" % rmse)
    print("        3. MAX  = %.4f" % max_e)
    return
