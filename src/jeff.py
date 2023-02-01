import argparse
import numpy as np
import pandas as pd
import scipy.optimize as opt
import time
from src.tools import print_cartesians, stats_to_latex_row
from qcelemental import constants


def d3data_stats(df):
    bases = [
        "HF_dz",
        "HF_jdz",
        "HF_adz",
        "HF_tz",
        # "HF_jdz_no_cp",
        # "HF_dz_no_cp",
        # "HF_qz",
        # "HF_qz_no_cp",
        # "HF_qz_no_df",
        # "HF_qz_conv_e_4",
    ]
    with open("stats.txt", "w") as f:
        for i in bases:
            print(i)
            mae, rmse, max_e, mad, mean_dif = compute_error_stats_d3(
                [0.713190, 0.079541, 3.627854], df, i
            )
            v = stats_to_latex_row(i, rmse, max_e, mad, mean_dif)
            f.write(v)
            print(rmse, max_e, mad, mean_dif)


def compute_bj(params, d3data):
    s8, a1, a2 = params
    energy = 0.0
    for pair in d3data:
        atom1, atom2, R, R0, C6, C8 = pair

        R0 = np.sqrt(C8 / C6)
        energy += C6 / (R**6.0 + (a1 * R0 + a2) ** 6.0)
        energy += s8 * C8 / (R**8.0 + (a1 * R0 + a2) ** 8.0)

    energy *= -constants.conversion_factor("hartree", "kcal / mol")
    return energy


def compute_error_stats_d3(
    params,
    df,
    hf_key,
    # params=[0.713190, 0.079541, 3.627854],
) -> []:
    """
    compute_error_stats uses jeffs d3date to compute error
    stats from different HF_ie
    """
    df["d3"] = df.apply(
        lambda r: compute_bj(params, r["D3Data"]),
        axis=1,
    )

    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d3"]), axis=1)
    df["y_pred"] = df.apply(lambda r: r[hf_key] + r["d3"], axis=1)
    mae = df["diff"].abs().mean()
    rmse = (df["diff"] ** 2).mean() ** 0.5
    max_e = df["diff"].abs().max()
    mad = df["diff"].mad()
    mean_dif = df["diff"].mean()
    return mae, rmse, max_e, mad, mean_dif


def compute_int_energy_d3(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
):
    """
    compute_int_energy_d3 is used to optimize paramaters for d3
    """
    for i in params:
        if i < 0:
            return 10
    rmse = 0
    diff = np.zeros(len(df))
    df["d3"] = df.apply(
        lambda r: compute_bj(params, r["D3Data"]),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d3"]), axis=1)
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params.tolist())
    df["diff"] = 0
    return rmse


def optimization_d3(
    df: pd.DataFrame,
    params: [] = [0.713190, 0.079541, 3.627854],
    hf_key: str = "HF_dz",
    opt_method: str = "powell",  # "lm"
):
    """
    Use with src/optimization.opt_cross_val()
    """
    print("RMSE\t\tparams")
    ret = opt.minimize(
        compute_int_energy_d3,
        args=(df, hf_key),
        x0=params,
        method="powell",
    )
    print("\nResults\n")
    out_params = ret.x
    return out_params

def error_stats_fixed_params(
    df,
    bases=[
        "HF_dz",
        "HF_jdz",
        "HF_adz",
        "HF_tz",
        "HF_atz",
    ],
    p=[0.732484, 0.094481, 3.632253],
    error_stat_func=compute_error_stats_d3,
    d4=True,
) -> None:
    """
    compute_d3_d4_errors_fixed computes error stats on d3 and d4
    """
    stats = {
        "method": [],
        "RMSE": [],
        "MAD": [],
        "MD": [],
        "MAX_E": [],
    }
    for i in bases:
        print(i)
        mae, rmse, max_e, mad, mean_diff = error_stat_func(p, df, i)
        stats["method"].append(hf_key_to_latex_cmd(i, d4))
        stats["RMSE"].append(rmse)
        stats["MAD"].append(mad)
        stats["MAX_E"].append(max_e)
        stats["MD"].append(mean_diff)
    print(stats)
    if d4:
        l_out = "D4_fixed_stats"
    else:
        l_out = "D3_fixed_stats"
    df_to_latex_table_round(pd.DataFrame(stats), l_out=l_out)
    return

