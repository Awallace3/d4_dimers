import os
from . import locald4
from . import r4r2
from . import jeff
from . import paramsTable
import scipy.optimize as opt
import time
import pandas as pd
import numpy as np
from .tools import print_cartesians, df_to_latex_table_round
from qcelemental import constants
import dispersion


def chunkify(df: pd.DataFrame, chunk_size: int):
    start = 0
    length = df.shape[0]

    # If DF is smaller than the chunk, return the DF
    if length <= chunk_size:
        yield df[:]
        return

    # Yield individual chunks
    while start + chunk_size <= length:
        yield df[start : chunk_size + start]
        start = start + chunk_size

    # Yield the remainder chunk, if needed
    if start < length:
        yield df[start:]


def HF_only() -> (float, float, float):
    """
    HF_only ...
    """
    df = pd.read_pickle("base.pkl")
    basis_set = "adz"
    HF_vals = [1.61679827, 0.44959224, 3.35743605]
    mae, rmse, max_e, mad, mean_dif = compute_int_energy_stats(HF_vals, df, "HF_jdz")
    print("        1. MAE  = %.4f" % mae)
    print("        2. RMSE = %.4f" % rmse)
    print("        3. MAX  = %.4f" % max_e)
    print("        4. MAD  = %.4f" % mad)
    print("        4. MD   = %.4f" % mean_dif)
    return mae, rmse, max_e


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
    r4r2_ls = r4r2.r4r2_vals_ls()
    df["d4"] = df.apply(
        lambda row: locald4.compute_bj_dimer_f90(
            params,
            row,
            r4r2_ls=r4r2_ls,
        ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    mae = df["diff"].abs().sum() / len(df["diff"])
    rmse = (df["diff"] ** 2).mean() ** 0.5
    df["diff_abs"] = df["diff"].abs()
    max_e = df["diff_abs"].max()
    mad = abs(df["diff"] - df["diff"].mean()).mean()
    df = df.sort_values(by=["diff_abs"], ascending=False)
    df = df.reset_index(drop=False)
    print("        1. MAE  = %.4f" % mae)
    print("        2. RMSE = %.4f" % rmse)
    print("        3. MAX  = %.4f" % max_e)
    print("        4. MAD  = %.4f" % mad)
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


def compute_int_energy_stats_dftd4_key(
    df: pd.DataFrame,
    hf_key: str = "HF_jdz",
    dftd4_key: str = "HF_jdz_dftd4",
) -> (float, float, float,):
    """
    stats for atm
    """
    params = [1.61679827, 0.44959224, 3.35743605]
    t = df[hf_key].isna().sum()
    assert t == 0, f"The HF_col provided has np.nan values present, {t}"
    t = df[dftd4_key].isna().sum()
    assert t == 0, f"The HF_col provided has np.nan values present, {t}"
    r4r2_ls = r4r2.r4r2_vals_ls()
    df[f"{hf_key}_d4"] = df.apply(
        lambda row: locald4.compute_bj_dimer_f90(
            params,
            row,
            r4r2_ls=r4r2_ls,
        ),
        axis=1,
    )
    df[f"{hf_key}_d4_sum"] = df.apply(lambda r: r[hf_key] + r["HF_jdz_d4"], axis=1)
    df["HF_diff"] = df.apply(lambda r: r[f"{hf_key}_d4_sum"] - r[dftd4_key], axis=1)
    return


def compute_int_energy_stats_DISP(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    parallel=False,
    print_results=False,
    chunk_count=1000,
    force_ATM_on=False,
) -> (float, float, float,):
    t = df[hf_key].isna().sum()
    assert t == 0, f"The HF_col provided has np.nan values present, {t}"
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(
        params, force_ATM_on=force_ATM_on
    )
    print(f"{params_2B = }")
    print(f"{params_ATM = }")

    diff = np.zeros(len(df))
    r4r2_ls = r4r2.r4r2_vals_ls()
    print(f"{params = }")
    if parallel:
        chunks = []
        for i in chunkify(df, chunk_count):
            c = i.p_apply(
                lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
                    row,
                    params_2B,
                    params_ATM,
                ),
                axis=1,
            )
            chunks.append(c)
        df = pd.concat(chunks, axis=1)
    else:
        df["d4"] = df.apply(
            lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
                row,
                params_2B,
                params_ATM,
            ),
            axis=1,
        )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    mae = df["diff"].abs().mean()
    rmse = (df["diff"] ** 2).mean() ** 0.5
    max_e = df["diff"].abs().max()
    mad = abs(df["diff"] - df["diff"].mean()).mean()
    mean_dif = df["diff"].mean()
    if print_results:
        print("        1. MAE  = %.4f" % mae)
        print("        2. RMSE = %.4f" % rmse)
        print("        3. MAX  = %.4f" % max_e)
        print("        4. MAD  = %.4f" % mad)
        print("        4. MD   = %.4f" % mean_dif)
    return mae, rmse, max_e, mad, mean_dif


def compute_int_energy_stats_DISP_TT(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    parallel=False,
    print_results=False,
    chunk_count=1000,
    force_ATM_on=False,
) -> (float, float, float,):
    t = df[hf_key].isna().sum()
    assert t == 0, f"The HF_col provided has np.nan values present, {t}"
    # params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params)
    # params_2B, params_ATM = paramsTable.get_params("SAPT0_adz_3_IE_2B")
    params_2B, params_ATM = paramsTable.get_params("HF_ATM_SHARED")
    params_ATM = np.array([0.0, 0.0, params[0], params[1], 1.0])
    print(f"{params_2B = }")
    print(f"{params_ATM = }")

    diff = np.zeros(len(df))
    r4r2_ls = r4r2.r4r2_vals_ls()
    df["d4"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_TT_dimer(
            row,
            params_2B,
            params_ATM,
        ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    mae = df["diff"].abs().mean()
    rmse = (df["diff"] ** 2).mean() ** 0.5
    max_e = df["diff"].abs().max()
    mad = abs(df["diff"] - df["diff"].mean()).mean()
    mean_dif = df["diff"].mean()
    if print_results:
        print("        1. MAE  = %.4f" % mae)
        print("        2. RMSE = %.4f" % rmse)
        print("        3. MAX  = %.4f" % max_e)
        print("        4. MAD  = %.4f" % mad)
        print("        4. MD   = %.4f" % mean_dif)
    return mae, rmse, max_e, mad, mean_dif


def compute_int_energy_stats_DISP_SR(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    SR_func=dispersion.disp.disp_SR_1,
    print_results=False,
    chunk_count=1000,
) -> (float, float, float,):
    t = df[hf_key].isna().sum()
    assert t == 0, f"The HF_col provided has np.nan values present, {t}"
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params)
    print("SETTING s9=1.0 for SR ATM to be non-zero!")

    params_ATM[-1] = 1.0
    print(f"{params_2B = }")
    print(f"{params_ATM = }")

    diff = np.zeros(len(df))
    print(f"{params = }")
    df["d4_ATM"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_SR_dimer(
            row,
            params_2B,
            params_ATM,
            SR_func=SR_func,
        ),
        axis=1,
    )
    df["d4_2B"] = df.apply(
        lambda r: locald4.compute_disp_2B_dimer(params_2B, r), axis=1
    )
    df["d4"] = df.apply(lambda r: r["d4_2B"] + r["d4_ATM"], axis=1)

    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)

    print("Mol\tDiff\tBench\tTotal\t2B\tATM\tATM_SR")
    for n, r in df.iterrows():
        line = f"{n} {r['diff']:5.4f} {r['Benchmark']:5.4f} {(r[hf_key] + r['d4']):5.4f} {r[hf_key]:5.4f} {r['d4']:5.4f} {r['d4_2B']:5.4f} {r['d4_ATM']:5.16f}"
        print(line)

    mae = df["diff"].abs().mean()
    rmse = (df["diff"] ** 2).mean() ** 0.5
    mse = (df["diff"] ** 2).mean()
    max_e = df["diff"].abs().max()
    mad = abs(df["diff"] - df["diff"].mean()).mean()
    mean_dif = df["diff"].mean()
    if print_results:
        print("SR ATM ENABLED")
        print("        1. MAE  = %.4f" % mae)
        print("        2. RMSE = %.4f" % rmse)
        print("        3. MAX  = %.4f" % max_e)
        print("        4. MAD  = %.4f" % mad)
        print("        4. MD   = %.4f" % mean_dif)
        print("        5. MSE  = %.4f" % mse)
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4_2B"]), axis=1)
    mae = df["diff"].abs().mean()
    rmse = (df["diff"] ** 2).mean() ** 0.5
    mse = (df["diff"] ** 2).mean()
    max_e = df["diff"].abs().max()
    mad = abs(df["diff"] - df["diff"].mean()).mean()
    mean_dif = df["diff"].mean()
    if print_results:
        print("2B ONLY")
        print("        1. MAE  = %.4f" % mae)
        print("        2. RMSE = %.4f" % rmse)
        print("        3. MAX  = %.4f" % max_e)
        print("        4. MAD  = %.4f" % mad)
        print("        4. MD   = %.4f" % mean_dif)
        print("        5. MSE  = %.4f" % mse)
    return mae, rmse, max_e, mad, mean_dif


def compute_int_energy_DISP(
    params,
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    force_ATM_on: bool = False,
    prevent_negative_params: bool = False,
    parallel=False,
    chunk_count=1000,
):
    """
    compute_int_energy_DISP is used to optimize paramaters for damping function in dftd4
    """
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(
        params, force_ATM_on=force_ATM_on
    )
    if prevent_negative_params:
        for i in params:
            if i < 0:
                return 10
    rmse = 0
    diff = np.zeros(len(df))
    if parallel:
        chunks = []
        cnt = 0
        for i in chunkify(df, chunk_count):
            print(f"{cnt = }")
            c = i.p_apply(
                lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
                    row,
                    params_2B,
                    params_ATM,
                ),
                axis=1,
            )
            cnt += 1
            chunks.append(c)
        chunks = pd.concat(chunks, axis=0)
        df["d4"] = chunks.to_list()
    else:
        df["d4"] = df.apply(
            lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
                row,
                params_2B,
                params_ATM,
            ),
            axis=1,
        )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params.tolist())
    return rmse


def compute_int_energy_DISP_TT(
    params,
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    force_ATM_on: bool = False,
):
    """
    compute_int_energy_DISP_TT is used to optimize paramaters for damping function in dftd4 with TT damping ATM function
    """
    # params_2B, params_ATM = paramsTable.get_params("SAPT0_adz_3_IE_2B")
    params_2B, params_ATM = paramsTable.get_params("HF_ATM_SHARED")
    params_ATM = np.array([0.0, 0.0, params[0], params[1], 1.0])
    rmse = 0
    df["d4"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_TT_dimer(
            row,
            params_2B,
            params_ATM,
        ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params.tolist())
    if np.isnan(rmse):
        return 10
    return rmse


def compute_int_energy_stats(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    parallel=False,
    print_results=False,
    ATM=False,
    chunk_count=1000,
) -> (float, float, float,):
    t = df[hf_key].isna().sum()
    assert t == 0, f"The HF_col provided has np.nan values present, {t}"
    compute_bj = locald4.compute_bj_dimer_f90
    if ATM:
        compute_bj = locald4.compute_bj_dimer_f90_ATM

    diff = np.zeros(len(df))
    r4r2_ls = r4r2.r4r2_vals_ls()
    print(f"{params = }")
    if parallel:
        chunks = []
        for i in chunkify(df, chunk_count):
            c = i.p_apply(
                lambda row: locald4.compute_bj_dimer_f90_ATM(
                    params,
                    row,
                    r4r2_ls=r4r2_ls,
                ),
                axis=1,
            )
            chunks.append(c)
        df = pd.concat(chunks, axis=1)
    else:
        df["d4"] = df.apply(
            lambda row: compute_bj(
                params,
                row,
                r4r2_ls=r4r2_ls,
            ),
            axis=1,
        )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    mae = df["diff"].abs().mean()
    rmse = (df["diff"] ** 2).mean() ** 0.5
    max_e = df["diff"].abs().max()
    mad = abs(df["diff"] - df["diff"].mean()).mean()
    mean_dif = df["diff"].mean()
    if print_results:
        print("        1. MAE  = %.4f" % mae)
        print("        2. RMSE = %.4f" % rmse)
        print("        3. MAX  = %.4f" % max_e)
        print("        4. MAD  = %.4f" % mad)
        print("        4. MD   = %.4f" % mean_dif)
    return mae, rmse, max_e, mad, mean_dif


def compute_int_energy_least_squares(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    force_ATM_on: bool = False,
    # ban_neg_params: bool = False,
):
    """
    compute_int_energy_least_squares is used to optimize paramaters for damping function in dftd4 with levenberg-Marquedt needing a difference list returned to optimizer
    """
    rmse = 0
    # if ban_neg_params:
    #     for i in params:
    #         if i < 0:
    #             return [10 for i in range(len(df))]
    r4r2_ls = r4r2.r4r2_vals_ls()
    df["d4"] = df.apply(
        lambda row: locald4.compute_bj_dimer_f90(
            params,
            row,
            r4r2_ls=r4r2_ls,
        ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params.tolist())
    return df["diff"].tolist()
    # return rmse


def compute_int_energy_least_squares_ATM(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    force_ATM_on: bool = False,
    # ban_neg_params: bool = False,
):
    """
    compute_int_energy_least_squares_ATM is used to optimize paramaters for damping function in dftd4 with levenberg-Marquedt needing a difference list returned to optimizer
    """
    rmse = 0
    # if ban_neg_params:
    #     for i in params:
    #         if i < 0:
    #             return [10 for i in range(len(df))]
    r4r2_ls = r4r2.r4r2_vals_ls()
    df["d4"] = df.apply(
        lambda row: locald4.compute_bj_dimer_f90_ATM(
            params,
            row,
            r4r2_ls=r4r2_ls,
        ),
        axis=1,
    )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params.tolist())
    return df["diff"].tolist()


def compute_int_energy(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    force_ATM_on: bool = False,
    prevent_negative_params: bool = False,
    parallel=False,
    chunk_count=1000,
):
    """
    compute_int_energy is used to optimize paramaters for damping function in dftd4
    """
    if prevent_negative_params:
        for i in params:
            if i < 0:
                return 10
    rmse = 0
    diff = np.zeros(len(df))
    r4r2_ls = r4r2.r4r2_vals_ls()
    if parallel:
        chunks = []
        cnt = 0
        for i in chunkify(df, chunk_count):
            print(f"{cnt = }")
            c = i.p_apply(
                lambda row: locald4.compute_bj_dimer_f90(
                    params,
                    row,
                    r4r2_ls=r4r2_ls,
                ),
                axis=1,
            )
            cnt += 1
            chunks.append(c)
        chunks = pd.concat(chunks, axis=0)
        df["d4"] = chunks.to_list()
    else:
        df["d4"] = df.apply(
            lambda row: locald4.compute_bj_dimer_f90(
                params,
                row,
                r4r2_ls=r4r2_ls,
            ),
            axis=1,
        )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params.tolist())
    return rmse


def compute_int_energy_ATM(
    params: [float],
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
    force_ATM_on: bool = False,
    prevent_negative_params: bool = False,
    parallel=False,
    chunk_count=1000,
):
    """
    compute_int_energy_ATM is used to optimize paramaters for damping function in dftd4
    """
    if prevent_negative_params:
        for i in params:
            if i < 0:
                return 10
    rmse = 0
    diff = np.zeros(len(df))
    r4r2_ls = r4r2.r4r2_vals_ls()
    if parallel:
        chunks = []
        for i in chunkify(df, chunk_count):
            i["d4"] = i.p_apply(
                lambda row: locald4.compute_bj_dimer_f90_ATM(
                    params,
                    row,
                    r4r2_ls=r4r2_ls,
                ),
                axis=1,
            )
            chunks.append(i)
        df = pd.concat(chunks, axis=1)
    else:
        df["d4"] = df.apply(
            lambda row: locald4.compute_bj_dimer_f90_ATM(
                params,
                row,
                r4r2_ls=r4r2_ls,
            ),
            axis=1,
        )
    df["diff"] = df.apply(lambda r: r["Benchmark"] - (r[hf_key] + r["d4"]), axis=1)
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse, params.tolist())
    return rmse


def compute_int_energy_NO_DAMPING(
    df: pd.DataFrame,
    hf_key: str = "HF INTERACTION ENERGY",
):
    """
    compute_int_energy_NO_DAMPING is used to optimize paramaters for damping function in dftd4
    """
    rmse = 0
    diff = np.zeros(len(df))
    df["d4_NO_DAMPING"] = df.apply(
        lambda row: locald4.compute_bj_from_dimer_AB_all_C6s_NO_DAMPING(
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
    df["diff"] = df.apply(
        lambda r: r["Benchmark"] - (r[hf_key] + r["d4_NO_DAMPING"]), axis=1
    )
    rmse = (df["diff"] ** 2).mean() ** 0.5
    print("%.8f\t" % rmse)
    df["diff"] = 0
    return rmse, df


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
    version={
        "method": "powell",
        "compute_energy": "compute_int_energy_DISP",
    },
    force_ATM_on=False,
    bounds=(-3.0, 8.0),
):
    if version["compute_energy"] == "compute_int_energy_DISP":
        compute = compute_int_energy_DISP
    elif version["compute_energy"] == "compute_int_energy_DISP_TT":
        compute = compute_int_energy_DISP_TT
    elif version["compute_energy"] == "compute_int_energy":
        compute = compute_int_energy
    elif version["compute_energy"] == "compute_int_energy_ATM":
        compute = compute_int_energy_ATM
    elif version["compute_energy"] == "compute_int_energy_least_squares":
        compute = compute_int_energy_least_squares
    elif version["compute_energy"] == "compute_int_energy_least_squares_ATM":
        compute = compute_int_energy_least_squares_ATM
    elif version["compute_energy"] == "jeff_d3":
        compute = jeff.compute_int_energy_d3
    else:
        raise Exception("compute_energy not defined")
    print("RMSE\t\tparams")
    ret = opt.minimize(
        compute,
        args=(df, hf_key, force_ATM_on),
        x0=params,
        method=version["method"],
        bounds=[bounds for i in range(len(params))],
    )
    print("\nResults\n")
    out_params = ret.x
    return out_params


def optimization_least_squares(
    df: pd.DataFrame,
    params: [] = [1.61679827, 0.44959224, 3.35743605],
    hf_key: str = "HF INTERACTION ENERGY",
):
    print("RMSE\t\tparams")
    ret = opt.least_squares(
        compute_int_energy_least_squares,
        args=(df, hf_key),
        x0=params,
        method="lm",
    )
    print("\nResults\n")
    out_params = ret.x[0]
    print(f"{out_params = }")
    mae, rmse, max_e, mad, mean_diff = compute_int_energy_stats(out_params, df, hf_key)
    print(out_params)
    return out_params, mae, rmse, max_e, mad, mean_diff


def avg_matrix(
    arr: np.array,
) -> np.array:
    """
    avg_matrix computes average of each column
    """
    s = len(arr[0, :])
    out = np.zeros(s)
    for i in range(s - 1):
        out[i] = arr[:, i].sum() / len(arr[:, 0])
    out[-1] = np.amax(arr[:, -1])
    return out

def opt_val_no_folds(
    df: pd.DataFrame,
    start_params: [] = [3.02227550, 0.47396846, 4.49845309],
    hf_key: str = "HF INTERACTION ENERGY",
    version={
        "method": "powell",
        "compute_energy": "compute_int_energy_DISP",
        "compute_stats": "compute_int_energy_stats_DISP",
    },
    force_ATM_on=False,
) -> None:
    """
    opt_cross_val performs n-fold cross validation on opt*.pkl df from
    """
    print(f"{force_ATM_on = }")

    if version["compute_stats"] == "compute_int_energy_stats_DISP":
        compute_stats = compute_int_energy_stats_DISP
    elif version["compute_stats"] == "compute_int_energy_stats_DISP_TT":
        compute_stats = compute_int_energy_stats_DISP_TT
    elif version["compute_stats"] == "compute_int_energy_stats":
        compute_stats = compute_int_energy_stats
    elif version["compute_stats"] == "jeff_d3":
        compute_stats = jeff.compute_error_stats_d3
    else:
        raise Exception("compute_stats not defined")
    opt_type = version["method"]
    print(f"{hf_key = }")
    print(f"{version = }")

    nans = df[hf_key].isna().sum()
    inds = df.index[df[hf_key].isna()]
    assert nans == 0, f"The HF_col provided has np.nan values present with {inds} nans"
    start = time.time()
    print(start_params)
    mp = optimization(df, start_params, hf_key, version, force_ATM_on=force_ATM_on)
    mmae, mrmse, mmax_e, mmad, mmean_diff = compute_stats(
        mp, df, hf_key, force_ATM_on=force_ATM_on
    )
    stats = {
        "method": [f"{hf_key} full"],
        # "Optimization Algorithm": [opt_type],
        "RMSE": [mrmse],
        "MAD": [mmad],
        "MD": [mmean_diff],
        "MAX_E": [mmax_e],
    }
    total_time = (time.time() - start) / 60
    print("\nTime = %.2f Minutes\n" % total_time)
    print("\nStarting Parameters\n")
    print(start_params)
    print("\nStats")
    print(stats)
    print("\nFinal Parameters for the whole data set\n")
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(mp)
    all_params = repr(np.array([params_2B, params_ATM], np.float64))
    print(f'"{hf_key}": np.{all_params},')
    return

def opt_cross_val(
    df: pd.DataFrame,
    nfolds: int = 5,
    start_params: [] = [3.02227550, 0.47396846, 4.49845309],
    hf_key: str = "HF INTERACTION ENERGY",
    output_l_marker: str = "G_",
    version={
        "method": "powell",
        "compute_energy": "compute_int_energy_DISP",
        "compute_stats": "compute_int_energy_stats_DISP",
    },
    force_ATM_on=False,
) -> None:
    """
    opt_cross_val performs n-fold cross validation on opt*.pkl df from
    """
    print(f"{force_ATM_on = }")

    if version["compute_stats"] == "compute_int_energy_stats_DISP":
        compute_stats = compute_int_energy_stats_DISP
    elif version["compute_stats"] == "compute_int_energy_stats_DISP_TT":
        compute_stats = compute_int_energy_stats_DISP_TT
    elif version["compute_stats"] == "compute_int_energy_stats":
        compute_stats = compute_int_energy_stats
    elif version["compute_stats"] == "jeff_d3":
        compute_stats = jeff.compute_error_stats_d3
    else:
        raise Exception("compute_stats not defined")
    opt_type = version["method"]
    print(f"{hf_key = }")
    print(f"{version = }")

    nans = df[hf_key].isna().sum()
    inds = df.index[df[hf_key].isna()]
    assert nans == 0, f"The HF_col provided has np.nan values present with {inds} nans"
    start = time.time()
    folds = get_folds(nfolds, len(df))
    stats_np = np.zeros((nfolds, 4))
    p_out = np.zeros((nfolds, len(start_params)))
    print(start_params)
    mp = optimization(df, start_params, hf_key, version, force_ATM_on=force_ATM_on)
    print(f"{mp = }")
    mmae, mrmse, mmax_e, mmad, mmean_diff = compute_stats(
        mp, df, hf_key, force_ATM_on=force_ATM_on
    )
    stats = {
        "method": [f"{hf_key} full"],
        # "Optimization Algorithm": [opt_type],
        "RMSE": [mrmse],
        "MAD": [mmad],
        "MD": [mmean_diff],
        "MAX_E": [mmax_e],
    }
    print("1. MAE = %.4f\n2. RMSE = %.4f\n3. MAX = %.4f" % (mmae, mrmse, mmax_e))
    # reset params to mp
    ff_start_params = mp.copy()
    for n, fold in enumerate(folds):
        print(f"Fold {n} Start")
        df["Fitset"] = fold
        training = df[df["Fitset"] == True]
        training = training.reset_index(drop=True)
        testing = df[df["Fitset"] == False]
        testing = testing.reset_index(drop=True)
        print(f"Training: {len(training)}")
        print(f"Testing: {len(testing)}")

        o_params = optimization(
            training, ff_start_params, hf_key, version, force_ATM_on=force_ATM_on
        )
        mae, rmse, max_e, mad, mean_diff = compute_stats(
            o_params, testing, hf_key, force_ATM_on=force_ATM_on
        )

        stats_np[n] = np.array([rmse, mad, mean_diff, max_e])
        p_out[n] = o_params
        print(f"Fold {n} End")

        stats["method"].append(f"{hf_key} fold {n+1}")
        # stats["Optimization Algorithm"].append(opt_type)
        stats["RMSE"].append(rmse)
        stats["MAD"].append(mad)
        stats["MD"].append(mean_diff)
        stats["MAX_E"].append(max_e)

    print("stats_np")
    print(stats_np)
    avg = avg_matrix(stats_np)
    rmse, mad, mean_diff, max_e = avg

    stats["method"].append(f"{hf_key}")
    # stats["Optimization Algorithm"].append(opt_type)
    stats["RMSE"].append(rmse)
    stats["MAX_E"].append(max_e)
    stats["MAD"].append(mad)
    stats["MD"].append(mean_diff)

    total_time = (time.time() - start) / 60
    print("\nTime = %.2f Minutes\n" % total_time)
    print("\n\t%d Fold Procedure" % nfolds)
    print("\nParameters:\n", p_out)
    print("\nStats:\n", stats)
    print("\nFinal Results")
    print("\n\tFull Optimization")
    print("\nStarting Parameters\n")
    print(start_params)
    print("\nFinal Parameters for the whole data set\n")
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(ff_start_params)
    all_params = repr(np.array([params_2B, params_ATM], np.float64))
    print(f'"{hf_key}": np.{all_params},')
    print("\nStats\n")
    print("        1. MAD  = %.4f" % mmad)
    print("        2. RMSE = %.4f" % mrmse)
    print("        3. MAX  = %.4f" % mmax_e)
    print("\nFinal %d-Fold Averaged Stats\n" % (nfolds))
    print("        1. MAD  = %.4f" % mad)
    print("        2. RMSE = %.4f" % rmse)
    print("        3. MAX  = %.4f" % max_e)
    print("Params:")
    print(mp)
    df2 = pd.DataFrame(stats)
    df_to_latex_table_round(
        df2,
        cols_round={
            "RMSE": 4,
            "MAX_E": 4,
            "MAD": 4,
            "MD": 4,
        },
        l_out=f"{output_l_marker}{hf_key}_5f_P",
    )
    return


def calc_dftd4_disp_pieces(atoms, geom, ma, mb, params, s9="0.0"):
    x, y, p, d = locald4.calc_dftd4_c6_c8_pairDisp2(atoms, geom, p=params, s9=s9)
    x, y, p, a = locald4.calc_dftd4_c6_c8_pairDisp2(
        atoms[ma], geom[ma, :], p=params, s9=s9
    )
    x, y, p, b = locald4.calc_dftd4_c6_c8_pairDisp2(
        atoms[mb], geom[mb, :], p=params, s9=s9
    )
    return d - (a + b)


def compute_dftd4_values(
    df,
    params=[1.61679827, 0.44959224, 3.35743605],
    s9="0.0",
    key="dftd4_disp_ie_grimme_params",
) -> pd.DataFrame:
    """
    compute_dftd4_values
    """
    m = constants.conversion_factor("hartree", "kcal / mol")
    df[key] = df.apply(
        lambda r: m
        * calc_dftd4_disp_pieces(
            r["Geometry"][:, 0],
            r["Geometry"][:, 1:],
            r["monAs"],
            r["monBs"],
            params,
            s9=s9,
        ),
        axis=1,
    )
    return df


def compute_stats_dftd4_values_fixed(
    df,
    cols=[
        "HF_dz",
        "HF_jdz_no_cp",
        "HF_jdz",
        "HF_qz_no_cp",
        "HF_qz_conv_e_4",
        "HF_qz",
    ],
    fixed_col="dftd4_disp_ie_grimme_params",
) -> None:
    """
    compute_stats_dftd4_values_fixed
    """
    print(f"\nfixed_col: {fixed_col}\n")
    stats = {
        "method": [],
        "RMSE": [],
        "MAX_E": [],
        "MAD": [],
        "MD": [],
    }
    assert df[fixed_col].isna().sum() == 0
    for c in cols:
        assert df[c].isna().sum() == 0
        df["ie"] = df[c] + df[fixed_col]
        df["diff"] = df["Benchmark"] - df["ie"]
        mean_diff = df["diff"].mean()
        rmse = ((df["diff"] ** 2).mean()) ** 0.5
        max_e = df["diff"].abs().max()
        mad = df["diff"].mad()
        stats["method"].append(c)
        stats["RMSE"].append(rmse)
        stats["MAX_E"].append(max_e)
        stats["MAD"].append(mad)
        stats["MD"].append(mean_diff)

    stats["method"].append("Grimme")
    stats["RMSE"].append(0.497190)
    stats["MAX_E"].append(np.nan)
    stats["MAD"].append(0.347320)
    stats["MD"].append(-0.025970)
    df2 = pd.DataFrame(stats)
    df_to_latex_table_round(
        df2,
        cols_round={"RMSE": 4, "MAX_E": 4, "MAD": 4, "MD": 4},
        l_out=fixed_col,
    )
    print(df2)
    return


def get_params():
    return {
        "HF_jdz": [1.61679827, 0.44959224, 3.35743605],
        "HF_adz": [1.61679827, 0.44959224, 3.35743605],
        "HF_dz": [1.61679827, 0.44959224, 3.35743605],
        "HF_tz": [1.61679827, 0.44959224, 3.35743605],
    }


def analyze_max_errors(
    df,
    count: int = 5,
) -> None:
    """
    analyze_max_errors looks at largest max errors
    """
    params_dc = get_params()
    for k, v in params_dc.items():
        find_max_e(df, v, k, count)
    return
