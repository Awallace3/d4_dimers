import numpy as np
import pandas as pd
import dispersion
from . import paramsTable
from . import locald4
from qm_tools_aw import tools
from sklearn.model_selection import train_test_split


def build_vals(pos, carts, C6, params_ATM, cols=7, max_N=None):
    if max_N is None:
        max_N = len(pos)
    vals = np.zeros(
        (int(max_N * (max_N - 1) * (max_N - 2) / 6), cols), dtype=np.float64
    )
    # energy = dispersion.disp.vals_for_SR(pos, carts, C6, params_ATM, vals)
    # energy = dispersion.disp.disp_SR_5_vals(pos, carts, C6, params_ATM, vals)
    energy = dispersion.disp.disp_SR_6_vals(pos, carts, C6, params_ATM, vals)
    # energy = dispersion.disp.disp_SR_4_vals(pos, carts, C6, params_ATM, vals)
    return energy, vals


def build_vals_molecule(r, params_ATM, max_N=None, cols=7):
    pos = r["Geometry_bohr"][:, 0]
    carts = r["Geometry_bohr"][:, 1:]
    monAs = r["monAs"]
    monBs = r["monBs"]
    C6s = r["C6s"]
    C6s_A = r["C6_A"]
    C6s_B = r["C6_B"]

    pos = np.array(pos, dtype=np.int32)
    pA, cA = pos[monAs].copy(), carts[monAs].copy()
    pB, cB = pos[monBs].copy(), carts[monBs].copy()
    # Generate ATM data for SR
    e_d, dimer_vals = build_vals(pos, carts, C6s, params_ATM, cols=cols, max_N=max_N)
    e_a, monA_vals = build_vals(pA, cA, C6s_A, params_ATM, cols=cols, max_N=max_N)
    e_b, monB_vals = build_vals(pB, cB, C6s_B, params_ATM, cols=cols, max_N=max_N)
    # Labeling as dimer and monomers for SR subtraction to get IE
    dimer_vals[:, 0] *= 1
    monA_vals[:, 0] *= -1
    monB_vals[:, 0] *= -1
    energies = np.array([e_d, -e_a, -e_b], dtype=np.float64)
    vals = np.concatenate((dimer_vals, monA_vals, monB_vals), axis=0)
    return energies, vals


def generate_SR_data_ATM(
    df,
    selected,
    target_HF_key="SAPT0_adz_3_IE",
    ncols=8,
    generate=True,
    params_key="SAPT0_adz_3_IE_ATM",
):
    r = df.iloc[0]
    params = paramsTable.get_params(params_key)
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params, force_ATM_on=True)
    print(f"{params_2B = }")
    print(f"{params_ATM = }")
    if generate:
        df[["SR_ATM", "xs_all"]] = df.apply(
            lambda r: build_vals_molecule(
                r,
                params_ATM,
                cols=ncols,
            ),
            axis=1,
            result_type="expand",
        )
        # for n, r in df.iterrows():
        #     df.at[n, "xs"][:, 0] *= locald4.hartree_to_kcalmol
        df["d4_ATM_E"] = df.apply(lambda r: sum(r["xs_all"][:, 0]), axis=1)
        df["xs"] = df.apply(lambda r: r["xs_all"][:, 1:], axis=1)
        print(df["xs"].iloc[0])
        print(params)
        # df["xs"] = df.apply(lambda r: r["xs"][:, 1:], axis=1)
        splits = []
        end = 0
        for i in range(len(df)):
            size = len(df.iloc[i]["xs"])
            start = end
            end += size
            splits.append(np.array([start, end], dtype=np.int64))
        df["splits"] = splits
        r1 = df.iloc[0]
        print(r1["xs"].shape, r1["splits"].shape)

        params_ATM[-1] = 0.0
        df["d4_2B"] = df.apply(
            lambda r: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
                r,
                params_2B,
                params_ATM,
            ),
            axis=1,
        )
    else:
        out = selected.replace(".pkl", "_SR.pkl")
        df = pd.read_pickle(out)

    df["y_pred"] = df.apply(
        lambda r: (r["Benchmark"] - (r[target_HF_key] + r["d4_2B"] + sum(r["SR_ATM"]))),
        axis=1,
    )
    df["ys"] = df.apply(
        lambda r: (r["Benchmark"] - (r[target_HF_key] + r["d4_2B"])) / r["d4_ATM_E"],
        axis=1,
    )
    df["ys_target"] = df.apply(
        lambda r: (r["Benchmark"] - (r[target_HF_key] + r["d4_2B"])),
        axis=1,
    )
    df['diff'] = df['ys_target'] - df['y_pred']
    tb_mae = df["ys_target"].abs().mean()
    tb_mse = (df["ys_target"] ** 2).mean()
    tb_rmse = np.sqrt((df["ys_target"] ** 2).mean())
    print(f"Target MAE: {tb_mae:.4f} MSE: {tb_mse:.4f} RMSE: {tb_rmse:.4f}")
    ta_mae = df["y_pred"].abs().mean()
    ta_mse = (df["y_pred"] ** 2).mean()
    ta_rmse = np.sqrt((df["y_pred"] ** 2).mean())
    print(f"SR     MAE: {ta_mae:.4f} MSE: {ta_mse:.4f} RMSE: {ta_rmse:.4f}")
    diff_mae = df["diff"].abs().mean()
    diff_mse = (df["diff"] ** 2).mean()
    diff_rmse = np.sqrt((df["diff"] ** 2).mean())
    print(f"Diff   MAE: {diff_mae:.4f} MSE: {diff_mse:.4f} RMSE: {diff_rmse:.4f}")

    # df['ys'] /= locald4.hartree_to_kcalmol
    print(df[["ys", "d4_ATM_E", "y_pred", "diff"]].describe())
    if False:
        # out = selected.replace(".pkl", "_SR.pkl")
        out = "/theoryfs2/ds/amwalla3/projects/symbolic_regression/sr/data/schr_dft2_SR.pkl"
        print(f"Saving to...\n{out}")
        df.to_pickle(out)

    df.sort_values("diff", inplace=True, ascending=False)
    cnt = 0
    for n, r in df.iterrows():
        line = f"{n}    START: {r['ys_target']:.4f} = {r['Benchmark']:.4f} - ({r[target_HF_key]:.4f} + {r['d4_2B']:.4f}) ::: fmp * {r['d4_ATM_E']:.4f}"
        print(line)
        line = f"{n}    y_target: {r['ys_target']:.4f}, y_pred: {r['y_pred']:.4f}, diff: {r['ys_target'] - r['y_pred']:.4f}"
        print(line)
        tools.print_cartesians(r['Geometry'])
        if cnt > 10:
            break
        cnt += 1
    return

def compute_mae_mse_rmse(df, y_pred):
    mae = df[y_pred].abs().mean()
    mse = (df[y_pred] ** 2).mean()
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def error_statistics_SR(
    df,
    selected,
    target_HF_key="SAPT0_adz_3_IE",
    ncols=8,
    generate=True,
    params_key="SAPT0_adz_3_IE_ATM",
):
    r = df.iloc[0]
    params = paramsTable.get_params(params_key)
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params, force_ATM_on=True)
    print(f"{params_2B = }")
    print(f"{params_ATM = }")
    df[["SR_ATM", "xs_all"]] = df.apply(
        lambda r: build_vals_molecule(
            r,
            params_ATM,
            cols=ncols,
        ),
        axis=1,
        result_type="expand",
    )
    params_ATM[-1] = 0.0
    df["d4_2B"] = df.apply(
        lambda r: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            r,
            params_2B,
            params_ATM,
        ),
        axis=1,
    )
    df["y_pred"] = df.apply(
        lambda r: (r["Benchmark"] - (r[target_HF_key] + r["d4_2B"] + sum(r["SR_ATM"]))),
        axis=1,
    )
    df['y_2b'] = df.apply(
        lambda r: (r["Benchmark"] - (r[target_HF_key] + r["d4_2B"])),
        axis=1,
    )
    print(df[['Benchmark', 'y_2b', 'y_pred']].describe())
    print(df[['Benchmark', target_HF_key, 'd4_2B', 'y_2b', 'y_pred']].head(10))
    df_tr, df_te = train_test_split(df, test_size=0.2, random_state=42)
    print(len(df_tr), len(df_te))

    tr_mae_2b, tr_mse_2b, tr_rmse_2b = compute_mae_mse_rmse(df_tr, "y_2b")
    te_mae_2b, te_mse_2b, te_rmse_2b = compute_mae_mse_rmse(df_te, "y_2b")
    print("2B")
    print(f"Train MAE: {tr_mae_2b:.4f} RMSE: {tr_rmse_2b:.4f} MSE: {tr_mse_2b:.4f}")
    print(f"Test  MAE: {te_mae_2b:.4f} RMSE: {te_rmse_2b:.4f} MSE: {tr_mse_2b:.4f}")
    tr_mae, tr_mse, tr_rmse = compute_mae_mse_rmse(df_tr, "y_pred")
    te_mae, te_mse, te_rmse = compute_mae_mse_rmse(df_te, "y_pred")
    print("ATM")
    print(f"Train MAE: {tr_mae:.4f} RMSE: {tr_rmse:.4f} MSE: {tr_mse:.4f}")
    print(f"Test  MAE: {te_mae:.4f} RMSE: {te_rmse:.4f} MSE: {tr_mse:.4f}")
    return

def evaluate_SR_function(
    row,
    syms,
    eq,
    x_col="xs",
):
    """
    Takes a string equation and evaluates it for a given row of data

    NOTE: the equation is DEPENDENT on the values of the columns in the xs column
    """
    for i in range(0, len(row[x_col])):
        xs = row[x_col][i]
        subs = {}
        for s, x in zip(syms, xs):
            subs[s] = x
        val = eq.evalf(subs=subs)
    return val


def evaluate_vals_sympy(
    df,
    ncols=7,
    equation: str = "((((x1 / (x1 + x4)) / x4) / x4) * x6)",
):
    from sympy import symbols, simplify, sympify
    from tqdm import tqdm

    tqdm.pandas()
    string_symbols = " ".join([f"x{i}" for i in range(1, ncols + 1)])
    syms = symbols(string_symbols)
    eq = sympify(equation)
    df["SR_ys"] = df.progress_apply(lambda r: evaluate_SR_function(r, syms, eq), axis=1)
    df["target"] = df["Benchmark"] - df["ys"]
    df["results"] = df["SR_ys"] - df["target"]
    print(df[["SR_ys", "target", "Benchmark", "ys"]])
    print(df["results"].describe())
    return


def evaluate_vals(df):
    df["SR_ys"] = df["xs"].apply(lambda x: x[:, 6])
    df["target"] = df["Benchmark"] - df["ys"]
    df["results"] = df["SR_ys"] - df["target"]
    print(df[["SR_ys", "target", "Benchmark", "ys"]])
    print(df["results"].describe())
    return


def compute_disp_2B_BJ_ATM_SR_dimer(r, params_ATM, max_N=None, cols=7):
    pos = r["Geometry_bohr"][:, 0]
    carts = r["Geometry_bohr"][:, 1:]
    monAs = r["monAs"]
    monBs = r["monBs"]
    C6s = r["C6s"]
    C6s_A = r["C6_A"]
    C6s_B = r["C6_B"]

    pos = np.array(pos, dtype=np.int32)
    pA, cA = pos[monAs].copy(), carts[monAs].copy()
    pB, cB = pos[monBs].copy(), carts[monBs].copy()
    # Generate ATM data for SR
    dimer_vals = build_vals(pos, carts, C6s, params_ATM, cols=cols, max_N=max_N)
    monA_vals = build_vals(pA, cA, C6s_A, params_ATM, cols=cols, max_N=max_N)
    monB_vals = build_vals(pB, cB, C6s_B, params_ATM, cols=cols, max_N=max_N)
    # Labeling as dimer and monomers for SR subtraction to get IE
    dimer_vals[:, 0] *= 1
    monA_vals[:, 0] *= -1
    monB_vals[:, 0] *= -1
    vals = np.concatenate((dimer_vals, monA_vals, monB_vals), axis=0)
    return vals, energies


# def compute_disp_2B_BJ_ATM_SR_dimer(
#     r,
#     params_2B,
#     params_ATM,
#     mult_out=hartree_to_kcalmol,
#     SR_func=disp.disp_SR_1,
# ):
#     pos, carts = (
#         np.array(r["Geometry_bohr"][:, 0], dtype=np.int32),
#         r["Geometry_bohr"][:, 1:],
#     )
#     charges = r["charges"]
#     monAs, monBs = r["monAs"], r["monBs"]
#     pA, cA = pos[monAs], carts[monAs, :]
#     pB, cB = pos[monBs], carts[monBs, :]
#     e_d = SR_func(
#         pos,
#         carts,
#         r["C6_ATM"],
#         params_ATM,
#     )
#     e_1 = SR_func(
#         pA,
#         cA,
#         r["C6_ATM_A"],
#         params_ATM,
#     )
#     e_2 = SR_func(
#         pB,
#         cB,
#         r["C6_ATM_B"],
#         params_ATM,
#     )
#     e_total = e_d - (e_1 + e_2)
#     e_total *= mult_out
#     return e_total
