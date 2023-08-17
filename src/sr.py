import numpy as np
import pandas as pd
import dispersion
from . import paramsTable
from . import locald4


def build_vals(pos, carts, C6, params, cols=7, max_N=None):
    if max_N is None:
        max_N = len(pos)
    vals = np.zeros(
        (int(max_N * (max_N - 1) * (max_N - 2) / 6), cols), dtype=np.float64
    )
    dispersion.disp.vals_for_SR(pos, carts, C6, params, vals)
    return vals


def build_vals_molecule(r, params, max_N=None, cols=7):
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
    dimer_vals = build_vals(pos, carts, C6s, params, cols=cols, max_N=max_N)
    monA_vals = build_vals(pA, cA, C6s_A, params, cols=cols, max_N=max_N)
    monB_vals = build_vals(pB, cB, C6s_B, params, cols=cols, max_N=max_N)
    # Labeling as dimer and monomers for SR subtraction to get IE
    dimer_vals[:, 0] *= 1
    monA_vals[:, 0] *= -1
    monB_vals[:, 0] *= -1
    # params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params)
    # IE_2B = locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
    #     r,
    #     params_2B,
    #     params_ATM,
    #     mult_out=1.0,
    # )
    # dimer_vals[0, 6] = IE_2B
    return np.concatenate((dimer_vals, monA_vals, monB_vals), axis=0)


def generate_SR_data_ATM(df, selected, target_HF_key="HF_adz", ncols=6, generate=True):
    r = df.iloc[0]
    params = paramsTable.get_params("SAPT0_adz_3_IE_ATM")
    if generate:
        df["xs"] = df.apply(
            lambda r: build_vals_molecule(
                r,
                params,
                cols=ncols,
            ),
            axis=1,
        )
        for n, r in df.iterrows():
            df.at[n, "xs"][:, 0] *= locald4.hartree_to_kcalmol
        df["default_xs"] = df.apply(lambda r: sum(r["xs"][:, 0]), axis=1)
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
        params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params)

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

    df["ys_no_ATM"] = df.apply(
        lambda r: (r["Benchmark"] - (r[target_HF_key] + r["d4_2B"])),
        axis=1,
    )
    df["ys"] = df.apply(
        lambda r: (r["Benchmark"] - (r[target_HF_key] + r["d4_2B"])),
        axis=1,
    )
    for n, r in df.iterrows():
        # line = f"{n} ys = {r['ys']:.4f}"
        # print(line)
        # line = f"{n} NO ATM: {r['ys']:.4f} = {r['Benchmark']:.4f} - ({r[target_HF_key]:.4f} + {r['d4_2B']:.4f})"
        # print(line)
        line = f"{n}    ATM: {r['ys']:.4f} = {r['Benchmark']:.4f} - ({r[target_HF_key]:.4f} + {r['d4_2B']:.4f}) ::: fmp * {r['default_xs']:.4f}"
        print(line)
    # df['ys'] /= locald4.hartree_to_kcalmol
    print(df[["ys", 'default_xs']].describe())
    if generate:
        out = selected.replace(".pkl", "_SR.pkl")
        df.to_pickle(out)
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
