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


def build_vals_molecule(r, params, max_N=None):
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
    dimer_vals = build_vals(pos, carts, C6s, params, cols=7, max_N=max_N)
    monA_vals = build_vals(pA, cA, C6s_A, params, cols=7, max_N=max_N)
    monB_vals = build_vals(pB, cB, C6s_B, params, cols=7, max_N=max_N)
    # Labeling as dimer and monomers for SR subtraction to get IE
    dimer_vals[:, 0] = 1
    monA_vals[:, 0] = -1
    monB_vals[:, 0] = -1
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params)
    IE_2B = locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
        r,
        params_2B,
        params_ATM,
        mult_out=1.0,
    )
    dimer_vals[0, 6] = IE_2B
    return np.concatenate((dimer_vals, monA_vals, monB_vals), axis=0)


def generate_SR_data_ATM(df, selected):
    r = df.iloc[0]
    params = paramsTable.get_params("HF_ATM_OPT_START")
    df["xs"] = df.apply(
        lambda r: build_vals_molecule(
            r,
            params,
        ),
        axis=1,
    )
    df["splits"] = df["xs"].apply(len)
    print(df["splits"])
    df["ys"] = df["Benchmark"] / locald4.hartree_to_kcalmol
    out = selected.replace(".pkl", "_SR.pkl")
    df.to_pickle(out)
    return
