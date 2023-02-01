import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qcelemental import constants

from .setup import (
    compute_bj_from_dimer_AB_all_C6s_NO_DAMPING,
    compute_bj_from_dimer_AB_all_C6s,
    compute_bj_from_dimer_AB_all_C6s_dimer_only,
    calc_dftd4_props_params,
)

def failed(
    df,
) -> None:
    """
    failed
    """
    df = pd.read_pickle("data/schr_dft.pkl")
    print(df.columns.values)
    t = df[df["pbe0_adz_saptdft"].isna()]
    print(len(t))
    # print(t)
    t2 = t[t["pbe0_adz_cation_A"].isna()]
    # t2 = t[t["pbe0_adz_grac_A"].isna()]
    print("cation_A")
    print(t2[["pbe0_adz_cation_A", "pbe0_adz_cation_B"]])
    print(t2["main_id"].to_list())
    # t3 = t[t["pbe0_adz_grac_B"].isna()]
    t3 = t[t["pbe0_adz_cation_B"].isna()]
    print("cation_B")
    print(t3[["pbe0_adz_cation_A", "pbe0_adz_cation_B"]])
    print(t3["main_id"].to_list())
    errors = list(set(t2["main_id"].to_list()).union(t3["main_id"].to_list()))
    print(errors)
    print(len(errors))


def df_empty_cols(df, col="pbe0_adz_grac_A") -> None:
    """
    df_empty_gracs
    """
    a = df[df[col].isna()]
    print(len(a))
    return


def plot_BJ_damping(s6, s8, a1, a2, R_0) -> None:
    """
    plot_BJ_damping
    """
    xs = np.arange(0.01, 5, 0.01)
    damp_c6s = [s6 * r / (r * a1 * (R_0 * a2))]

def compute_disp_3_forms(
    row,
    params=[1.61679827, 0.44959224, 3.35743605],  # HF
) -> None:
    """
    compute_disp_3_forms
    """
    mult_out=constants.conversion_factor("hartree", "kcal / mol")
    print("\tEnergies are in kcal/mol\n")
    print("Params", params, "\n")
    *_, grimme_d4_disp_e = calc_dftd4_props_params(
        row["Geometry"][:, 0], row["Geometry"][:, 1:], p=params
    )
    grimme_d4_disp_e *= mult_out
    print(f"{grimme_d4_disp_e = }")
    disp_no_damp = compute_bj_from_dimer_AB_all_C6s_NO_DAMPING(
        row["Geometry"][:, 0],  # pos
        row["Geometry"][:, 1:],  # carts
        row["monAs"],
        row["monBs"],
        row["C6s"],
        C6_A=row["C6_A"],
        C6_B=row["C6_B"],
    )
    print()
    print(f"{disp_no_damp = }")

    disp_damp_HF = compute_bj_from_dimer_AB_all_C6s(
        params,
        row["Geometry"][:, 0],  # pos
        row["Geometry"][:, 1:],  # carts
        row["monAs"],
        row["monBs"],
        row["C6s"],
        C6_A=row["C6_A"],
        C6_B=row["C6_B"],
    )
    print()
    print(f"{disp_damp_HF = }")
    disp_damp_HF_dimer_only = compute_bj_from_dimer_AB_all_C6s_dimer_only(
        params,
        row["Geometry"][:, 0],  # pos
        row["Geometry"][:, 1:],  # carts
        row["monAs"],
        row["monBs"],
        row["C6s"],
        C6_A=row["C6_A"],
        C6_B=row["C6_B"],
    )
    print()
    print(f"{disp_damp_HF_dimer_only = }")
    return
