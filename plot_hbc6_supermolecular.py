import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from src import paramsTable, locald4
from qm_tools_aw import tools
import os

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)

def plot_hbc6_1():
    df = pd.read_pickle("./plots/basis_study.pkl")
    # pp(df.columns.values.tolist())
    p_2b, p_atm = paramsTable.param_lookup("sadz_supra")
    print(p_2b, p_atm)
    df["distance (A)"] = df.apply(
        lambda r: tools.closest_intermolecular_contact_dimer(
            r["Geometry"], r["monAs"], r["monBs"]
        ),
        axis=1,
    )

    df["d4_supra"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_dimer_supra(
            row,
            p_2b,
            p_atm,
        ),
        axis=1,
    )
    p_2b, p_atm = paramsTable.param_lookup("sadz")
    df["d4_super"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            p_2b,
            p_atm,
        ),
        axis=1,
    )
    df["SAPT0_disp"] = df.apply(lambda r: r["SAPT0_adz"][-1], axis=1)
    df["E_res"] = df.apply(lambda r: r["Benchmark"] - sum(r["SAPT0_adz"][1:-1]), axis=1)
    print(df["DB"].unique())
    df_hbc6 = df[df["DB"] == "HBC1"]
    print(
        df_hbc6[
            ["SAPT0_disp", "d4_supra", "d4_super", "E_res", "System #", "distance (A)"]
        ]
    )
    # plt usetex
    # for db in df["DB"].unique():
    sys_numbers = df_hbc6['System #'].unique()
    for i in sys_numbers:
        df_sys = df_hbc6[df_hbc6['System #'] == i]
        print(df_sys[["SAPT0_disp", "d4_supra", "d4_super", "E_res", "System #", "distance (A)"]])
        df_sys = df_sys.sort_values("distance (A)")
        plt.plot(df_sys["distance (A)"], df_sys["d4_supra"], label=f"-D4 Non-Super", marker="o", markersize=2.0)
        plt.plot(df_sys["distance (A)"], df_sys["d4_super"], label=f"-D4 Super", marker="o", markersize=2.0)
        plt.plot(df_sys["distance (A)"], df_sys["SAPT0_disp"], label=f"SAPT0 Disp.", marker="o", markersize=2.0)
        plt.plot(df_sys["distance (A)"], df_sys["E_res"], label=r"E_{res}", marker="o", markersize=2.0)
        plt.title(f"HBC6 System {i}")
        plt.xlabel("Distance (A)", fontsize=16)
        plt.ylabel("Error (kcal/mol)", fontsize=16)
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.legend()
        plt.savefig(f"./plots/hbc1/{i}_d4_super.png")
        plt.clf()
    return


def main():
    plot_hbc6_1()
    return


if __name__ == "__main__":
    main()
