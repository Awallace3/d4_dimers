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

def df_setup():
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

    # p_2b, p_atm = paramsTable.param_lookup("SAPT0_adz_3_IE_2B_TT")
    # df["d4_supra_TT"] = df.apply(
    #     lambda row: locald4.compute_disp_2B_BJ_dimer_supra(
    #         row,
    #         p_2b,
    #         p_atm,
    #     ),
    #     axis=1,
    # )
    # p_2b, p_atm = paramsTable.param_lookup("sadz")
    # df["d4_super_TT"] = df.apply(
    #     lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
    #         row,
    #         p_2b,
    #         p_atm,
    #     ),
    #     axis=1,
    # )

    df["SAPT0_disp"] = df.apply(lambda r: r["SAPT0_adz"][-1], axis=1)
    df["E_res"] = df.apply(lambda r: r["Benchmark"] - sum(r["SAPT0_adz"][1:-1]), axis=1)
    return df


def plot_all_curves(df):
    print(df["DB"].unique())
    df_hbc6 = df[df["DB"] == "HBC1"]
    print(
        df_hbc6[
            ["SAPT0_disp", "d4_supra", "d4_super", "E_res", "System #", "distance (A)"]
        ]
    )
    # plt usetex
    for db in df["DB"].unique():
        print(db)
        if db.lower() == "achc":
            continue
        df_db = df[df["DB"] == db]
        sys_numbers = df_db['System #'].unique()
        if len(sys_numbers) > 0:
            os.makedirs(f"./plots/disp_curves/{db}", exist_ok=True)
            for i in sys_numbers:
                df_sys = df_db[df_db['System #'] == i]
                if len(df_sys) < 4:
                    continue
                print(df_sys[["SAPT0_disp", "d4_supra", "d4_super", "E_res", "System #", "distance (A)"]])
                df_sys = df_sys.sort_values("distance (A)")
                plt.plot(df_sys["distance (A)"], df_sys["d4_supra"], label=f"-D4 Non-Super", marker="o", markersize=2.0)
                plt.plot(df_sys["distance (A)"], df_sys["d4_super"], label=f"-D4 Super", marker="o", markersize=2.0)
                plt.plot(df_sys["distance (A)"], df_sys["SAPT0_disp"], label=f"SAPT0 Disp.", marker="o", markersize=2.0)
                plt.plot(df_sys["distance (A)"], df_sys["E_res"], label=r"E_{res}", marker="o", markersize=2.0)
                plt.title(f"{db} System {i}")
                plt.xlabel("Distance (A)", fontsize=16)
                plt.ylabel("Energy (kcal/mol)", fontsize=16)
                plt.tick_params(axis="both", which="major", labelsize=14)
                plt.legend()
                plt.savefig(f"./plots/disp_curves/{db}/{i}_d4_super.png")
                plt.clf()
    return

def plot_hbc6(df):
    print(df["DB"].unique())
    df_db = df[df["DB"] == "HBC1"]
    print(
        df_db[
            ["SAPT0_disp", "d4_supra", "d4_super", "E_res", "System #", "distance (A)"]
        ]
    )
    # plt usetex
    sys_numbers = df_db['System #'].unique()
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.flatten()
    if len(sys_numbers) > 0:
        os.makedirs(f"./plots/disp_curves/HBC1/", exist_ok=True)
        for n, i in enumerate(sys_numbers):
            print(n)
            df_sys = df_db[df_db['System #'] == i]
            print(df_sys[["SAPT0_disp", "d4_supra", "d4_super", "E_res", "System #", "distance (A)"]])
            df_sys = df_sys.sort_values("distance (A)")



            ax[n].plot(df_sys["distance (A)"], df_sys["d4_supra"], label=f"-D4 Non-Super", marker="o", markersize=2.0)
            ax[n].plot(df_sys["distance (A)"], df_sys["d4_super"], label=f"-D4 Super", marker="o", markersize=2.0)
            ax[n].plot(df_sys["distance (A)"], df_sys["SAPT0_disp"], label=f"SAPT0 Disp.", marker="o", markersize=2.0)
            ax[n].plot(df_sys["distance (A)"], df_sys["E_res"], label=r"E$_{\rm res}$", marker="o", markersize=2.0)
            sys_name = df_sys["System"].iloc[0]
            ax[n].set_title(f"{sys_name}")
            if n == 0:
                ax[n].legend(loc="lower right")
            if n % 3 == 0:
                ax[n].set_ylabel(r"Disp. IE Energy (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")
            if n >= 3:
                ax[n].set_xlabel("Distance (\AA)", fontsize=14)
            ax[n].set_ylim([-17.5, 0.2])
            if n ==0:
                ax[n].set_ylim([-20, 0.2])
            ax[n].tick_params(axis="both", which="major", labelsize=14)
            # set minor ticks
            ax[n].minorticks_on()
            # plt.legend()
        plt.savefig(f"./plots/disp_curves/HBC1/HBC1_all_d4_super.png")
        plt.clf()
    return


def main():
    df = df_setup()
    plot_hbc6(df)

    return


if __name__ == "__main__":
    main()
