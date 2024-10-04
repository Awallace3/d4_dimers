import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from src import paramsTable, locald4
from qm_tools_aw import tools
import os
from scipy.optimize import curve_fit
from src.plotting import prep_saptdft_components
from qcelemental import constants

h2kcalmol = constants.conversion_factor("hartree", "kcal/mol")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)


def df_setup(df=None, ddft=False,
             functionals=[
        "pbe0",
        "b2plyp",
        "b3lyp",
        "wb97x",
                 ],
             basis_sets=["adz", "atz"],
             ):
    # pp(df.columns.values.tolist())
    if os.path.exists("./plots/ddft_curves.pkl"):
        return pd.read_pickle("./plots/ddft_curves.pkl")
    if df is None:
        raise ValueError("No dataframe provided")
    p_2b, p_atm = paramsTable.param_lookup("sadz_supra")
    print(p_2b, p_atm)
    df["Geometry"] = df.apply(lambda r: np.array(r["Geometry"]), axis=1)
    df["monAs"] = df.apply(lambda r: np.array(r["monAs"]), axis=1)
    df["monBs"] = df.apply(lambda r: np.array(r["monBs"]), axis=1)
    df["distance (A)"] = df.apply(
        lambda r: tools.closest_intermolecular_contact_dimer(
            r["Geometry"], r["monAs"], r["monBs"]
        ),
        axis=1,
    )
    # df["d4_supra"] = df.apply(
    #     lambda row: locald4.compute_disp_2B_BJ_dimer_supra(
    #         row,
    #         p_2b,
    #         p_atm,
    #     ),
    #     axis=1,
    # )
    # p_2b, p_atm = paramsTable.param_lookup("sadz")
    # df["d4_super"] = df.apply(
    #     lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
    #         row,
    #         p_2b,
    #         p_atm,
    #     ),
    #     axis=1,
    # )
    if ddft:
        df["d4_ddft"] = df["SAPT_DFT_pbe0_adz_d4_disp"]
    print(df["SAPT_DFT_b2plyp_atz"])

    def compute_residue(row):
        if row["SAPT_DFT_b2plyp_atz"]:
            return row["Benchmark"] - sum(row["SAPT_DFT_b2plyp_atz"][1:-1]) * h2kcalmol
        else:
            return None

    df["SAPT0_disp"] = df.apply(lambda r: r["SAPT0_adz"][-1], axis=1)
    df["E_res"] = df.apply(lambda r: r["Benchmark"] -
                           sum(r["SAPT0_adz"][1:-1]), axis=1)
    if ddft:
        df["E_res_saptdft_b2plyp_atz"] = df.apply(
            lambda r: compute_residue(r), axis=1)
        df["E_ref_hlsapt_atz"] = df.apply(
            lambda r: r["SAPT2+(3)(CCD)DMP2 DISP ENERGY atz"] * h2kcalmol, axis=1
        )
    for functional in functionals:
        for basis_set in basis_sets:
            df = prep_saptdft_components(df, functional, basis_set)
            df[f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""] = df[f"""{
                functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""] * h2kcalmol
    df.to_pickle("./plots/ddft_curves.pkl")
    return df


def function_A_div_r6(r, A):
    return A / r**6


def function_A_div_r6_B_div_r8(r, A, B):
    return A / r**6 + B / r**8


def plot_all_curves(df):
    print(df["DB"].unique())
    df_hbc6 = df[df["DB"] == "HBC1"]
    print(
        df_hbc6[
            ["SAPT0_disp", "d4_supra", "d4_super",
                "E_res", "System #", "distance (A)"]
        ]
    )
    # plt usetex
    for db in df["DB"].unique():
        print(db)
        if db.lower() == "achc":
            continue
        df_db = df[df["DB"] == db]
        sys_numbers = df_db["System #"].unique()
        if len(sys_numbers) > 0:
            if ddft:
                os.makedirs(f"./plots/disp_curves_ddft/{db}", exist_ok=True)
            else:
                os.makedirs(f"./plots/disp_curves/{db}", exist_ok=True)
            for i in sys_numbers:
                df_sys = df_db[df_db["System #"] == i]
                if len(df_sys) < 4:
                    continue
                print(
                    df_sys[
                        [
                            "SAPT0_disp",
                            "d4_supra",
                            "d4_super",
                            "E_res",
                            "System #",
                            "distance (A)",
                        ]
                    ]
                )
                df_sys = df_sys.sort_values("distance (A)")
                fig = plt.figure(dpi=400)
                plt.plot(
                    df_sys["distance (A)"],
                    df_sys["d4_supra"],
                    label=f"-D4 Non-Super",
                    marker="o",
                    markersize=2.0,
                )
                plt.plot(
                    df_sys["distance (A)"],
                    df_sys["d4_super"],
                    label=f"-D4 Super",
                    marker="o",
                    markersize=2.0,
                )
                plt.plot(
                    df_sys["distance (A)"],
                    df_sys["SAPT0_disp"],
                    label=f"SAPT0/aDZ Disp.",
                    marker="o",
                    markersize=2.0,
                )
                plt.plot(
                    df_sys["distance (A)"],
                    df_sys["E_res"],
                    label=r"E_{res}",
                    marker="o",
                    markersize=2.0,
                    color="k",
                )
                plt.title(f"{db} System {i}")
                plt.xlabel("Distance (A)", fontsize=16)
                plt.ylabel("Energy (kcal/mol)", fontsize=16)
                plt.tick_params(axis="both", which="major", labelsize=14)
                plt.legend()
                if ddft:
                    plt.savefig(
                        f"./plots/disp_curves_ddft/{db}/{i}_ddft_super.png")
                else:
                    plt.savefig(f"./plots/disp_curves/{db}/{i}_d4_super.png")
                plt.clf()
    return


def plot_all_curves_LoS(
    df,
    plot_ddft_curve=True,
    functionals=[
        "pbe0",
        "b2plyp",
        "b3lyp",
        "wb97x",
    ],
    basis_sets=["adz", "atz"],
):
    print(df["DB"].unique())
    print(
        df[
            [
                "d4_ddft",
                "E_res",
                "system_id",
                "distance (A)",
            ]
        ]
    )
    # plt usetex
    dbs = df["DB"].unique()
    # dbs = ["s66x8"]
    for db in dbs:
        print(db)
        if db.lower() == "achc":
            continue
        df_db = df[df["DB"] == db]

        for functional in functionals:
            for basis_set in basis_sets:
                func_col = f"""{
                    functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                mae = np.mean(
                    np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                me = np.mean(
                    np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                print(f"""DB: {db} w {
                      functional}/{basis_set}, MAE: {mae:.2f} ME: {me:.2f}""")
        sys_numbers = df_db["System Label"].unique()
        if len(sys_numbers) > 0:
            os.makedirs(f"./plots/disp_curves_ddft/{db}", exist_ok=True)
            for n, i in enumerate(sys_numbers):

                df_sys = df_db[df_db["System Label"] == i]
                if len(df_sys) < 4:
                    continue
                print(db, i)
                # fit to 1/r^6 function for both SAPT(DFT)/aDZ and PBE0+dDFT+D4
                # popt, pcov = curve_fit(
                #     function_A_div_r6_B_div_r8,
                #     df_sys["distance (A)"],
                #     df_sys["SAPT_DFT_pbe0_adz_disp"],
                # )
                # A_saptdft_adz = popt[0]
                # B_saptdft_adz = popt[1]
                # popt, pcov = curve_fit(
                #     function_A_div_r6_B_div_r8,
                #     df_sys["distance (A)"],
                #     df_sys["d4_ddft"],
                # )
                # A_ddft = popt[0]
                # B_ddft = popt[1]

                df_sys = df_sys.sort_values("distance (A)")
                fig = plt.figure(dpi=400)
                for functional in functionals:
                    for basis_set in basis_sets:
                        func_col = f"""{
                            functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                        mae = np.mean(
                            np.abs(
                                df_sys[func_col]
                                - df_sys["E_ref_hlsapt_atz"]
                            )
                        )
                        me = np.mean(
                            df_sys[func_col]
                            - df_sys["E_ref_hlsapt_atz"]
                        )
                        plt.plot(
                            df_sys["distance (A)"],
                            df_sys[func_col],
                            label=f"""{
                                functional.upper()}-D4/{basis_set} MAE: {mae:.2f}, ME: {me:.2f}""",
                            marker="o",
                            markersize=2.0,
                        )
                plt.plot(
                    df_sys["distance (A)"],
                    df_sys["E_ref_hlsapt_atz"],
                    label=r"E$_{\rm res}^{SAPT(DFT)[B2PLYP]/aTZ}$",
                    marker="o",
                    markersize=2.0,
                    color="k",
                )
                plt.plot(
                    df_sys["distance (A)"],
                    df_sys["E_ref_hlsapt_atz"],
                    label=r"SAPT2+(3)(CCD)$\delta$MP2",
                    marker="o",
                    markersize=2.0,
                    color="k",
                )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     df_sys["d4_supra"],
                #     label=f"-D4 Non-Super",
                #     marker="o",
                #     markersize=2.0,
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     df_sys["d4_super"],
                #     label=f"-D4 Super",
                #     marker="o",
                #     markersize=2.0,
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     df_sys["SAPT0_disp"],
                #     label=f"SAPT0/aDZ Disp.",
                #     marker="o",
                #     markersize=2.0,
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     df_sys["SAPT_DFT_pbe0_adz_disp"],
                #     label=f"SAPT(DFT)/aDZ Disp.",
                #     marker="o",
                #     markersize=2.0,
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     function_A_div_r6_B_div_r8(
                #         df_sys["distance (A)"], A_saptdft_adz, B_saptdft_adz
                #     ),
                #     label=f"SAPT(DFT)/aDZ fit $\\frac{{{A_saptdft_adz:.2f}}}{{r^6}} + \\frac{{{B_saptdft_adz:.2f}}}{{r^8}}$",
                #     marker="o",
                #     markersize=2.0,
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     df_sys["SAPT_DFT_pbe0_atz_disp"],
                #     label=f"SAPT(DFT)/aTZ Disp.",
                #     marker="o",
                #     markersize=2.0,
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     df_sys["E_res"],
                #     label=r"E$_{\rm res}^{SAPT0}$",
                #     marker="o",
                #     markersize=2.0,
                #     # color="k",
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     df_sys["E_res_saptdft_atz"],
                #     label=r"E$_{\rm res}^{SAPT(DFT)/aTZ}$",
                #     marker="o",
                #     markersize=2.0,
                #     color="k",
                # )
                # plt.plot(
                #     df_sys["distance (A)"],
                #     function_A_div_r6_B_div_r8(df_sys["distance (A)"], A_ddft, B_ddft),
                #     label=f"PBE0-D4 Disp. fit $\\frac{{{A_ddft:.2f}}}{{r^6}} + \\frac{{{B_ddft:.2f}}}{{r^8}}$",
                #     marker="o",
                #     markersize=2.0,
                # )
                plt.title(f"{db} {i}")
                plt.xlabel("Distance (A)", fontsize=16)
                plt.ylabel("Energy (kcal/mol)", fontsize=16)
                plt.tick_params(axis="both", which="major", labelsize=14)
                plt.legend()
                plt.savefig(
                    f"./plots/disp_curves_ddft/{db}/{i}_ddft_super.png")
                plt.clf()
                if plot_ddft_curve:
                    popt, pcov = curve_fit(
                        function_A_div_r6_B_div_r8,
                        df_sys["distance (A)"],
                        df_sys["d4_ddft"],
                    )
                    A_ddft = popt[0]
                    B_ddft = popt[1]
                    # mae between d4_ddft and SAPT(DFT)/aDZ
                    mae = np.mean(
                        np.abs(df_sys["d4_ddft"] -
                               df_sys["SAPT_DFT_pbe0_adz_disp"])
                    )
                    me = np.mean(df_sys["d4_ddft"] -
                                 df_sys["SAPT_DFT_pbe0_adz_disp"])

                    df_sys = df_sys.sort_values("distance (A)")
                    fig = plt.figure(dpi=400)
                    # plt.plot(
                    #     df_sys["distance (A)"],
                    #     df_sys["SAPT_DFT_pbe0_atz_disp"],
                    #     label=f"SAPT(DFT)/aTZ Disp.",
                    #     marker="o",
                    #     markersize=2.0,
                    # )
                    # plt.plot(
                    #     df_sys["distance (A)"],
                    #     function_A_div_r6_B_div_r8(df_sys["distance (A)"], A_saptdft_adz, B_saptdft_adz),
                    #     label=f"SAPT(DFT)/aDZ fit $\\frac{{{A_saptdft_adz:.2f}}}{{r^6}} + \\frac{{{B_saptdft_adz:.2f}}}{{r^8}}$",
                    #     marker="o",
                    #     markersize=2.0,
                    # )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_adz_dDFT"]
                        - df_sys["SAPT_DFT_pbe0_adz_dHF"],
                        label=f"PBE0(dDFT)/aDZ - dHF/aDZ",
                        marker="o",
                        markersize=2.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_adz_dDFT"],
                        label=f"PBE0(dDFT)/aDZ",
                        marker="o",
                        markersize=2.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_adz_dHF"],
                        label=f"dHF/aDZ",
                        marker="o",
                        markersize=2.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_adz_D4_IE"],
                        label=f"-D4",
                        marker="o",
                        markersize=2.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["d4_ddft"],
                        label=f"PBE0-D4 Disp: ME {me:.2f} kcal/mol",
                        marker="o",
                        markersize=2.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_atz_disp"],
                        label=f"SAPT(DFT)/aTZ Disp: ME {0.00:.2f} kcal/mol",
                        marker="o",
                        markersize=2.0,
                        color="k",
                    )
                    # plt.plot(
                    #     df_sys["distance (A)"],
                    #     function_A_div_r6_B_div_r8(df_sys["distance (A)"], A_ddft, B_ddft),
                    #     label=f"PBE0+dDFT+D4 fit $\\frac{{{A_ddft:.2f}}}{{r^6}} + \\frac{{{B_ddft:.2f}}}{{r^8}}$",
                    #     marker="o",
                    #     markersize=2.0,
                    # )
                    # Annotate equation
                    # get location of middle of plot for text annotation
                    x = df_sys["distance (A)"].iloc[int(len(df_sys) / 2)]
                    x += 0.1 * x
                    y = df_sys["d4_ddft"].iloc[0]
                    y = 0.65 * y
                    # plt.text(
                    #     x,
                    #     y,
                    #     f"PBE0-D4 Disp. = PBE0(dDFT)/aDZ - dHF/aDZ + D4",
                    # )
                    plt.title(f"{db} {i}")
                    plt.xlabel("Distance (A)", fontsize=16)
                    plt.ylabel("Energy (kcal/mol)", fontsize=16)
                    plt.tick_params(axis="both", which="major", labelsize=14)
                    plt.legend(loc="lower right")
                    # fmt: off
                    plt.savefig(
                        f"""./plots/disp_curves_ddft/{db}/{i}_ddft_super_ddft_curve.png"""
                    )
                    # fmt: on
                    plt.clf()
                # if n > 5:
                #     break
                # break
    return

def subplot_all_curves_LoS(
    df,
    plot_ddft_curve=True,
    functionals=[
        "pbe0",
        "b2plyp",
        "b3lyp",
        "wb97x",
    ],
    basis_sets=["adz", "atz"],
):
    print(df["DB"].unique())
    print(
        df[
            [
                "d4_ddft",
                "E_res",
                "system_id",
                "distance (A)",
            ]
        ]
    )
    # plt usetex
    dbs = df["DB"].unique()
    # dbs = ["s66x8"]
    for db in dbs:
        print(db)
        if db.lower() == "achc":
            continue
        df_db = df[df["DB"] == db]

        for functional in functionals:
            for basis_set in basis_sets:
                func_col = f"""{
                    functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                mae = np.mean(
                    np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                me = np.mean(
                    np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                print(f"""DB: {db} w {
                      functional}/{basis_set}, MAE: {mae:.2f} ME: {me:.2f}""")
        sys_numbers = df_db["System Label"].unique()
        if len(sys_numbers) > 0:
            os.makedirs(f"./plots/disp_curves_ddft/{db}", exist_ok=True)
            for n, i in enumerate(sys_numbers):

                df_sys = df_db[df_db["System Label"] == i]
                if len(df_sys) < 4:
                    continue
                print(db, i)

                df_sys = df_sys.sort_values("distance (A)")
                fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=400, sharey=True, sharex=True)
                for functional in functionals:
                    # for basis_set in basis_sets:
                    basis_set = "adz"
                    func_col = f"""{
                        functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                    mae = np.mean(
                        np.abs(
                            df_sys[func_col]
                            - df_sys["E_ref_hlsapt_atz"]
                        )
                    )
                    me = np.mean(
                        df_sys[func_col]
                        - df_sys["E_ref_hlsapt_atz"]
                    )
                    axs[0].plot(
                        df_sys["distance (A)"],
                        df_sys[func_col],
                        label=f"""{
                            functional.upper()}-D4/{basis_set} MAE: {mae:.2f}, ME: {me:.2f}""",
                        marker="o",
                        markersize=2.0,
                    )
                axs[0].plot(
                    df_sys["distance (A)"],
                    df_sys["E_ref_hlsapt_atz"],
                    label=r"E$_{\rm res}^{SAPT(DFT)[B2PLYP]/aTZ}$",
                    marker="X",
                    markersize=5.0,
                    color="gray",
                )
                axs[0].plot(
                    df_sys["distance (A)"],
                    df_sys["E_ref_hlsapt_atz"],
                    label=r"SAPT2+(3)(CCD)$\delta$MP2",
                    marker="o",
                    markersize=2.0,
                    color="k",
                )
                axs[0].set_title(f"(A)")
                axs[0].set_xlabel("Distance (A)", fontsize=16)
                axs[0].set_ylabel("Energy (kcal/mol)", fontsize=16)
                axs[0].tick_params(axis="both", which="major", labelsize=14)
                axs[0].legend()
                popt, pcov = curve_fit(
                    function_A_div_r6_B_div_r8,
                    df_sys["distance (A)"],
                    df_sys["d4_ddft"],
                )
                A_ddft = popt[0]
                B_ddft = popt[1]
                # mae between d4_ddft and SAPT(DFT)/aDZ
                mae = np.mean(
                    np.abs(df_sys["d4_ddft"] -
                           df_sys["SAPT_DFT_pbe0_adz_disp"])
                )
                me = np.mean(df_sys["d4_ddft"] -
                             df_sys["SAPT_DFT_pbe0_adz_disp"])

                df_sys = df_sys.sort_values("distance (A)")
                axs[1].plot(
                    df_sys["distance (A)"],
                    df_sys["SAPT_DFT_pbe0_adz_dHF"],
                    label=r"$\delta$HF/aDZ",
                    marker="o",
                    markersize=2.0,
                )
                axs[1].plot(
                    df_sys["distance (A)"],
                    df_sys["SAPT_DFT_pbe0_adz_dDFT"]
                    - df_sys["SAPT_DFT_pbe0_adz_dHF"],
                    label=r"$\delta$DFT[PBE0]/aDZ - $\delta$HF/aDZ",
                    marker="o",
                    markersize=2.0,
                )
                axs[1].plot(
                    df_sys["distance (A)"],
                    df_sys["SAPT_DFT_pbe0_adz_dDFT"],
                    label=r"$\delta$DFT[PBE0]/aDZ",
                    marker="o",
                    markersize=2.0,
                )
                axs[1].plot(
                    df_sys["distance (A)"],
                    df_sys["SAPT_DFT_pbe0_adz_D4_IE"],
                    label="-D4[PBE0]",
                    marker="o",
                    markersize=2.0,
                )
                for functional in functionals:
                    axs[1].plot(
                        df_sys["distance (A)"],
                        df_sys[f"{functional.upper()}-D4 DISP ENERGY adz"],
                        label=f"{functional.upper()}-D4/adz disp.",
                        marker="o",
                        markersize=2.0,
                    )
                axs[1].plot(
                    df_sys["distance (A)"],
                    df_sys["E_ref_hlsapt_atz"],
                    label=r"SAPT2+(3)(CCD)$\delta$MP2 disp.",
                    marker="o",
                    markersize=2.0,
                    color="k",
                )
                axs[1].set_title(f"(B)")
                axs[1].set_xlabel("Distance (A)", fontsize=16)
                # axs[1].set_ylabel("Energy (kcal/mol)", fontsize=16)
                axs[1].tick_params(axis="both", which="major", labelsize=14)
                axs[1].legend(loc="lower right")
                # fmt: off
                plt.savefig(
                    f"""./plots/disp_curves_ddft/{db}/{i}_ddft_super_ddft_curve.png"""
                )
                # fmt: on
                plt.close()
                # if n > 5:
                #     break
                # break
    return


def plot_hbc6(df):
    print(df["DB"].unique())
    df_db = df[df["DB"] == "HBC1"]
    print(
        df_db[
            ["SAPT0_disp", "d4_supra", "d4_super",
                "E_res", "System #", "distance (A)"]
        ]
    )
    # plt usetex
    sys_numbers = df_db["System #"].unique()
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.flatten()
    if len(sys_numbers) > 0:
        os.makedirs(f"./plots/disp_curves/HBC1/", exist_ok=True)
        for n, i in enumerate(sys_numbers):
            print(n)
            df_sys = df_db[df_db["System #"] == i]
            print(
                df_sys[
                    [
                        "SAPT0_disp",
                        "d4_supra",
                        "d4_super",
                        "E_res",
                        "System #",
                        "distance (A)",
                    ]
                ]
            )
            df_sys = df_sys.sort_values("distance (A)")

            ax[n].plot(
                df_sys["distance (A)"],
                df_sys["E_res"],
                label=r"E$_{\rm res}$",
                marker="o",
                markersize=2.0,
                color="k",
            )
            ax[n].plot(
                df_sys["distance (A)"],
                df_sys["d4_supra"],
                label=f"-D4 Non-Super",
                marker="o",
                markersize=2.0,
            )
            ax[n].plot(
                df_sys["distance (A)"],
                df_sys["d4_super"],
                label=f"-D4 Super",
                marker="o",
                markersize=2.0,
            )
            ax[n].plot(
                df_sys["distance (A)"],
                df_sys["SAPT0_disp"],
                label=f"SAPT0 Disp.",
                marker="o",
                markersize=2.0,
            )
            sys_name = df_sys["System"].iloc[0]
            ax[n].set_title(f"{sys_name}")
            if n == 0:
                ax[n].legend(loc="lower right")
            if n % 3 == 0:
                ax[n].set_ylabel(
                    r"Disp. IE Energy (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14"
                )
            if n >= 3:
                ax[n].set_xlabel("Distance (\AA)", fontsize=14)
            ax[n].set_ylim([-17.5, 0.2])
            if n == 0:
                ax[n].set_ylim([-20, 0.2])
            ax[n].tick_params(axis="both", which="major", labelsize=14)
            # set minor ticks
            ax[n].minorticks_on()
            # plt.legend()
        plt.savefig(f"./plots/disp_curves/HBC1/HBC1_all_d4_super.png")
        plt.clf()
    return


def main():
    # df = pd.read_pickle("./plots/basis_study.pkl")
    # plot_hbc6(df)
    # plot_all_curves(df)

    # df = pd.read_pickle("./plots/ddft_study.pkl")
    df = df_setup(None, ddft=True)
    subplot_all_curves_LoS(df)
    return


if __name__ == "__main__":
    main()
