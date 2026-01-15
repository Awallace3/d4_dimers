import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from src import paramsTable, locald4, plotting
from qm_tools_aw import tools
import os
from scipy.optimize import curve_fit
from src.plotting import prep_saptdft_components
from qcelemental import constants
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

h2kcalmol = constants.conversion_factor("hartree", "kcal/mol")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)

BLUE = "#1F77B4"  # BLUE -
GREEN = "#2CA02C"  # GREEN
LIME_GREEN = "#8CA83C"  # LIME GREEN
Aquamarine = "#7FFFD4"  # Aquamarine
Slate_Blue = "#6A5ACD"  # Slate Blue
# Functionals
TEAL = "#17BECF"  # TEAL - PBE0
LIGHT_BLUE = "#7F7FFF"  # LIGHT BLUE - B3LYP
Medium_Sea_Green = "#3CB371"  # Medium Sea Green - B2PLYP
INDIGO = "#7057FF"  # INDIGO - WB97X
color_map = {
    "PBE0": TEAL,
    "B3LYP": LIGHT_BLUE,
    "B2PLYP": Medium_Sea_Green,
    "WB97X": INDIGO,
}


def df_setup(
    df=None,
    ddft=False,
    functionals=[
        "pbe0",
        "b2plyp",
        "b3lyp",
        "wb97x",
    ],
    basis_sets=["adz", "atz"],
):
    # pp(df.columns.values.tolist())
    if os.path.exists("./plots/ddft_curves.pkl") and df is None:
        return pd.read_pickle("./plots/ddft_curves.pkl")
    elif df is None:
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
    df["E_res"] = df.apply(lambda r: r["Benchmark"] - sum(r["SAPT0_adz"][1:-1]), axis=1)
    if ddft:
        df["E_res_saptdft_b2plyp_atz"] = df.apply(lambda r: compute_residue(r), axis=1)
        df["E_ref_hlsapt_atz"] = df.apply(
            lambda r: r["SAPT2+3(CCD)DMP2 DISP ENERGY atz"] * h2kcalmol, axis=1
        )
    for functional in functionals:
        for basis_set in basis_sets:
            df = prep_saptdft_components(df, functional, basis_set)
            df[f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""] = (
                df[f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""]
                * h2kcalmol
            )
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
            ["SAPT0_disp", "d4_supra", "d4_super", "E_res", "System #", "distance (A)"]
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
                func_col = (
                    f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                )
                mae = np.mean(np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                me = np.mean(np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                print(
                    f"""DB: {db} w {functional}/{basis_set}, MAE: {mae:.2f} ME: {
                        me:.2f}"""
                )
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
                        func_col = f"""{functional.upper()}-D4 DISP ENERGY {
                            basis_set.lower()
                        }"""
                        mae = np.mean(
                            np.abs(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                        )
                        me = np.mean(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                        plt.plot(
                            df_sys["distance (A)"],
                            df_sys[func_col],
                            label=f"""{functional.upper()}-D4/{basis_set} MAE: {
                                mae:.2f}, ME: {me:.2f}""",
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
                    label=r"SAPT2+3(CCD)$\delta$MP2",
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
                plt.savefig(f"./plots/disp_curves_ddft/{db}/{i}_ddft_super.png")
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
                        np.abs(df_sys["d4_ddft"] - df_sys["SAPT_DFT_pbe0_adz_disp"])
                    )
                    me = np.mean(df_sys["d4_ddft"] - df_sys["SAPT_DFT_pbe0_adz_disp"])

                    df_sys = df_sys.sort_values("distance (A)")
                    fig = plt.figure(figsize=(4, 8), dpi=400)
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
                        markersize=3.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_adz_dDFT"],
                        label=f"PBE0(dDFT)/aDZ",
                        marker="o",
                        markersize=3.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_adz_dHF"],
                        label=f"dHF/aDZ",
                        marker="o",
                        markersize=3.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_adz_D4_IE"],
                        label=f"-D4",
                        marker="o",
                        markersize=3.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["d4_ddft"],
                        label=f"PBE0-D4 Disp: ME {me:.2f} kcal/mol",
                        marker="o",
                        markersize=3.0,
                    )
                    plt.plot(
                        df_sys["distance (A)"],
                        df_sys["SAPT_DFT_pbe0_atz_disp"],
                        label=f"SAPT(DFT)/aTZ Disp: ME {0.00:.2f} kcal/mol",
                        marker="o",
                        markersize=3.0,
                        color="k",
                    )
                    # plt.plot(
                    #     df_sys["distance (A)"],
                    #     function_A_div_r6_B_div_r8(df_sys["distance (A)"], A_ddft, B_ddft),
                    #     label=f"PBE0+dDFT+D4 fit $\\frac{{{A_ddft:.2f}}}{{r^6}} + \\frac{{{B_ddft:.2f}}}{{r^8}}$",
                    #     marker="o",
                    #     markersize=3.0,
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
    # plt usetex
    dbs = df["DB"].unique()
    print(dbs)
    # dbs = ["nbc10"]
    for db in dbs:
        print(db)
        if db.lower() == "achc":
            continue
        df_db = df[df["DB"] == db]

        for functional in functionals:
            for basis_set in basis_sets:
                func_col = (
                    f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                )
                mae = np.mean(np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                me = np.mean(np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                print(
                    f"""DB: {db} w {functional}/{basis_set}, MAE: {mae:.2f} ME: {
                        me:.2f}"""
                )
        sys_numbers = df_db["System Label"].unique()
        if len(sys_numbers) > 0:
            os.makedirs(f"./plots/disp_curves_ddft/{db}", exist_ok=True)
            for n, i in enumerate(sys_numbers):
                df_sys = df_db[df_db["System Label"] == i]
                if len(df_sys) < 4:
                    continue
                print(db, i)

                df_sys = df_sys.sort_values("distance (A)")
                n_basis_sets = len(basis_sets)
                fig, axs = plt.subplots(
                    n_basis_sets,
                    2,
                    figsize=(10, 5 * n_basis_sets),
                    dpi=400,
                    sharey=True,
                    sharex=True,
                )
                axs = axs.flatten()
                for n, basis_set in enumerate(basis_sets):
                    basis_set_label = f"{basis_set[0]}{basis_set[1:].upper()}"
                    mae = np.mean(
                        np.abs(
                            df_sys[
                                f"SAPT(DFT) [{functional.upper()}] DISP ENERGY {basis_set}"
                            ]
                            - df_sys["E_ref_hlsapt_atz"]
                        )
                    )
                    me = np.mean(
                        df_sys[
                            f"SAPT(DFT) [{functional.upper()}] DISP ENERGY {basis_set}"
                        ]
                        - df_sys["E_ref_hlsapt_atz"]
                    )
                    axs[n * 2].plot(
                        df_sys["distance (A)"],
                        df_sys[
                            f"SAPT(DFT) [{functional.upper()}] DISP ENERGY {basis_set}"
                        ]
                        * h2kcalmol,
                        label=r"SAPT(DFT)[PBE0]/"
                        + basis_set_label
                        + f" MAE: {mae:.2f}, ME: {me:.2f}",
                        marker="o",
                        markersize=2.0,
                    )

                    for functional in functionals:
                        func_col = f"""{functional.upper()}-D4 DISP ENERGY {
                            basis_set.lower()
                        }"""
                        mae = np.mean(
                            np.abs(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                        )
                        me = np.mean(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                        axs[n * 2].plot(
                            df_sys["distance (A)"],
                            df_sys[func_col],
                            label=f"""{functional.upper()}-D4/{basis_set_label} MAE: {
                                mae:.2f}, ME: {me:.2f}""",
                            marker="o",
                            markersize=2.0,
                        )
                    # axs[n * 2].plot(
                    #     df_sys["distance (A)"],
                    #     df_sys["E_ref_hlsapt_atz"],
                    #     label=r"E$_{\rm res}^{SAPT(DFT)[B2PLYP]/aTZ}$",
                    #     marker="X",
                    #     markersize=5.0,
                    #     color="gray",
                    # )
                    axs[n * 2].plot(
                        df_sys["distance (A)"],
                        df_sys["E_ref_hlsapt_atz"],
                        label=r"SAPT2+3(CCD)$\delta$MP2/aTZ",
                        marker="o",
                        markersize=2.0,
                        color="k",
                    )
                    axs[n * 2].set_title(f"(A)")
                    if n >= (n_basis_sets - 1) * 2 - 1:
                        axs[n * 2].set_xlabel("Distance (A)", fontsize=16)
                    axs[n * 2].set_ylabel(
                        f"{basis_set_label}\nEnergy (kcal/mol)", fontsize=16
                    )
                    axs[n * 2].tick_params(axis="both", which="major", labelsize=14)
                    axs[n * 2].legend(fontsize=8)
                    popt, pcov = curve_fit(
                        function_A_div_r6_B_div_r8,
                        df_sys["distance (A)"],
                        df_sys["d4_ddft"],
                    )
                    A_ddft = popt[0]
                    B_ddft = popt[1]
                    df_sys = df_sys.sort_values("distance (A)")
                    axs[n * 2 + 1].plot(
                        df_sys["distance (A)"],
                        df_sys[f"SAPT_DFT_pbe0_{basis_set}_dHF"],
                        label=r"$\delta$HF",  # + f"{basis_set_label}",
                        marker="o",
                        markersize=2.0,
                    )
                    axs[n * 2 + 1].plot(
                        df_sys["distance (A)"],
                        df_sys[f"SAPT_DFT_pbe0_{basis_set}_dDFT"]
                        - df_sys[f"SAPT_DFT_pbe0_{basis_set}_dHF"],
                        label=r"$\delta$DFT[PBE0] - $\delta$HF",
                        marker="o",
                        markersize=2.0,
                    )
                    axs[n * 2 + 1].plot(
                        df_sys["distance (A)"],
                        df_sys[f"SAPT_DFT_pbe0_{basis_set}_dDFT"],
                        label=r"$\delta$DFT[PBE0]",
                        marker="o",
                        markersize=2.0,
                    )
                    axs[n * 2 + 1].plot(
                        df_sys["distance (A)"],
                        df_sys[
                            f"SAPT(DFT) [{functional.upper()}] DISP ENERGY {basis_set}"
                        ]
                        * h2kcalmol,
                        label=r"SAPT(DFT)[PBE0]",
                        marker="o",
                        markersize=2.0,
                    )
                    axs[n * 2 + 1].plot(
                        df_sys["distance (A)"],
                        df_sys[f"SAPT_DFT_pbe0_{basis_set}_D4_IE"],
                        label="-D4[PBE0]",
                        marker="o",
                        markersize=2.0,
                    )
                    for functional in functionals:
                        axs[n * 2 + 1].plot(
                            df_sys["distance (A)"],
                            df_sys[f"{functional.upper()}-D4 DISP ENERGY {basis_set}"],
                            label=f"{functional.upper()}-D4/{basis_set_label} disp.",
                            marker="o",
                            markersize=2.0,
                        )
                    axs[n * 2 + 1].plot(
                        df_sys["distance (A)"],
                        df_sys["E_ref_hlsapt_atz"],
                        label=r"SAPT2+3(CCD)$\delta$MP2 disp.",
                        marker="o",
                        markersize=2.0,
                        color="k",
                    )
                    axs[n * 2 + 1].set_title(f"(B)")
                    if n >= (n_basis_sets - 1) * 2 - 1:
                        axs[n * 2 + 1].set_xlabel("Distance (A)", fontsize=16)
                    # axs[1].set_ylabel("Energy (kcal/mol)", fontsize=16)
                    axs[n * 2 + 1].tick_params(axis="both", which="major", labelsize=14)
                    axs[n * 2 + 1].legend(loc="lower right", fontsize=8)
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


def compute_N(df_l, col_E, sign_flip=True, print_lvl=0):
    df_l_neg = df_l[df_l[col_E] < 0]
    df_l_neg = df_l[df_l["R"] > 1.05]
    R = df_l_neg["distance (A)"]
    f_R = df_l_neg[col_E]
    if sign_flip:
        f_R = -f_R
    # Step 1: Take the logarithm of R and f(R)
    log_R = np.log(R)
    log_f_R = np.log(f_R)

    # Step 2: Perform linear regression on log_f_R vs. log_R
    # Calculate the slope (m) and intercept (b) using numpy's polyfit
    slope, intercept = np.polyfit(log_R, log_f_R, 1)

    # Step 3: Get N from the slope
    N = -slope  # Since log(f(R)) = -N * log(R), slope = -N
    if print_lvl > 0:
        print(f"{col_E:.12} N: {N:.2f}")
    return N


def subplot_all_curves_LoS_basis_set(
    df,
    plot_ddft_curve=True,
    functionals=[
        "pbe0",
        # "b2plyp",
        "b3lyp",
        # "wb97x",
    ],
    basis_sets=["adz", "atz"],
    build_pdf=True,
):
    # plt usetex
    dbs = df["DB"].unique()
    print(dbs)
    dbs = [
        "s66x8",
        # "hbc6",
        # "X4010",
    ]
    # dbs = ["nbc10"]
    tex_header = r"""
% arara: pdflatex
\documentclass{article}
\usepackage{graphicx} % For including images
\usepackage{adjustbox} % For adjusting image sizes
\usepackage{longtable}
\usepackage{chemformula}
\usepackage[margin=0.1in]{geometry}
\begin{document}
"""
    with open("./plots/disp_curves_ddft/LoS_disp_curves.tex", "w") as f:
        f.write(tex_header)
        for db in dbs:
            print(db)
            df_db = df[df["DB"] == db]
            f.write(f"\\section*{{{db}}}\n")
            # write a latex table for MAE and ME for each functional and basis set
            f.write("\\begin{table}[h!]\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Functional & Basis Set & MAE & ME \\\\\n")
            f.write("\\hline\n")
            # Error statistics
            # for method in ["SAPT0", "SAPT2+3(CCD)DMP2", "SAPT(DFT) [PBE0]", "SAPT(DFT) [B2PLYP]", "SAPT(DFT) [B3LYP]"]:
            for method in ["SAPT0", "SAPT2+3(CCD)DMP2", "SAPT(DFT) [PBE0]"]:
                for basis_set in basis_sets:
                    methbs = f"""{method} DISP ENERGY {basis_set.lower()}"""
                    print(methbs)
                    local_energies = df_db[methbs] * h2kcalmol
                    mae = np.mean(np.abs(local_energies - df["E_ref_hlsapt_atz"]))
                    me = np.mean(local_energies - df["E_ref_hlsapt_atz"])
                    print(f"{methbs}, MAE: {mae:.2f}, ME: {me:.2f}")
                    f.write(f"{method} & {basis_set} & {mae:.2f} & {me:.2f} \\\\")
            for functional in functionals:
                for basis_set in basis_sets:
                    func_col = (
                        f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                    )
                    mae = np.mean(np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                    me = np.mean(df_db[func_col] - df_db["E_ref_hlsapt_atz"])
                    print(
                        f"""DB: {db} w {functional}/{basis_set}, MAE: {mae:.2f} ME: {
                            me:.2f}"""
                    )
                    f.write(
                        f"{functional.upper()}-D4 & {basis_set} & {mae:.2f} & {me:.2f} \\\\"
                    )
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(
                "\\caption{Error statistics are in kcal/mol versus SAPT2+3(CCD)DMP2 DISP ENERGY atz}\n"
            )
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")
            f.write("\\clearpage\n")
            if db.lower() in ["achc", "ssi", "ion43"]:
                continue
            sys_numbers = df_db["System Label"].unique()
            if len(sys_numbers) > 0:
                os.makedirs(f"./plots/disp_curves_ddft/{db}", exist_ok=True)
                for n1, i in enumerate(sys_numbers):
                    df_sys = df_db[df_db["System Label"] == i]
                    print("sys:", df_sys["system_id"].iloc[0])
                    df_sys = df_sys.sort_values("distance (A)")
                    n_basis_sets = len(basis_sets)
                    fig, axs = plt.subplots(
                        n_basis_sets,
                        1,
                        figsize=(6.5, 5.5 * n_basis_sets),
                        dpi=400,
                        sharey=True,
                        sharex=True,
                    )
                    axs = axs.flatten()
                    for n, basis_set in enumerate(basis_sets):
                        basis_set_label = f"{basis_set[0]}{basis_set[1:].upper()}"
                        df_sys = df_sys.sort_values("distance (A)")
                        for n_func, functional in enumerate(functionals):
                            func_col = f"""{functional.upper()}-D4 DISP ENERGY {
                                basis_set.lower()
                            }"""
                            c = color_map[functional.upper()]

                            mae = np.mean(
                                np.abs(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                            )
                            me = np.mean(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                            if n_func == 0:
                                # N_neg = compute_N(
                                #     df_sys,
                                #     f"SAPT_DFT_{functional.lower()}_{basis_set}_D4_IE",
                                #     sign_flip=True,
                                # )
                                axs[n].plot(
                                    df_sys["distance (A)"],
                                    df_sys[
                                        f"SAPT_DFT_{functional.lower()}_{basis_set}_D4_IE"
                                    ],
                                    # label=rf"$E_{{\rm int}}^{{\rm D4,{functional.upper()}}}$ ($R^{{-{N_neg:.1f}}}$)",
                                    label=rf"$E_{{\rm int}}^{{\rm D4,{functional.upper()}}}$",
                                    marker="x",
                                    markersize=8.5,
                                    linestyle="-.",
                                    linewidth=2.5,
                                    color=c,
                                )
                                df_sys["dDFT - dHF"] = (
                                    df_sys[
                                        f"SAPT_DFT_{functional.lower()}_{basis_set}_dDFT"
                                    ]
                                    - df_sys[
                                        f"SAPT_DFT_{functional.lower()}_{basis_set}_dHF"
                                    ]
                                )
                                # N_neg = compute_N(df_sys, "dDFT - dHF", sign_flip=True)
                                axs[n].plot(
                                    df_sys["distance (A)"],
                                    df_sys["dDFT - dHF"],
                                    # label=r"$\delta$DFT[" + functional.upper() + r"] - $\delta$HF",
                                    # label=rf"$\delta_{{\rm DFT,{functional.upper()}}}^{{[2]}} - \delta_{{\rm HF}}^{{[2]}}$ ($R^{{-{N_neg:.1f}}}$)",
                                    label=rf"$\delta_{{\rm DFT,{functional.upper()}}}^{{[2]}} - \delta_{{\rm HF}}^{{[2]}}$",
                                    marker="x",
                                    linestyle="--",
                                    markersize=8.5,
                                    linewidth=2.5,
                                    color=c,
                                )
                            N_neg = compute_N(df_sys, func_col, sign_flip=True)
                            axs[n].plot(
                                df_sys["distance (A)"],
                                df_sys[func_col],
                                # label=f"""{
                                # functional.upper()}-D4 \\emph{{MAE: {mae:.2f}, ME: {me:.2f}}}""",
                                label=f"""{functional.upper()}-D4 ($R^{{-{N_neg:.1f}}}$)""",
                                marker="o",
                                markersize=4.0,
                                linewidth=2.5,
                                color=c,
                            )
                        sapt0_col = f"""SAPT0 DISP ENERGY {basis_set.lower()}"""
                        df_sys[sapt0_col] = df_sys[sapt0_col] * h2kcalmol
                        mae = np.mean(
                            np.abs(df_sys[sapt0_col] - df_sys["E_ref_hlsapt_atz"])
                        )
                        me = np.mean(df_sys[sapt0_col] - df_sys["E_ref_hlsapt_atz"])
                        N_neg = compute_N(df_sys, sapt0_col, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[sapt0_col],
                            label=f"SAPT0 ($R^{{-{N_neg:.1f}}}$)",
                            marker="o",
                            markersize=4.0,
                            color="orange",
                        )
                        func_col = (
                            f"SAPT(DFT) [{functional.upper()}] DISP ENERGY {basis_set}"
                        )
                        df_sys[func_col] = df_sys[func_col] * h2kcalmol
                        mae = np.mean(
                            np.abs(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                        )
                        me = np.mean(df_sys[func_col] - df_sys["E_ref_hlsapt_atz"])
                        N_neg = compute_N(df_sys, func_col, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[func_col],
                            label=rf"SAPT(PBE0) ($R^{{-{N_neg:.1f}}}$)",
                            marker="o",
                            markersize=4.5,
                            linewidth=2.0,
                            color="gray",
                        )
                        N_neg = compute_N(df_sys, "E_ref_hlsapt_atz", sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys["E_ref_hlsapt_atz"],
                            label=rf"SAPT2+3(CCD)/aTZ ($R^{{-{N_neg:.1f}}}$)",
                            marker="o",
                            markersize=4.5,
                            linewidth=2.0,
                            color="k",
                        )
                        axs[n].set_title(f"\\textbf{{{basis_set_label}}}", fontsize=20)
                        if n >= (n_basis_sets - 1) * 2 - 1:
                            axs[n].set_xlabel(r"Distance (\AA)", fontsize=16)
                        axs[n].set_ylabel(
                            f"Disp. Energy (kcal$\cdot$mol$^{-1}$)", fontsize=20
                        )
                        axs[n].tick_params(axis="both", which="major", labelsize=18)
                        axs[n].legend(loc="lower right", fontsize=18)
                        axs[n].yaxis.set_minor_locator(AutoMinorLocator())
                        axs[n].xaxis.set_minor_locator(AutoMinorLocator())
                        # make x-axis log scale
                        # axs[n].set_xscale('log')
                    # fmt: off
                    plt.tight_layout()
                    # plt.savefig(
                    #     f"""./plots/disp_curves_ddft/{db}/{i}_ddft_super_ddft_curve.png"""
                    # )
                    plt.savefig(
                        f"""./plots/disp_curves_ddft/{db}/{i}_ddft_super_ddft_curve.pdf"""
                    )
                    # fmt: on
                    plt.close()
                    # if n1 > 1:
                    #     break
                    # add figure to tex file
                    i_safe = i.replace("_", f"\\_")
                    # \\includegraphics[width=0.9\\textwidth]{{{db}/{i}_ddft_super_ddft_curve.png}}
                    f.write(
                        f"""\\begin{{figure}}[ht]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{{db}/{i}_ddft_super_ddft_curve.pdf}}
    \\caption{{LoS Dispersion Curves for \\textbf{{{db} {i_safe}}}}}.
\\end{{figure}}

\\clearpage

"""
                    )
        f.write(r"""\end{document}""")
    if build_pdf:
        os.chdir("./plots/disp_curves_ddft/")
        os.system("pdflatex LoS_disp_curves.tex")
        os.chdir("../../")
    return


def subplot_all_curves_LoS_basis_set_D4_versions(
    df,
    plot_ddft_curve=True,
    functionals=[
        "pbe0",
        # "b2plyp",
        # "b3lyp",
        # "wb97x",
    ],
    basis_sets=["adz", "atz"],
    build_pdf=True,
):
    # plt usetex
    dbs = df["DB"].unique()
    print(dbs)
    dbs = [
        "s66x8",
        # "hbc6",
        # "X4010",
    ]
    # dbs = ["nbc10"]
    tex_header = r"""
% arara: pdflatex
\documentclass{article}
\usepackage{graphicx} % For including images
\usepackage{adjustbox} % For adjusting image sizes
\usepackage{longtable}
\usepackage{chemformula}
\usepackage[margin=0.1in]{geometry}
\begin{document}
"""
    with open("./plots/disp_curves_ddft_d4/LoS_disp_curves.tex", "w") as f:
        f.write(tex_header)
        for db in dbs:
            print(db)
            df_db = df[df["DB"] == db]
            f.write(f"\\section*{{{db}}}\n")
            # write a latex table for MAE and ME for each functional and basis set
            f.write("\\begin{table}[h!]\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Functional & Basis Set & MAE & ME \\\\\n")
            f.write("\\hline\n")
            # Error statistics
            # for method in ["SAPT0", "SAPT2+3(CCD)DMP2", "SAPT(DFT) [PBE0]", "SAPT(DFT) [B2PLYP]", "SAPT(DFT) [B3LYP]"]:
            for method in ["SAPT0", "SAPT2+3(CCD)DMP2", "SAPT(DFT) [PBE0]"]:
                for basis_set in basis_sets:
                    methbs = f"""{method} DISP ENERGY {basis_set.lower()}"""
                    print(methbs)
                    local_energies = df_db[methbs] * h2kcalmol
                    mae = np.mean(np.abs(local_energies - df["E_ref_hlsapt_atz"]))
                    me = np.mean(local_energies - df["E_ref_hlsapt_atz"])
                    print(f"{methbs}, MAE: {mae:.2f}, ME: {me:.2f}")
                    f.write(f"{method} & {basis_set} & {mae:.2f} & {me:.2f} \\\\")
            for functional in functionals:
                for basis_set in basis_sets:
                    func_col = (
                        f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
                    )
                    mae = np.mean(np.abs(df_db[func_col] - df_db["E_ref_hlsapt_atz"]))
                    me = np.mean(df_db[func_col] - df_db["E_ref_hlsapt_atz"])
                    print(
                        f"""DB: {db} w {functional}/{basis_set}, MAE: {mae:.2f} ME: {
                            me:.2f}"""
                    )
                    f.write(
                        f"{functional.upper()}-D4 & {basis_set} & {mae:.2f} & {me:.2f} \\\\"
                    )
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(
                "\\caption{Error statistics are in kcal/mol versus SAPT2+3(CCD)DMP2 DISP ENERGY atz}\n"
            )
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")
            f.write("\\clearpage\n")
            if db.lower() in ["achc", "ssi", "ion43"]:
                continue
            sys_numbers = df_db["System Label"].unique()
            sys_numbers = sys_numbers[:3]
            if len(sys_numbers) > 0:
                os.makedirs(f"./plots/disp_curves_ddft_d4/{db}", exist_ok=True)
                for n1, i in enumerate(sys_numbers):
                    df_sys = df_db[df_db["System Label"] == i].copy()
                    print("sys:", df_sys["system_id"].iloc[0])
                    df_sys = df_sys.sort_values("distance (A)")
                    n_basis_sets = len(basis_sets)
                    fig, axs = plt.subplots(
                        n_basis_sets,
                        1,
                        figsize=(6, 6),
                        dpi=300,
                    )
                    if n_basis_sets == 1:
                        axs = [axs]
                    else:
                        axs = axs.flatten()

                    # Define colors and markers for each method
                    colors = {
                        "-D4 (HF_ATM)": "blue",
                        # "-D4 (SAPT_DFT_pbe0_adz_3_IE)": "red",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)": "green",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)": "purple",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)": "orange",
                        "PBD0-D4": "red",
                        "SAPT(PBE0)": "gray",
                        "E_ref_hlsapt_atz": "black",
                    }
                    markers = {
                        "-D4 (HF_ATM)": "o",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE)": "s",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)": "^",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)": "d",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)": "v",
                        "PBD0-D4": "*",
                        "SAPT(PBE0)": "X",
                        "E_ref_hlsapt_atz": "o",
                    }
                    labels = {
                        "-D4 (HF_ATM)": "HF-D4 (S)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE)": "SAPT(PBE0)-D4 (S)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)": "SAPT(PBE0)-D4 (I)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)": "SAPT(PBE0)-D4 (S, ND)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)": "SAPT(PBE0)-D4 (I, ND)",
                        "PBD0-D4": "PBE0-D4 DISP ENERGY adz",
                        "SAPT(PBE0)": "SAPT(PBE0)",
                        "E_ref_hlsapt_atz": "SAPT2+3(CCD)/aTZ",
                    }

                    for n, basis_set in enumerate(basis_sets):
                        basis_set_label = f"{basis_set[0]}{basis_set[1:].upper()}"
                        df_sys = df_sys.sort_values("distance (A)")

                        # Get equilibrium distance from minimum energy
                        func_col = f"SAPT(DFT) [pbe0] DISP ENERGY {basis_set}"
                        if func_col in df_sys.columns:
                            df_sys[func_col] = df_sys[func_col] * h2kcalmol
                            min_idx = df_sys[func_col].idxmin()
                            min_distance = df_sys.loc[min_idx, "distance (A)"]
                            axs[n].axvline(
                                min_distance,
                                color="grey",
                                linestyle="--",
                                label="Equilibrium",
                            )

                        # Plot each D4 method
                        d4_cols = [
                            "-D4 (HF_ATM)",
                            "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
                            "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)",
                            "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)",
                            "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)",
                        ]

                        for col in d4_cols:
                            if col not in df_sys.columns:
                                continue
                            # plot plot for data points
                            axs[n].plot(
                                df_sys["distance (A)"],
                                df_sys[col],
                                color=colors[col],
                                marker=markers[col],
                                label=f"{labels[col]} (Data)",
                            )
                            # Set y-limits based on data
                            axs[n].set_ylim(
                                df_sys[col].min() + 0.05 * df_sys[col].min(), 0.5
                            )

                        # Plot SAPT(PBE0) reference
                        if func_col in df_sys.columns:
                            axs[n].plot(
                                df_sys["distance (A)"],
                                df_sys[func_col],
                                color=colors["SAPT(PBE0)"],
                                marker=markers["SAPT(PBE0)"],
                                label=f"{labels['SAPT(PBE0)']} (Data)",
                            )

                        # Plot SAPT2+3(CCD) reference
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys["E_ref_hlsapt_atz"],
                            color=colors["E_ref_hlsapt_atz"],
                            marker=markers["E_ref_hlsapt_atz"],
                            label=f"{labels['E_ref_hlsapt_atz']} (Data)",
                        )

                        # Set plot properties
                        axs[n].set_title(f"{basis_set_label}")
                        axs[n].set_ylabel("Disp. Energy (kcal/mol)")
                        axs[n].grid(True, linestyle="--", alpha=0.7)
                        axs[n].minorticks_on()
                        axs[n].tick_params(which="both", width=1)
                        axs[n].legend(fontsize=10, loc="lower right")

                        # Format tick labels
                        axs[n].xaxis.set_major_formatter(ScalarFormatter())
                        axs[n].yaxis.set_major_formatter(ScalarFormatter())

                    axs[-1].set_xlabel(r"Distance (\AA)")
                    plt.tight_layout()
                    plt.savefig(
                        f"./plots/disp_curves_ddft_d4/{db}/{i}_ddft_super_ddft_curve.pdf"
                    )
                    plt.close()

                    # add figure to tex file
                    i_safe = i.replace("_", r"\_")
                    # \\includegraphics[width=0.9\\textwidth]{{{db}/{i}_ddft_super_ddft_curve.png}}
                    f.write(
                        f"""\\begin{{figure}}[ht]
    \\centering
    \\includegraphics[width=0.7\\textwidth]{{{db}/{i}_ddft_super_ddft_curve.pdf}}
    \\caption{{LoS Dispersion Curves for \\textbf{{{db} {i_safe}}}}}.
\\end{{figure}}

\\clearpage

"""
                    )
        f.write(r"""\end{document}""")
    if build_pdf:
        os.chdir("./plots/disp_curves_ddft_d4/")
        os.system("pdflatex LoS_disp_curves.tex")
        os.chdir("../../")
    return


def subplot_all_curves_LoS_basis_set_D4_versions_nondamped(
    df,
    plot_ddft_curve=True,
    functionals=[
        "pbe0",
    ],
    build_pdf=True,
):
    """
    Plot D4 dispersion curves with two subplots:
    - Top: ND (non-damped) curves vs reference and -D4 curves
    - Bottom: -D4(S), -D4(I), HF-D4, SAPT(PBE0)/aDZ, SAPT(PBE0)/aTZ, reference

    Uses only aTZ data for the reference.
    """
    dbs = df["DB"].unique()
    print(dbs)
    dbs = [
        "s66x8",
    ]
    tex_header = r"""
% arara: pdflatex
\documentclass{article}
\usepackage{graphicx}
\usepackage{adjustbox}
\usepackage{longtable}
\usepackage{chemformula}
\usepackage[margin=0.1in]{geometry}
\begin{document}
"""
    tick_fontsize = 14
    legend_fontsize = 12
    with open("./plots/disp_curves_ddft_d4/LoS_disp_curves_nd.tex", "w") as f:
        f.write(tex_header)
        for db in dbs:
            print(db)
            df_db = df[df["DB"] == db]
            f.write(f"\\section*{{{db}}}\n")
            f.write("\\clearpage\n")
            if db.lower() in ["achc", "ssi", "ion43"]:
                continue
            sys_numbers = df_db["System Label"].unique()
            # sys_numbers = sys_numbers[:3]
            if len(sys_numbers) > 0:
                os.makedirs(f"./plots/disp_curves_ddft_d4/{db}", exist_ok=True)
                for n1, i in enumerate(sys_numbers):
                    df_sys = df_db[df_db["System Label"] == i].copy()
                    print("sys:", df_sys["system_id"].iloc[0])
                    df_sys = df_sys.sort_values("distance (A)")

                    # Create 2x1 subplot (top: ND curves, bottom: damped curves)
                    fig, axs = plt.subplots(
                        2,
                        1,
                        figsize=(6, 7),
                        dpi=300,
                    )

                    # Define colors and markers for each method
                    colors = {
                        "-D4 (HF_ATM)": "blue",
                        "-D4 (HF)": "cyan",
                        "-D4 (SAPT0_adz_3_IE)": "purple",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE)": "teal",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)": "orange",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)": "teal",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)": "orange",
                        "SAPT(PBE0)/aDZ": "brown",
                        "SAPT(PBE0)/aTZ": "gray",
                        "E_ref_hlsapt_atz": "black",
                        "PBE0-D4 DISP ENERGY adz": "red",
                    }
                    markers = {
                        "-D4 (HF_ATM)": "o",
                        "-D4 (HF)": "o",
                        "-D4 (SAPT0_adz_3_IE)": "o",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE)": "d",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)": "d",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)": "^",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)": "^",
                        "SAPT(PBE0)/aDZ": "P",
                        "SAPT(PBE0)/aTZ": "P",
                        "E_ref_hlsapt_atz": "s",
                        "PBE0-D4 DISP ENERGY adz": "*",
                    }
                    labels = {
                        "-D4 (HF_ATM)": "HF-D4(ATM)",
                        "-D4 (HF)": "HF-D4",
                        "-D4 (SAPT0_adz_3_IE)": "SAPT0-D4(S)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE)": "SAPT(PBE0)-D4(S)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)": "SAPT(PBE0)-D4(I)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)": "SAPT(PBE0)-D4(S, ND)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)": "SAPT(PBE0)-D4(I, ND)",
                        "SAPT(PBE0)/aDZ": "SAPT(PBE0)/aDZ",
                        "SAPT(PBE0)/aTZ": "SAPT(PBE0)/aTZ",
                        "E_ref_hlsapt_atz": "SAPT2+3(CCD)/aTZ",
                        "PBE0-D4 DISP ENERGY adz": "PBE0-D4/aDZ",
                    }

                    # Prepare SAPT(PBE0) columns for aDZ and aTZ
                    sapt_adz_col = "SAPT(DFT) [PBE0] DISP ENERGY adz"
                    sapt_atz_col = "SAPT(DFT) [PBE0] DISP ENERGY atz"
                    if sapt_adz_col in df_sys.columns:
                        df_sys[sapt_adz_col] = df_sys[sapt_adz_col] * h2kcalmol
                    if sapt_atz_col in df_sys.columns:
                        df_sys[sapt_atz_col] = df_sys[sapt_atz_col] * h2kcalmol

                    # Get equilibrium distance from aTZ minimum energy
                    if sapt_atz_col in df_sys.columns:
                        min_idx = df_sys["SAPT2+3(CCD)DMP2 TOTAL ENERGY atz"].idxmin()
                        min_distance = df_sys.loc[min_idx, "distance (A)"]
                    else:
                        min_distance = None

                    # ===== TOP PLOT: ND curves vs reference and -D4 =====
                    ax_top = axs[0]

                    if min_distance is not None:
                        ax_top.axvline(
                            min_distance,
                            color="grey",
                            linestyle="--",
                            # label="Equilibrium",
                        )

                    # Plot ND curves
                    nd_cols = [
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)",
                    ]
                    for col in nd_cols:
                        if col in df_sys.columns:
                            ax_top.plot(
                                df_sys["distance (A)"],
                                df_sys[col],
                                color=colors[col],
                                marker=markers[col],
                                label=labels[col],
                            )

                    # Plot damped -D4 curves for comparison
                    d4_cols = [
                        # "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
                        # "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)",
                    ]
                    for col in d4_cols:
                        if col in df_sys.columns:
                            ax_top.plot(
                                df_sys["distance (A)"],
                                df_sys[col],
                                color=colors[col],
                                marker=markers[col],
                                label=labels[col],
                                alpha=0.5,
                            )

                    # Plot reference
                    ax_top.plot(
                        df_sys["distance (A)"],
                        df_sys["E_ref_hlsapt_atz"],
                        color=colors["E_ref_hlsapt_atz"],
                        marker=markers["E_ref_hlsapt_atz"],
                        label=labels["E_ref_hlsapt_atz"],
                    )

                    ax_top.text(
                        -0.12,
                        1.0,
                        "(A)",
                        transform=ax_top.transAxes,
                        fontsize=16,
                        fontweight="bold",
                        va="top",
                        ha="left",
                    )
                    ax_top.set_ylabel("Disp. Energy (kcal/mol)")
                    # ax_top.grid(True, linestyle="--", alpha=0.7)
                    ax_top.minorticks_on()
                    ax_top.tick_params(
                        which="both",
                        width=1,
                        labelsize=tick_fontsize,
                        direction="in",
                        top=True,
                        right=True,
                    )
                    ax_top.legend(fontsize=legend_fontsize, loc="lower right")
                    ax_top.xaxis.set_major_formatter(ScalarFormatter())
                    ax_top.yaxis.set_major_formatter(ScalarFormatter())

                    # ===== INSET: Zoom on Reference vs SAPT(PBE0)-D4(I, ND) =====
                    # Create inset in upper right, above the legend
                    ax_inset = inset_axes(
                        ax_top,
                        width="40%",
                        height="35%",
                        loc="upper right",
                        borderpad=1.5,
                    )

                    # Filter data up to equilibrium distance for inset
                    if min_distance is not None:
                        df_inset = df_sys[df_sys["distance (A)"] <= min_distance]
                    else:
                        df_inset = df_sys

                    # Plot SAPT(PBE0)-D4 (I, ND) in inset
                    i_nd_col = "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)"
                    if i_nd_col in df_inset.columns:
                        ax_inset.plot(
                            df_inset["distance (A)"],
                            df_inset[i_nd_col],
                            color=colors[i_nd_col],
                            marker=markers[i_nd_col],
                            label=labels[i_nd_col],
                            markersize=4,
                        )

                    # Plot reference in inset
                    ax_inset.plot(
                        df_inset["distance (A)"],
                        df_inset["E_ref_hlsapt_atz"],
                        color=colors["E_ref_hlsapt_atz"],
                        marker=markers["E_ref_hlsapt_atz"],
                        label=labels["E_ref_hlsapt_atz"],
                        markersize=4,
                    )

                    # Style the inset
                    ax_inset.tick_params(
                        labelsize=8,
                        direction="in",
                        top=True,
                        right=True,
                    )
                    ax_inset.set_xlabel("")
                    ax_inset.set_ylabel("")
                    ax_inset.minorticks_on()
                    # Add light box around inset
                    for spine in ax_inset.spines.values():
                        spine.set_edgecolor("gray")
                        spine.set_linewidth(0.8)

                    # Add rectangle and connecting lines to show inset region
                    # mark_inset draws a box on ax_top and lines to ax_inset
                    mark_inset(
                        ax_top,
                        ax_inset,
                        loc1=2,  # upper left corner of inset
                        loc2=3,  # lower left corner of inset
                        fc="none",
                        ec="gray",
                        linestyle="--",
                        linewidth=0.8,
                    )

                    # ===== BOTTOM PLOT: -D4(S), -D4(I), HF-D4, SAPT(PBE0), ref =====
                    ax_bot = axs[1]

                    if min_distance is not None:
                        ax_bot.axvline(
                            min_distance,
                            color="grey",
                            linestyle="--",
                            # label="Equilibrium",
                        )

                    # Plot damped D4 methods
                    damped_cols = [
                        "-D4 (HF_ATM)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
                        "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)",
                        "PBE0-D4 DISP ENERGY adz",
                        # "-D4 (HF)",
                        "-D4 (SAPT0_adz_3_IE)",
                    ]
                    for col in damped_cols:
                        if col in df_sys.columns:
                            ax_bot.plot(
                                df_sys["distance (A)"],
                                df_sys[col],
                                color=colors[col],
                                marker=markers[col],
                                label=labels[col],
                            )

                    # Plot SAPT(PBE0)/aDZ
                    if sapt_adz_col in df_sys.columns:
                        ax_bot.plot(
                            df_sys["distance (A)"],
                            df_sys[sapt_adz_col],
                            color=colors["SAPT(PBE0)/aDZ"],
                            marker=markers["SAPT(PBE0)/aDZ"],
                            label=labels["SAPT(PBE0)/aDZ"],
                        )

                    # Plot SAPT(PBE0)/aTZ
                    if sapt_atz_col in df_sys.columns:
                        ax_bot.plot(
                            df_sys["distance (A)"],
                            df_sys[sapt_atz_col],
                            color=colors["SAPT(PBE0)/aTZ"],
                            marker=markers["SAPT(PBE0)/aTZ"],
                            label=labels["SAPT(PBE0)/aTZ"],
                        )

                    # Plot reference
                    ax_bot.plot(
                        df_sys["distance (A)"],
                        df_sys["E_ref_hlsapt_atz"],
                        color=colors["E_ref_hlsapt_atz"],
                        marker=markers["E_ref_hlsapt_atz"],
                        label=labels["E_ref_hlsapt_atz"],
                    )

                    ax_bot.text(
                        -0.12,
                        1.0,
                        "(B)",
                        transform=ax_bot.transAxes,
                        fontsize=16,
                        fontweight="bold",
                        va="top",
                        ha="left",
                    )
                    ax_bot.set_xlabel(r"Distance (\AA)")
                    ax_bot.set_ylabel("Disp. Energy (kcal/mol)")
                    # ax_bot.grid(True, linestyle="--", alpha=0.7)
                    ax_bot.minorticks_on()
                    ax_bot.tick_params(
                        which="both",
                        width=1,
                        labelsize=tick_fontsize,
                        direction="in",
                        top=True,
                        right=True,
                    )
                    ax_bot.legend(fontsize=legend_fontsize, loc="lower right")
                    ax_bot.xaxis.set_major_formatter(ScalarFormatter())
                    ax_bot.yaxis.set_major_formatter(ScalarFormatter())

                    # Set consistent y-limits
                    ax_top.set_ylim(
                        ax_top.get_ylim()[0], max(ax_top.get_ylim()[1], 1.0)
                    )
                    ax_bot.set_ylim(
                        ax_bot.get_ylim()[0], max(ax_bot.get_ylim()[1], 1.0)
                    )

                    plt.tight_layout()
                    plt.savefig(
                        f"./plots/disp_curves_ddft_d4/{db}/{i}_nd_comparison.pdf"
                    )
                    plt.close()

                    # add figure to tex file
                    i_safe = i.replace("_", r"\_")
                    f.write(
                        f"""\\begin{{figure}}[ht]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{db}/{i}_nd_comparison.pdf}}
    \\caption{{\\textbf{{{db} {i_safe}}}:
        Dispersion potential Energy curve for SAPT(PBE0) with aDZ/aTZ and -D4
        variants versus the reference SAPT2+3(CCD)/aug-cc-pVTZ dispersion
        energy.
        (A) highlights the completely non-damped (ND) D4 curves compared to the
        reference for both supermolecular (S) and intermolecular (I). The
        remaining $s_8=0.89529649$. (B) shows the damped -D4(S), -D4(I), HF-D4,
        SAPT(PBE0)/aDZ, SAPT(PBE0)/aTZ, and reference dispersion energies.
        Here, (S) and (I) refer to the supermolecular and intermolecular
        dispersion energy approaches, respectively.
        The ``ND'' states that no damping function is used. For the damped
        SAPT(PBE0)-D4(S) and SAPT(PBE0)-D4(I) terms, $a_1$, $a_2$, and $s_8$
        are fit to the residual CCSD(T)/CBS energies on the 4470 dimer dataset.
    }}.
\\end{{figure}}

\\clearpage

"""
                    )
        f.write(r"""\end{document}""")
    if build_pdf:
        os.chdir("./plots/disp_curves_ddft_d4/")
        os.system("pdflatex LoS_disp_curves_nd.tex")
        os.chdir("../../")
    return


def subplot_all_curves_water_benzene_functional_form(
    plot_ddft_curve=True,
    functionals=[
        "pbe0",
    ],
    basis_sets=["atz"],
    build_pdf=True,
    sys_labels=[
        "01_Water-Water",
        "54_Benzene-Water",
    ],
    db="s66x8",
):
    df = pd.read_pickle("./plots/ddft_curves.pkl")
    df = df[df["System Label"].isin(sys_labels)]
    # df = pd.read_pickle("./plots/ddft_study.pkl")
    pp(df.columns.tolist())
    for functional in functionals:
        for basis_set in basis_sets:
            # df[f'SAPT_DFT_{functional.lower()}_{basis_set}'] = df[f'SAPT_LP_DFT_RP__{basis_set}']
            # df[f'SAPT_DFT_{functional.lower()}_{basis_set}'] = df[f'SAPT_DFT__{basis_set}']
            df[f"SAPT_DFT_{functional.lower()}_{basis_set}_total"] = df.apply(
                lambda r: r[f"SAPT_DFT_{functional.lower()}_{basis_set}"][0], axis=1
            )
            df[f"SAPT_DFT_{functional.lower()}_{basis_set}_total"] = df.apply(
                lambda r: r[f"SAPT_DFT_{functional.lower()}_{basis_set}"][0], axis=1
            )
            print(df[f"SAPT_DFT_{functional.lower()}_{basis_set}_total"])
            df = prep_saptdft_components(df, functional, basis_set)
            df[f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""] = (
                df[f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""]
                * h2kcalmol
            )
    df["E_ref_hlsapt_atz"] = df.apply(
        lambda r: r["SAPT2+3(CCD)DMP2 DISP ENERGY atz"] * h2kcalmol, axis=1
    )
    for functional in functionals:
        for basis_set in basis_sets:
            func_col = f"""{functional.upper()}-D4 DISP ENERGY {basis_set.lower()}"""
            mae = np.mean(np.abs(df[func_col] - df["E_ref_hlsapt_atz"]))
            me = np.mean(df[func_col] - df["E_ref_hlsapt_atz"])
            print(
                f"""{functional}/{basis_set}, MAE: {mae:.2f} ME: {me:.2f}, count: {len(df)}"""
            )
            sys_numbers = df["System Label"].unique()
            pp(sys_numbers)
            if len(sys_numbers) > 0:
                os.makedirs(f"./plots/disp_curves_ddft_d4/", exist_ok=True)
                for n1, i in enumerate(sys_numbers):
                    print("Plotting system:", i)
                    df_sys = df[df["System Label"] == i]
                    if len(df_sys) == 0:
                        print("No data for system:", i)
                        continue
                    print("sys:", df_sys["system_id"].iloc[0])
                    df_sys = df_sys.sort_values("distance (A)")
                    n_basis_sets = len(basis_sets)
                    fig, axs = plt.subplots(
                        n_basis_sets,
                        1,
                        figsize=(6.5, 5.5 * n_basis_sets),
                        dpi=400,
                        sharey=True,
                        sharex=True,
                    )
                    # if len(axs) > 1:
                    if ans := isinstance(axs, np.ndarray):
                        axs = axs.flatten()
                    else:
                        axs = [axs]
                    for n, basis_set in enumerate(basis_sets):
                        basis_set_label = f"{basis_set[0]}{basis_set[1:].upper()}"
                        df_sys = df_sys.sort_values("distance (A)")
                        label = "-D4 (HF_ATM)"
                        # N_neg = compute_N(df_sys, label, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[label],
                            label=f"HF-D4 (S)",
                            marker="o",
                            markersize=4.0,
                            # color="orange",
                        )
                        label = "-D4 (SAPT_DFT_pbe0_adz_3_IE)"
                        # N_neg = compute_N(df_sys, label, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[label],
                            label=f"SAPT(PBE0)-D4 (S)",
                            marker="o",
                            markersize=4.0,
                            # color="orange",
                        )
                        label = "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)"
                        # N_neg = compute_N(df_sys, label, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[label],
                            label=f"SAPT(PBE0)-D4 (I)",
                            marker="o",
                            markersize=4.0,
                            # color="orange",
                        )
                        label = "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)"
                        # N_neg = compute_N(df_sys, label, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[label],
                            label=f"SAPT(PBE0)-D4 (S, ND)",
                            marker="o",
                            markersize=4.0,
                            # color="orange",
                        )
                        label = "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)"
                        # N_neg = compute_N(df_sys, label, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[label],
                            label=f"SAPT(PBE0)-D4 (I, ND)",
                            marker="o",
                            markersize=4.0,
                            # color="orange",
                        )
                        func_col = (
                            f"SAPT(DFT) [{functional.upper()}] DISP ENERGY {basis_set}"
                        )
                        df_sys[func_col] = df_sys[func_col] * h2kcalmol
                        # N_neg = compute_N(df_sys, func_col, sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys[func_col],
                            label=rf"SAPT(PBE0)",
                            marker="o",
                            markersize=4.5,
                            linewidth=2.0,
                            color="gray",
                        )
                        # N_neg = compute_N(df_sys, "E_ref_hlsapt_atz", sign_flip=True)
                        axs[n].plot(
                            df_sys["distance (A)"],
                            df_sys["E_ref_hlsapt_atz"],
                            label=rf"SAPT2+3(CCD)/aTZ",
                            marker="o",
                            markersize=4.5,
                            linewidth=2.0,
                            color="k",
                        )
                        axs[n].set_title(f"\\textbf{{{basis_set_label}}}", fontsize=20)
                        if n >= (n_basis_sets - 1) * 2 - 1:
                            axs[n].set_xlabel(r"Distance (\AA)", fontsize=16)
                        axs[n].set_ylabel(
                            f"Disp. Energy (kcal$\cdot$mol$^{-1}$)", fontsize=20
                        )
                        axs[n].tick_params(axis="both", which="major", labelsize=18)
                        axs[n].legend(loc="lower right", fontsize=18)
                        axs[n].yaxis.set_minor_locator(AutoMinorLocator())
                        axs[n].xaxis.set_minor_locator(AutoMinorLocator())
                        # make x-axis log scale
                        # axs[n].set_xscale('log')
                    # get y-axis limits
                    y_min = np.min([ax.get_ylim()[0] for ax in axs])
                    y_max = np.max([ax.get_ylim()[1] for ax in axs])
                    for ax in axs:
                        ax.set_ylim(y_min, 8)
                    # fmt: off
                    plt.tight_layout()
                    # plt.savefig(
                    #     f"""./plots/disp_curves_ddft/{db}/{i}_ddft_super_ddft_curve.png"""
                    # )
                    path = f"""./plots/disp_curves_ddft_d4/{db}/{i}_ddft_super_ddft_curve.pdf"""

                    plt.savefig(path)
                    print(path)
                    # fmt: on
                    plt.close()
                    # if n1 > 1:
                    #     break
                    # add figure to tex file
                    i_safe = i.replace("_", f"\\_")
                    # \\includegraphics[width=0.9\\textwidth]{{{db}/{i}_ddft_super_ddft_curve.png}}
    #                     f.write(
    #                         f"""\\begin{{figure}}[ht]
    #     \\centering
    #     \\includegraphics[width=0.7\\textwidth]{{{db}/{i}_ddft_super_ddft_curve.pdf}}
    #     \\caption{{LoS Dispersion Curves for \\textbf{{{db} {i_safe}}}}}.
    # \\end{{figure}}
    #
    # \\clearpage
    #
    # """
    #                     )
    #         f.write(r"""\end{document}""")
    # if build_pdf:
    #     os.chdir("./plots/disp_curves_ddft_d4/")
    #     os.system("pdflatex LoS_disp_curves.tex")
    #     os.chdir("../../")
    return


def monomer_C6s_from_dimer(dimer_C6s, monA_C6s, monB_C6s):
    dimer_monA_C6s = dimer_C6s[: len(monA_C6s), : len(monA_C6s)]
    dimer_monB_C6s = dimer_C6s[len(monA_C6s) :, len(monA_C6s) :]
    intermolecular_C6s = dimer_C6s[: len(monA_C6s), len(monA_C6s) :] * 2
    return dimer_monA_C6s, dimer_monB_C6s, intermolecular_C6s


def c6_change_mon_dimer(df, system_label="45_Ethyne-Pentane", print_lvl=1):
    df_sys = df[df["System Label"] == system_label]
    # df_42 = df[df['System Label'] == '42_Uracil-Cyclopentane']
    # df_sys = df[df["System Label"] == "01_Water-Water"]

    # Positive Dispersion
    # df_sys = df[df["System Label"] == "43_Uracil-Neopentane"]
    # Fig2, 45 SAPT(PBE0-D4(S)) goes positive but nondamped doesn't
    # df_sys = df[df["System Label"] == "45_Ethyne-Pentane"]
    # df_sys = df[df['System Label'] == '50_Benzene-Ethyne'].copy()
    df_sys.sort_values("distance (A)", inplace=True)
    # make df_sys use .4f format for floats
    pd.set_option("display.float_format", "{:.4f}".format)
    # df_sys = plotting.compute_d4_from_opt_params(
    #     df_sys,
    #     bases=[
    #         [
    #             "SAPT_DFT_pbe0_adz_total",
    #             "SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING",
    #             "SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING",
    #             # "pbe0",
    #             "SAPT_DFT_pbe0_adz_3_IE",
    #         ],
    #     ],
    #     benchmark_label="benchmark ref energy",
    #     disp_compute=locald4.compute_disp_2B_NO_DAMPING,
    # )
    print(df_sys[["system_id", "R", "distance (A)"]])
    print(
        df_sys[
            ["R", "-D4 (SAPT_DFT_pbe0_adz_3_IE)", "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)"]
        ]
    )
    print(
        df_sys[
            [
                "R",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)",
            ]
        ]
    )
    print(df_sys[["R", "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)"]])
    # params, _ = paramsTable.get_params("SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING")
    # params, _ = paramsTable.get_params("SAPT_DFT_pbe0_adz_3_IE")
    print(df_sys["System Label"].iloc[0])
    print("Energy in kcal/mol")
    print(df_sys[["R", "-D4 (SAPT0_adz_3_IE)", "-D4 (HF)", "-D4 (HF_ATM)"]])
    if False:
        for n, r in df_sys.iterrows():
            # print()
            # print(r["Geometry"])
            # print(r["Geometry"][r['monAs']])
            # print(r["Geometry"][r['monBs']])
            # print()
            e = locald4.compute_gd4(
                r["Geometry_bohr"][:, 0],
                r["Geometry_bohr"][:, 1:],
                r["monAs"],
                r["monBs"],
            )
            e_pbe0 = locald4.compute_gd4(
                r["Geometry_bohr"][:, 0],
                r["Geometry_bohr"][:, 1:],
                r["monAs"],
                r["monBs"],
                method="pbe0",
            )
            e_b3lyp = locald4.compute_gd4(
                r["Geometry_bohr"][:, 0],
                r["Geometry_bohr"][:, 1:],
                r["monAs"],
                r["monBs"],
                method="b3lyp",
            )
            print(
                f"{r['R']}, GD4 HF-D4(ATM): {e:.6f}, GD4 PBE0-D4(ATM): {e_pbe0:.6f}, B3LYP-D4(ATM) {e_b3lyp:.6f}"
            )
    params = "sadz"
    params_no_damping = paramsTable.get_params(
        "SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING"
    )[0]
    print(params_no_damping)
    params = "SAPT_DFT_pbe0_adz_3_IE"
    print(f"Using params: {params}")
    if params == "sadz":
        params = paramsTable.get_params(params)
    else:
        params, _ = paramsTable.get_params(params)
    for n, r in df_sys[::-1].iterrows():
        print(f"System: {r['system_id']}, R: {r['R']}")
        dimer_C6s = r["C6s"]
        monA_C6s = r["C6_A"]
        monB_C6s = r["C6_B"]
        dimer_monA_C6s, dimer_monB_C6s, intermolecular_C6s = monomer_C6s_from_dimer(
            dimer_C6s, monA_C6s, monB_C6s
        )
        dimer_geom = r["Geometry_bohr"][:, 1:]
        distance_matrix = np.linalg.norm(dimer_geom[:, np.newaxis] - dimer_geom, axis=2)
        monomer_distance_A = distance_matrix[: len(monA_C6s), : len(monA_C6s)]
        monomer_distance_B = distance_matrix[len(monA_C6s) :, len(monA_C6s) :]
        intermolecular_distance = distance_matrix[: len(monA_C6s), len(monA_C6s) :]
        for i in range(len(monA_C6s)):
            monA_C6s[i, i] = 0.0
            dimer_monA_C6s[i, i] = 0.0
            monomer_distance_A[i, i] = 1.0
        for i in range(len(monB_C6s)):
            monB_C6s[i, i] = 0.0
            dimer_monB_C6s[i, i] = 0.0
            monomer_distance_B[i, i] = 1.0
        avg_change_A = np.mean(dimer_monA_C6s - monA_C6s)
        avg_change_B = np.mean(dimer_monB_C6s - monB_C6s)
        # average C6 percentage change - need to make diagonals 1 first to avoid div by 0
        dimer_monA_C6s_diag1 = dimer_monA_C6s.copy()
        dimer_monB_C6s_diag1 = dimer_monB_C6s.copy()
        monA_C6s_diag1 = monA_C6s.copy()
        monB_C6s_diag1 = monB_C6s.copy()
        for i in range(len(monA_C6s)):
            dimer_monA_C6s_diag1[i, i] = 1.0
            monA_C6s_diag1[i, i] = 1.0
        for i in range(len(monB_C6s)):
            dimer_monB_C6s_diag1[i, i] = 1.0
            monB_C6s_diag1[i, i] = 1.0
        avg_pct_change_A = (
            np.mean((dimer_monA_C6s_diag1 - monA_C6s_diag1) / monA_C6s_diag1) * 100.0
        )
        avg_pct_change_B = (
            np.mean((dimer_monB_C6s_diag1 - monB_C6s_diag1) / monB_C6s_diag1) * 100.0
        )
        c6_diff_A = dimer_monA_C6s - monA_C6s
        c6_diff_B = dimer_monB_C6s - monB_C6s
        avg_change_A = np.mean(c6_diff_A)  # * constants.hartree2kcalmol
        avg_change_B = np.mean(c6_diff_B)  # * constants.hartree2kcalmol
        # * constants.hartree2kcalmol
        mae_change_A = np.mean(np.abs(c6_diff_A))
        mae_change_B = np.mean(np.abs(c6_diff_B))  # * constants.hart
        min_change_A = np.min(c6_diff_A)  # * constants.hartree2kcalmol
        min_change_B = np.min(c6_diff_B)  # * constants.hartree2kcalmol
        max_change_A = np.max(c6_diff_A)  # * constants.hartree2kcalmol
        max_change_B = np.max(c6_diff_B)
        if False:
            print(
                f"C6s*h2km    , avg change A: {avg_change_A:.2f}, avg change B: {avg_change_B:.2f}"
            )
            print("divided by 1/r^6")
            avg_change_A_r6 = (
                np.mean(
                    dimer_monA_C6s / monomer_distance_A**6
                    - monA_C6s / monomer_distance_A**6
                )
                * constants.hartree2kcalmol
            )
            avg_change_B_r6 = (
                np.mean(
                    dimer_monB_C6s / monomer_distance_B**6
                    - monB_C6s / monomer_distance_B**6
                )
                * constants.hartree2kcalmol
            )
            avg_change_inter_r6 = (
                np.mean(intermolecular_C6s / intermolecular_distance**6)
                * constants.hartree2kcalmol
            )
            print(
                f"C6s*h2km/r^6, avg change A: {avg_change_A_r6:.2f}, avg change B: {avg_change_B_r6:.2f}"
            )
            print("divided by 1/r^8")
            avg_change_A_r8 = (
                np.mean(
                    dimer_monA_C6s / monomer_distance_A**8
                    - monA_C6s / monomer_distance_A**8
                )
                * constants.hartree2kcalmol
            )
            avg_change_B_r8 = (
                np.mean(
                    dimer_monB_C6s / monomer_distance_B**8
                    - monB_C6s / monomer_distance_B**8
                )
                * constants.hartree2kcalmol
            )
            avgerage_inter_r8 = (
                np.mean(intermolecular_C6s / intermolecular_distance**8)
                * constants.hartree2kcalmol
            )
            print(
                f"C6s*h2km/r^8, avg change A: {avg_change_A_r8:.2f}, avg change B: {avg_change_B_r8:.2f}"
            )

        dimer_dispersion_supra = locald4.compute_disp_2B_supra_from_C6s(
            r["Geometry_bohr"][:, 0],
            r["Geometry_bohr"][:, 1:],
            dimer_C6s,
            r["monAs"],
            r["monBs"],
            params,
        )
        dimer_dispersion_supra_no_damping = (
            locald4.compute_disp_2B_supra_from_C6s_NO_DAMPING(
                r["Geometry_bohr"][:, 0],
                r["Geometry_bohr"][:, 1:],
                dimer_C6s,
                r["monAs"],
                r["monBs"],
                params_no_damping,
            )
        )
        dimer_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][:, 0],
            r["Geometry_bohr"][:, 1:],
            dimer_C6s,
            params_no_damping,
        )
        monA_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][: len(monA_C6s), 0],
            r["Geometry_bohr"][: len(monA_C6s), 1:],
            monA_C6s,
            params_no_damping,
        )
        monB_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][len(monA_C6s) :, 0],
            r["Geometry_bohr"][len(monA_C6s) :, 1:],
            monB_C6s,
            params_no_damping,
        )
        dimer_monA_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][: len(monA_C6s), 0],
            r["Geometry_bohr"][: len(monA_C6s), 1:],
            dimer_monA_C6s,
            params_no_damping,
        )
        dimer_monB_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][len(monA_C6s) :, 0],
            r["Geometry_bohr"][len(monA_C6s) :, 1:],
            dimer_monB_C6s,
            params_no_damping,
        )

        # Intermolecular contributions of no_damping

        bj = True
        non_damping = True
        print("===============  NOTE  ==================")
        print("dmonA/dmonB compute using dimer C6s subblocks")
        print("monA/monB compute using monomer C6s")
        print("change in C6s is defined as: dimer_monX_C6s - monX_C6s, X=A,B")
        print("=========================================")
        c6_sum_A = np.sum(monA_C6s)
        c6_sum_B = np.sum(monB_C6s)
        print(f"C6s sum, monA: {c6_sum_A:.2f}, monB: {c6_sum_B:.2f}")
        c6_sum_dimer_monA = np.sum(dimer_monA_C6s)
        c6_sum_dimer_monB = np.sum(dimer_monB_C6s)
        print(
            f"C6s sum, dmonA: {c6_sum_dimer_monA:.2f}, dmonB: {c6_sum_dimer_monB:.2f}"
        )
        print(
            f"C6s sum change A: {c6_sum_dimer_monA - c6_sum_A:.2f}, change B: {c6_sum_dimer_monB - c6_sum_B:.2f}"
        )
        print(
            f"C6s, avg change A: {avg_change_A:.2f}, avg change B: {avg_change_B:.2f}"
        )
        print(
            f"C6s, mae change A: {mae_change_A:.2f}, mae change B: {mae_change_B:.2f}"
        )
        print(
            f"C6s, min change A: {min_change_A:.2f}, min change B: {min_change_B:.2f}"
        )
        print(
            f"C6s, max change A: {max_change_A:.2f}, max change B: {max_change_B:.2f}"
        )
        print(
            f"C6s pct chg, avg pct change A: {avg_pct_change_A:.2f}%, avg pct change B: {avg_pct_change_B:.2f}%"
        )
        if non_damping:
            print("----- NO DAMPING -----")
            print(
                f"Disp.    dimer: {dimer_dispersion:.4f}, monA: {monA_dispersion:.4f}, monB: {monB_dispersion:.4f}"
            )
            print(
                f"Disp.    Total: {dimer_dispersion - monA_dispersion - monB_dispersion:.4f}"
            )
            # print("    Using only C6s from the MONOMERS for intramolecular pairs (should agree with supra)")
            # print(
            #     f"Disp.MC6 dimer: {dimer_dispersion_c6s_mon:.4f}, monA: {monA_dispersion:.4f}, monB: {monB_dispersion:.4f}"
            # )
            # print(
            #     f"Disp.MC6 Total: {dimer_dispersion_c6s_mon - monA_dispersion - monB_dispersion:.4f}"
            # )
            print(
                f"Disp.    dmonA: {dimer_monA_dispersion:.4f}, dmonB: {dimer_monB_dispersion:.4f}"
            )
            print(
                f"Disp.     monA: {monA_dispersion:.4f},  monB: {monB_dispersion:.4f}"
            )
            # Difference between monA_dispersion and dimer_monA_dispersion
            monA_diff = dimer_monA_dispersion - monA_dispersion
            monB_diff = dimer_monB_dispersion - monB_dispersion
            print(f"Disp.  delta A: {monA_diff:.4f}, delta B: {monB_diff:.4f}")
            # Sum changes
            print(f"Disp.  delta A+B: {monA_diff + monB_diff:.4f}")
            print(f"Disp.  Intermolecular: {dimer_dispersion_supra_no_damping:.4f}")
            print(
                f"Disp.  delta A+B+Intermolecular: {monA_diff + monB_diff + dimer_dispersion_supra_no_damping:.4f}"
            )
            print(f"{len(monA_C6s) = }, {len(monB_C6s) = }")
        if bj:
            print("----- BJ DAMPING -----")
            # Now for BJ
            dimer_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][:, 0], r["Geometry_bohr"][:, 1:], dimer_C6s, params
            )
            monA_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][: len(monA_C6s), 0],
                r["Geometry_bohr"][: len(monA_C6s), 1:],
                monA_C6s,
                params,
            )
            monB_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][len(monA_C6s) :, 0],
                r["Geometry_bohr"][len(monA_C6s) :, 1:],
                monB_C6s,
                params,
            )
            dimer_monA_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][: len(monA_C6s), 0],
                r["Geometry_bohr"][: len(monA_C6s), 1:],
                dimer_monA_C6s,
                params,
            )
            dimer_monB_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][len(monA_C6s) :, 0],
                r["Geometry_bohr"][len(monA_C6s) :, 1:],
                dimer_monB_C6s,
                params,
            )
            print(
                f"Disp.BJ dimer: {dimer_dispersion_BJ:.4f}, monA: {monA_dispersion_BJ:.4f}, monB: {monB_dispersion_BJ:.4f}"
            )
            print(
                f"Disp.BJ Total: {dimer_dispersion_BJ - monA_dispersion_BJ - monB_dispersion_BJ:.4f}"
            )
            print(
                f"Disp.BJ  dmonA: {dimer_monA_dispersion_BJ:.4f}, dmonB: {dimer_monB_dispersion_BJ:.4f}"
            )
            print(
                f"Disp.BJ   monA: {monA_dispersion_BJ:.4f},  monB: {monB_dispersion_BJ:.4f}"
            )
            delta_monA_BJ = dimer_monA_dispersion_BJ - monA_dispersion_BJ
            delta_monB_BJ = dimer_monB_dispersion_BJ - monB_dispersion_BJ
            print(f"Disp.BJ delta A: {delta_monA_BJ:.4f}, delta B: {delta_monB_BJ:.4f}")
            print(f"Disp.BJ delta A+B: {delta_monA_BJ + delta_monB_BJ:.4f}")
            print(f"Disp.BJ Intermolecular: {dimer_dispersion_supra:.4f}")
            print(
                f"Disp.BJ delta A+B+Intermolecular: {delta_monA_BJ + delta_monB_BJ + dimer_dispersion_supra:.4f}"
            )
            print(f"{len(monA_C6s) = }, {len(monB_C6s) = }")

        return
        if print_lvl == 0:
            params_damped, _ = paramsTable.param_lookup("sadz")
            params_undamped = [1.0, 1.0, 0.0, 0.0]
            t6_2, t8_2, energies = locald4.compute_bj_terms(
                dimer_geom[:, 0],
                dimer_geom[:, 1:],
                dimer_C6s,
                params=params_damped,
                # params=params_undamped,
                damping_2d=True,
            )
            print(sum(energies))
            print("dimer C6s")
            print(dimer_C6s)
            print("Monomer A C6s")
            print(monA_C6s)
            print("Dimer Monomer A C6s")
            print(dimer_monA_C6s)
            print("Monomer B C6s")
            print(monB_C6s)
            print("Dimer Monomer B C6s")
            print(dimer_monB_C6s)
            print("C6 change")
            print(dimer_monA_C6s - monA_C6s)
            print(dimer_monB_C6s - monB_C6s)
            monA_t6s = t6_2[: len(monA_C6s), : len(monA_C6s)]
            monB_t6s = t6_2[len(monA_C6s) :, len(monA_C6s) :]
            monA_t8s = t8_2[: len(monA_C6s), : len(monA_C6s)]
            monB_t8s = t8_2[len(monA_C6s) :, len(monA_C6s) :]
            print("Dimer t6s")
            print(t6_2)
            print("Monomer t6s")
            print(monA_t6s)
            print(monB_t6s)
            print("Monomer t8s")
            print(monA_t8s)
            print(monB_t8s)
        print()
    return


def main():
    # df = pd.read_pickle("./plots/basis_study.pkl")
    # plot_hbc6(df)
    # plot_all_curves(df)
    #
    # df = pd.read_pickle("./curves/ddft_curves_start.pkl")
    # df = df_setup(df, ddft=True)
    # subplot_all_curves_water_benzene_functional_form()
    # return
    df = df_setup(None, ddft=True)
    c6_change_mon_dimer(df, system_label="45_Ethyne-Pentane", print_lvl=1)
    c6_change_mon_dimer(df, system_label="43_Uracil-Neopentane", print_lvl=1)
    return
    # pp(df.columns.values.tolist())
    # print(df['SAPT(DFT) [PBE0] DISP ENERGY atz'])
    # pp(df.columns.values.tolist())
    # return
    # return
    # print(df['R'])
    # subplot_all_curves_LoS(df, basis_sets=["adz"])
    # subplot_all_curves_LoS_basis_set(df, basis_sets=["adz", "atz"])
    # subplot_all_curves_LoS_basis_set_D4_versions(
    #     df, basis_sets=["adz", "atz"], build_pdf=True
    # )

    # Precursors
    print(df["SAPT(DFT) [PBE0] DISP ENERGY atz"])
    df = plotting.prep_saptdft_components(df, "pbe0", "adz")
    df = plotting.prep_saptdft_components(df, "pbe0", "atz")
    df["SAPT(DFT) [PBE0] DISP ENERGY atz"] = (
        df["SAPT(DFT) [PBE0] DISP ENERGY atz"] * h2kcalmol
    )
    df["SAPT(DFT) [PBE0] DISP ENERGY adz"] = (
        df["SAPT(DFT) [PBE0] DISP ENERGY adz"] * h2kcalmol
    )
    df["PBE0-D4 DISP ENERGY adz"] = df["PBE0-D4 DISP ENERGY adz"] * h2kcalmol
    subplot_all_curves_LoS_basis_set_D4_versions_nondamped(df, build_pdf=True)
    return


if __name__ == "__main__":
    main()
