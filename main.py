import numpy as np
import pandas as pd
import pickle
from pprint import pprint as pp
import src
import qcelemental as qcel
from qm_tools_aw import tools
import ase


def data_wrangling() -> None:
    """
    data_wrangling
    """
    df = pd.read_pickle("data/schr_dft.pkl")
    print(df.columns)
    pd.set_option("display.max_rows", None)
    df2 = df[df["DB"].isin(["HBC1", "NBC10"])]
    df2 = df[df["DB"].isin(["NBC10"])]
    systems = list(set(df2["System"].to_list()))
    print(systems)
    t1 = df2[df["System"] == "Pyridine Dimer S2 Configuration"]
    # print(t1[['System', 'System #']])
    print("[")
    for n, r in t1.iterrows():
        c1, c2 = r["charges"][1], r["charges"][2]
        print()
        print('[\n"""')
        print(f"{c1[0]} {c1[1]}")
        tools.print_cartesians(r["Geometry"][r["monAs"], :])
        print("--")
        print(f"{c2[0]} {c2[1]}")
        tools.print_cartesians(r["Geometry"][r["monBs"], :])
        print('"""\n],')
        atomic_numbers, atoms = [], []
        for i in r["Geometry"]:
            el, x, y, z = i
            atoms.append((x, y, z))
            atomic_numbers.append(el)
        atoms = ase.Atoms(symbols=atomic_numbers, positions=atoms)
        # print(atoms)
        # ase.io.write(f"images/{n}_geom.png", atoms)
    print("]")
    # v = tools.convert_schr_row_to_mol(df2.iloc[0])
    # df2.to_pickle("data/hbc1_nbc10.pkl")
    return


def optimize_paramaters(df, bases) -> None:
    # bases = [
    #     "TAG",
    #     # "HF_dz",
    #     # "HF_jdz",
    #     # "HF_adz",
    #     # "HF_tz",
    #     # "HF_atz",
    #     # "HF_jdz_no_cp",
    #     # "HF_dz_no_cp",
    #     # "HF_qz",
    #     # "HF_qz_no_cp",
    #     # "HF_qz_no_df",
    #     # "HF_qz_conv_e_4",
    #     # "pbe0_adz_saptdft_ndisp",
    # ]

    adz_opt_params = [0.829861, 0.706055, 1.123903]
    params = [1.61679827, 0.44959224, 3.35743605]
    params_d3 = [0.7683276390453782, 0.09699087897359535, 3.6407701963142745]
    params = src.paramsTable.paramsDict()["HF"][1:]
    for i in bases:
        print(i)
        # print("D3")
        # src.optimization.opt_cross_val(
        #     df,
        #     nfolds=5,
        #     start_params=params_d3,
        #     hf_key=i,
        #     output_l_marker="D3_",
        #     optimizer_func=src.jeff.optimization_d3,
        #     compute_int_energy_stats_func=src.jeff.compute_error_stats_d3,
        #     opt_type="Powell",
        # )
        print("D4")
        # src.optimization.opt_cross_val(
        #     df,
        #     nfolds=5,
        #     # start_params=params,
        #     start_params=adz_opt_params,
        #     hf_key=i,
        #     output_l_marker="G_",
        #     optimizer_func=src.optimization.optimization,
        # )
        src.optimization.opt_cross_val(
            df,
            nfolds=5,
            start_params=params,
            hf_key=i,
            output_l_marker="least",
            optimizer_func=src.optimization.optimization_least_squares,
        )
    return


def compute_D3_D4_values_for_params(
    df: pd.DataFrame,
    params_d3: [],
    params_d4: [],
    label: str,
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    """

    df[f"-D3 ({label})"] = df.apply(
        lambda r: src.jeff.compute_bj(params_d3, r["D3Data"]),
        axis=1,
    )
    df[f"-D4 ({label})"] = df.apply(
        lambda row: src.setup.compute_bj_from_dimer_AB_all_C6s(
            params_d4,
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
    return df


def plots(df) -> None:
    """
    plots
    """
    df["SAPT0-D4/aug-cc-pVDZ"] = df.apply(
        lambda row: row["HF_adz"] + row["-D4 (adz)"],
        axis=1,
    )
    df["SAPT0-D3/aug-cc-pVDZ"] = df.apply(
        lambda row: row["HF_adz"] + row["-D3 (adz)"],
        axis=1,
    )
    df["HF_adz_diff"] = df["HF_adz"] - df["Benchmark"]

    df["adz_diff_d4"] = df["SAPT0-D4/aug-cc-pVDZ"] - df["Benchmark"]
    df["adz_diff_d3"] = df["SAPT0-D3/aug-cc-pVDZ"] - df["Benchmark"]
    print(df["adz_diff_d3"].describe())
    print(df["adz_diff_d4"].describe())
    src.plotting.plot_dbs(df, "adz_diff", "SAPT0-D4/aug-cc-pVDZ", "adz_diff")
    src.plotting.plot_dbs(df, "HF_adz_diff", "HF/aug-cc-pVDZ", "HF_adz_diff")
    src.plotting.plot_dbs_d3_d4(
        df,
        "adz_diff_d3",
        "adz_diff_d4",
        "D3",
        "D4",
        "SAPT0-D/aug-cc-pVDZ",
        "adz_diff_d3_d4",
    )
    return


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # TODO: plot damping function (f vs. r_ab)

    pkl_name = "data/schr_dft.pkl"
    df = pd.read_pickle(pkl_name)
    print(df.columns)
    params_dict = src.paramsTable.paramsDict()
    params_d4 = params_dict["sadz"][1:]
    params_d3 = params_dict["sdadz"][1:]
    undamped = params_dict["undamped"][1:]
    # print(params_d3)
    # print(params_d4)
    # df = compute_D3_D4_values_for_params(df, params_d3, params_d4, "adz")
    # df = compute_D3_D4_values_for_params(df, undamped, undamped, "undamped")

    # df.to_pickle(pkl_name)

    df_saptdft = df[~df["pbe0_adz_saptdft"].isna()].copy()
    k = qcel.constants.conversion_factor("hartree", "kcal / mol")
    df_saptdft["pbe0_adz_saptdft_ndisp"] = df_saptdft.apply(
        lambda r: (sum(r["pbe0_adz_saptdft"][:2]) + r["pbe0_adz_saptdft"][3]) * k,
        axis=1,
    )
    df_saptdft["pbe0_adz_saptdft_sum"] = df_saptdft.apply(
        lambda r: (sum(r["pbe0_adz_saptdft"][:4])) * k,
        axis=1,
    )
    print(
        df_saptdft[
            ["Benchmark", "pbe0_adz_saptdft_ndisp", "pbe0_adz_saptdft_sum", "HF_adz"]
        ]
    )

    df_saptdft["SAPT0-D4/aug-cc-pVDZ"] = df_saptdft.apply(
        lambda row: row["HF_adz"] + row["-D4 (adz)"],
        axis=1,
    )
    df_saptdft["adz_diff_d4"] = df_saptdft["SAPT0-D4/aug-cc-pVDZ"] - df_saptdft["Benchmark"]

    df_saptdft["pbe0_adz_d4_adz"] = (
        df_saptdft["pbe0_adz_saptdft_ndisp"] + df_saptdft["-D4 (adz)"]
    )
    df_saptdft["pbe0_adz_d4_undamped"] = (
        df_saptdft["pbe0_adz_saptdft_ndisp"] + df_saptdft["-D4 (undamped)"]
    )
    df_saptdft["pbe0_adz_d3_undamped"] = (
        df_saptdft["pbe0_adz_saptdft_ndisp"] + df_saptdft["-D3 (undamped)"]
    )
    df_saptdft["pbe0_adz_d4_adz_diff"] = (
        df_saptdft["pbe0_adz_d4_adz"] - df_saptdft["Benchmark"]
    )
    df_saptdft["pbe0_adz_d4_undamped_diff"] = (
        df_saptdft["pbe0_adz_d4_undamped"] - df_saptdft["Benchmark"]
    )
    df_saptdft["pbe0_adz_d3_undamped_diff"] = (
        df_saptdft["pbe0_adz_d3_undamped"] - df_saptdft["Benchmark"]
    )

    print(
        df_saptdft[
            [
                "pbe0_adz_d4_undamped_diff",
                "pbe0_adz_d3_undamped_diff",
                "pbe0_adz_d4_adz_diff",
                "adz_diff_d4",
            ]
        ].describe()
    )

    return

    bases = [
        # "HF_dz",
        # "HF_jdz",
        # "HF_adz",
        # "HF_tz",
        # "HF_atz",
        # "HF_jdz_no_cp",
        # "HF_dz_no_cp",
        # "HF_qz",
        # "HF_qz_no_cp",
        # "HF_qz_no_df",
        # "HF_qz_conv_e_4",
        "pbe0_adz_saptdft_ndisp",
    ]
    optimize_paramaters(df_saptdft, bases)
    # print(df.iloc[0]["pbe0_adz_saptdft"] * 627.509)

    # src.setup.gather_data6(
    #     output_path="sr3.pkl",
    #     from_master=True,
    #     HF_columns=["HF_dz", "HF_jdz", "HF_adz", "HF_tz", "HF_jdz_dftd4", "HF_atz", "HF_jtz"],
    #     overwrite=True,
    # )
    # return

    # src.optimization.compute_stats_dftd4_values_fixed(df, fixed_col="dftd4_disp_ie_grimme_params")
    # src.optimization.compute_stats_dftd4_values_fixed(df, fixed_col='dftd4_disp_ie_grimme_params_ATM')
    return


if __name__ == "__main__":
    main()
