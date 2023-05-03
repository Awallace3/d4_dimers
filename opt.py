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


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # TODO: plot damping function (f vs. r_ab)

    df = pd.read_pickle("data/schr_dft.pkl")
    print(df.columns)
    HF_params = [1.61679827, 0.44959224, 3.35743605]  # HF
    adz_opt_params = [0.829861, 0.706055, 1.123903]

    """
    df = df[~df["pbe0_adz_saptdft"].isna()]
    k = qcel.constants.conversion_factor("hartree", "kcal / mol")
    df["pbe0_adz_saptdft_ndisp"] = df.apply(
        lambda r: (sum(r["pbe0_adz_saptdft"][:2]) + r["pbe0_adz_saptdft"][3]) * k,
        axis=1,
    )
    df["pbe0_adz_saptdft_sum"] = df.apply(
        lambda r: (sum(r["pbe0_adz_saptdft"][:4])) * k,
        axis=1,
    )
    print(df[["Benchmark", "pbe0_adz_saptdft_ndisp", "pbe0_adz_saptdft_sum", "HF_adz"]])
    return
    """
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

    bases = [
        "HF_atz",
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
        # "pbe0_adz_saptdft_ndisp",
    ]

    adz_opt_params = [0.829861, 0.706055, 1.123903]
    params = [1.61679827, 0.44959224, 3.35743605]
    params_d3 = [0.7683276390453782, 0.09699087897359535, 3.6407701963142745]
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
        src.optimization.opt_cross_val(
            df,
            nfolds=5,
            start_params=params,
            # start_params=adz_opt_params,
            hf_key=i,
            output_l_marker="G_",
            optimizer_func=src.optimization.optimization,
        )
        src.optimization.opt_cross_val(
            df,
            nfolds=5,
            start_params=params,
            hf_key=i,
            output_l_marker="least",
            optimizer_func=src.optimization.optimization_least_squares,
        )
    return


if __name__ == "__main__":
    main()
