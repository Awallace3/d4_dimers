import numpy as np
import pandas as pd
import pickle
from pprint import pprint as pp

from src import saptdft
from src.jeff import (
    compute_error_stats_d3,
    d3data_stats,
    optimization_d3,
)
from src.setup import (
    gather_data6,
    ram_data_2,
)
from src.optimization import (
    optimization,
    opt_cross_val,
    find_max_e,
    optimization_least_squares,
    compute_int_energy_stats,
    compute_int_energy_stats_dftd4_key,
    compute_dftd4_values,
    compute_stats_dftd4_values_fixed,
    compute_int_energy,
)
from src.grimme_setup import (
    gather_BLIND_geoms,
    create_Grimme_db,
    create_grimme_s22s66blind,
    gather_grimme_from_db,
)
from src.tools import (
    print_cartesians,
    print_cartesians_pos_carts,
    stats_to_latex_row,
    hf_key_to_latex_cmd,
    df_to_latex_table_round,
    print_cartesians_dimer,
)


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # TODO: plot damping function (f vs. r_ab)

    df = pd.read_pickle("data/schr_dft.pkl")
    print(df.columns.values)
    for n, r in df.iterrows():
        # print(r)
        for i in r["Geometry"]:
            el = int(i[0])
            if el == 17:
                print(n)
                print_cartesians_dimer(
                    r["Geometry"],
                    r["monAs"],
                    r["monBs"],
                    r["charges"],
                )

    # for i in range(1, 10):
    #     r = df.iloc[i * 150]
    #     geom = r["Geometry"]
    #     pos = geom[:,0]
    #     carts = geom[:,1:]
    #     mona, monb = r["monAs"], r["monBs"]
    #     ma_g = carts[mona, :]
    #     mb_g = carts[monb, :]
    #     ma_p = pos[mona]
    #     mb_p = pos[monb]
    #     print("index: ", i)
    #     # print(r)
    #     # print(ma_g)
    #     print_cartesians_pos_carts(ma_p, ma_g)
    #     print_cartesians_pos_carts(mb_p, mb_g)
    return
    HF_params = [1.61679827, 0.44959224, 3.35743605]  # HF
    adz_opt_params = [0.829861, 0.706055, 1.123903]
    row = df.iloc[3000]
    saptdft.compute_disp_3_forms(row, HF_params)
    # saptdft.compute_disp_3_forms(row, adz_opt_params)
    #
    # print_cartesians(row["Geometry"])
    # print(row[["Benchmark", "HF_adz"]])
    # print_cartesians(row["Geometry"])
    # print()
    # row = df.iloc[0]
    # saptdft.compute_disp_3_forms(row, HF_params)
    # saptdft.compute_disp_3_forms(row, adz_opt_params)
    # print(row[["Benchmark", "HF_adz"]])
    # print_cartesians(row["Geometry"])
    # saptdft.plot_BJ_damping(1.61679827, 0.44959224, 3.35743605)

    return
    """
    df = df[~df["pbe0_adz_saptdft"].isna()]
    k = constants.conversion_factor("hartree", "kcal / mol")
    df["pbe0_adz_saptdft_ndisp"] = df.apply(
        lambda r: (sum(r["pbe0_adz_saptdft"][:2]) + r["pbe0_adz_saptdft"][3]) * k,
        axis=1,
    )
    df["pbe0_adz_saptdft_sum"] = df.apply(
        lambda r: (sum(r["pbe0_adz_saptdft"][:4])) * k,
        axis=1,
    )
    print(df[["Benchmark", "pbe0_adz_saptdft_ndisp", "pbe0_adz_saptdft_sum", "HF_adz"]])
    print(df.iloc[0]["pbe0_adz_saptdft"] * 627.509)
    """

    # gather_data6(
    #     output_path="sr3.pkl",
    #     from_master=True,
    #     HF_columns=["HF_dz", "HF_jdz", "HF_adz", "HF_tz", "HF_jdz_dftd4", "HF_atz", "HF_jtz"],
    #     overwrite=True,
    # )
    # return

    # compute_stats_dftd4_values_fixed(df, fixed_col="dftd4_disp_ie_grimme_params")
    # compute_stats_dftd4_values_fixed(df, fixed_col='dftd4_disp_ie_grimme_params_ATM')

    bases = [
        # "HF_dz",
        # "HF_jdz",
        "HF_adz",
        # "HF_tz",
        # "HF_atz",
        # "HF_jdz_no_cp",
        # "HF_dz_no_cp",
        # "HF_qz",
        # "HF_qz_no_cp",
        # "HF_qz_no_df",
        # "HF_qz_conv_e_4",
        # "pbe0_adz_saptdft_ndisp"
    ]

    for i in bases:
        print(i)
        params = [1.61679827, 0.44959224, 3.35743605]
        # params = [0.7683276390453782 , 0.09699087897359535 , 3.6407701963142745]
        # print("D3")
        # opt_cross_val(
        #     df,
        #     nfolds=5,
        #     start_params=params,
        #     hf_key=i,
        #     output_l_marker="D3_",
        #     optimizer_func=optimization_d3,
        #     compute_int_energy_stats_func=compute_error_stats_d3,
        #     opt_type="Powell",
        # )
        print("D4")
        # opt_cross_val(
        #     df,
        #     nfolds=5,
        #     start_params=params,
        #     hf_key=i,
        #     output_l_marker="G_",
        #     optimizer_func=optimization,
        # )
        # opt_cross_val(df, nfolds=5, start_params=params, hf_key=i, output_l_marker="least", optimizer_func=optimization_least_squares)

    return


if __name__ == "__main__":
    main()
