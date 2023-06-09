import numpy as np
import pandas as pd
import src


def gather_data():
    # Gather data
    src.setup.gather_data6(
        output_path="data/d4.pkl",
        from_master=True,
        HF_columns=[
            "HF_dz",
            "HF_jdz",
            "HF_adz",
            "HF_tz",
            "HF_jdz_dftd4",
            "HF_atz",
            "HF_jtz",
        ],
        overwrite=True,
    )
    return


def optimize_paramaters(
    df,
    bases,
    start_params_d3=[0.7683276390453782, 0.09699087897359535, 3.6407701963142745],
    start_params_d4_key="HF",
    D3={"powell": True},
    D4={"powell": True, "least_squares": True},
) -> None:
    # Optimize parameters through 5-fold cross validation
    params = src.paramsTable.paramsDict()[start_params_d4_key][1:]
    for i in bases:
        print(i)
        if D3["powell"]:
            print("D3 powell")
            src.optimization.opt_cross_val(
                df,
                nfolds=5,
                start_params=start_params_d3,
                hf_key=i,
                output_l_marker="D3_",
                optimizer_func=src.jeff.optimization_d3,
                compute_int_energy_stats_func=src.jeff.compute_error_stats_d3,
                opt_type="Powell",
            )
        if D4["powell"]:
            print("D4 powell")
            src.optimization.opt_cross_val(
                df,
                nfolds=5,
                start_params=params,
                hf_key=i,
                output_l_marker="G_",
                optimizer_func=src.optimization.optimization,
            )
        if D4["least_squares"]:
            print("D4 least_squares")
            src.optimization.opt_cross_val(
                df,
                nfolds=5,
                start_params=params,
                hf_key=i,
                output_l_marker="least",
                optimizer_func=src.optimization.optimization_least_squares,
            )
    return


def total_bases():
    return [
        "HF_dz",
        "HF_jdz",
        "HF_adz",
        "HF_tz",
        "HF_atz",
        "HF_jdz_no_cp",
        "HF_dz_no_cp",
        "HF_qz",
        "HF_qz_no_cp",
        "HF_qz_no_df",
        "HF_qz_conv_e_4",
        "pbe0_adz_saptdft_ndisp",
    ]


def main():
    # gather_data()
    df = pd.read_pickle("data/d4.pkl")
    adz_opt_params = [0.829861, 0.706055, 1.123903]
    bases = [
        # "TAG",
        "HF_adz",
    ]
    optimize_paramaters(df, bases, start_params_d4_key="sadz", D3={"powell": False}, D4={"powell": True, "least_squares": False})

    return


if __name__ == "__main__":
    main()
