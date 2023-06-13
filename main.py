import numpy as np
import pandas as pd
import src
import qcelemental as qcel

ang_to_bohr = src.constants.Constants().g_aatoau()
hartree_to_kcal_mol = qcel.constants.conversion_factor("hartree", "kcal / mol")


def gather_data(version="schr"):
    # Gather data
    if version == "schr":
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
    elif version == "grimme":
        # src.grimme_setup.create_grimme_s22s66blind()
        src.grimme_setup.combine_data_with_new_df()
        # src.grimme_setup.gather_grimme_from_db()

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
    # params = src.paramsTable.paramsDict()[start_params_d4_key][1:]
    params = src.paramsTable.paramsDict()[start_params_d4_key]
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


def df_names(i):
    names = [
        "data/d4.pkl",
        "data/grimme_fitset_db3.pkl",
        "data/schr_dft.pkl",
        "data/grimme_fitset_total.pkl",
    ]
    selected = names[i]
    print(f"Selected: {selected} for df")
    df = pd.read_pickle(selected)
    return df


def make_bohr(geometry):
    # df["Geometry_bohr"] = df.apply(lambda x: make_bohr(x["Geometry"]), axis=1)
    return np.hstack(
        (np.reshape(geometry[:, 0], (-1, 1)), ang_to_bohr * geometry[:, 1:])
    )


def grimme_test_atm() -> None:
    """
    grimme_test_atm
    """
    df = df_names(1)
    df["diff"] = df.apply(
        lambda r: r["Benchmark"] - (r["HF_qz"] + r["dftd4_disp_ie_grimme_params_ATM"]),
        axis=1,
    )
    print(df[['diff', 'Benchmark', "HF_qz", "dftd4_disp_ie_grimme_params_ATM"]].describe())
    # root mean square error of diff
    RMSE = np.sqrt(np.mean(df["diff"] ** 2))
    print(f"{RMSE = }\n\n")

    df = df_names(4)
    df["HF_qz"].dropna(inplace=True)
    df['dftd4_ie'] = df.apply(lambda r: r['d4Ds'] - r['d4As'] - r['d4Bs'], axis=1)
    df["diff"] = df.apply(
        lambda r: r["Benchmark"] - (r["HF_qz"] + r['dftd4_ie']),
        axis=1,
    )
    print(df[['diff', 'Benchmark', "HF_qz", "dftd4_ie"]].describe())
    # root mean square error of diff
    RMSE = np.sqrt(np.mean(df["diff"] ** 2))
    print(f"{RMSE = }\n\n")
    return


def main():
    # gather_data("grimme")
    df = df_names(0)

    def opt():
        adz_opt_params = [0.829861, 0.706055, 1.123903]
        bases = [
            # "TAG",
            # "HF_adz",
            # "HF_jdz",
            # "HF_qz"
            "pbe0_adz_saptdft_ndisp",
        ]
        optimize_paramaters(
            df,
            bases,
            # start_params_d4_key="sadz",
            start_params_d4_key="sadz",
            D3={"powell": False},
            D4={"powell": True, "least_squares": False},
        )

    # opt()
    grimme_test_atm()
    return


if __name__ == "__main__":
    main()
