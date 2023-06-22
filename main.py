import numpy as np
import pandas as pd
import src
import qcelemental as qcel
import tqdm
from qm_tools_aw import tools

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
        src.grimme_setup.combine_data_with_new_df()
    elif version == "grimme_paper":
        src.grimme_setup.read_grimme_dftd4_paper_HF_energies()
    else:
        raise ValueError(f"version {version} not recognized")
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
        "data/grimme_fitset_test2.pkl",
        "data/HBC6.pkl"
    ]
    # NOTE: data/grimme_fitset_db3.pkl Geometry is in Angstrom!!!
    selected = names[i]
    print(f"Selected: {selected} for df")
    df = pd.read_pickle(selected)
    return df, selected


def make_bohr(geometry):
    # df["Geometry_bohr"] = df.apply(lambda x: make_bohr(x["Geometry"]), axis=1)
    return np.hstack(
        (np.reshape(geometry[:, 0], (-1, 1)), ang_to_bohr * geometry[:, 1:])
    )


def grimme_test_atm(df_names_inds=[3, 4]) -> None:
    """
    grimme_test_atm
    """

    for n, i in enumerate(df_names_inds):
        hf_qz_no_cp = "HF_qz_no_cp"
        if i == 4:
            hf_qz_no_cp = "HF_qz"
        df, _ = df_names(i)
        df[hf_qz_no_cp].dropna(inplace=True)
        df["dftd4_ie"] = df.apply(lambda r: r["d4Ds"] - r["d4As"] - r["d4Bs"], axis=1)
        df["diff"] = df.apply(
            lambda r: r["Benchmark"] - (r[hf_qz_no_cp] + r["dftd4_ie"]),
            axis=1,
        )
        print(df[["diff", "Benchmark", hf_qz_no_cp, "dftd4_ie"]].describe())
        # root mean square error of diff
        RMSE = np.sqrt(np.mean(df["diff"] ** 2))
        print(f"{RMSE = :.4f}\n\n")
        if n == 0:
            df1 = df
        elif n == 1:
            df2 = df

    df1["diff_diff"] = df1.apply(
        lambda r: -(r["diff"] - df2.loc[r.name]["diff"]), axis=1
    )
    df1["HF_qz_diff"] = df1.apply(
        lambda r: r["HF_qz_no_cp"] - df2.loc[r.name]["HF_qz"], axis=1
    )
    print(df1[["diff_diff", "HF_qz_diff"]].describe())
    print("HF_qz_df1\tHF_qz_df2\tHF_qz_diff")
    for n in range(len(df1)):
        print(
            df1.iloc[n]["HF_qz_no_cp"], df2.iloc[n]["HF_qz"], df1.iloc[n]["HF_qz_diff"]
        )

    return


def compute_ie_differences(df_num=0):
    df, selected = df_names(df_num)
    if False:
        d4_dimers, d4_mons = [], []
        r4r2_ls = src.r4r2.r4r2_vals_ls()
        for n, r in df.iterrows():
            print(n)
            d4_dimer, d4_mon = src.locald4.compute_bj_with_different_C6s(
                r["Geometry_bohr"],
                r["monAs"],
                r["monBs"],
                r["charges"],
                r["C6s"],
                r["C6_A"],
                r["C6_B"],
                params=src.paramsTable.get_params("HF"),
                s9=0.0,
                r4r2_ls=r4r2_ls,
            )
            d4_dimers.append(d4_dimer)
            d4_mons.append(d4_mon)
        df["d4_C6s_dimer"] = d4_dimers
        df["d4_C6s_monomers"] = d4_mons
        df["d4_C6s_diff"] = df.apply(
            lambda r: r["d4_C6s_dimer"] - r["d4_C6s_monomers"], axis=1
        )
        print(df[["d4_C6s_dimer", "d4_C6s_monomers", "d4_C6s_diff"]].describe())
        df.to_pickle(selected)
    investigate = [
        "d4_C6s_dimer",
        "d4_C6s_monomers",
        "d4_C6s_diff",
        # "HF_adz",
        # "Benchmark",
        "DB",
        # "monAs",
        # "monBs",
    ]
    if True:
        print(df[investigate].describe())
        print()
        df["d4_C6s_diff_abs"] = abs(df["d4_C6s_diff"])
        df.sort_values(by="d4_C6s_diff_abs", inplace=True, ascending=False)

        def print_rows(df, break_after=1000):
            print("i", end="      ")
            for l in investigate:
                if len(l) > 8:
                    print(f"{l[:8]}", end="   ")
                else:
                    while len(l) < 8:
                        l += " "
                    print(f"{l}", end="   ")
            print()
            cnt = 0
            for n, r in df.iterrows():
                print(n, end="\t")
                for l in investigate:
                    if type(r[l]) != float:
                        print(f"{r[l]}   ", end="    ")
                    else:
                        print(f"{r[l]:.4f}", end="    ")
                print()
                cnt += 1
                if cnt == break_after:
                    return
        print_rows(df)
    if False:
        indices = [
            515,
            500,
            450,
            385,
        ]
        for i in indices:
            print(i, df.loc[i]['charges'][0], 'angstrom')
            tools.print_cartesians(df.loc[i]["Geometry"], True)
            print()

    return df


def main():
    # gather_data("schr")
    df, selected = df_names(0)

    def opt():
        adz_opt_params = [0.829861, 0.706055, 1.123903]
        bases = [
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

    compute_ie_differences(0)
    compute_ie_differences(5)

    # opt()
    # grimme_test_atm()
    # src.plotting.plotting_setup(df_names(0), False)
    return


if __name__ == "__main__":
    main()
