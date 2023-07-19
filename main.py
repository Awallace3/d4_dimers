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
        "data/HBC6.pkl",
        "data/schr_dft2.pkl",
    ]
    # NOTE: data/grimme_fitset_db3.pkl Geometry is in Angstrom!!!
    selected = names[i]
    print(f"Selected: {selected} for df")
    df = pd.read_pickle(selected)
    return df, selected


def make_bohr(geometry, ang_to_bohr_convert):
    if ang_to_bohr_convert:
        return np.hstack(
            (np.reshape(geometry[:, 0], (-1, 1)), ang_to_bohr * geometry[:, 1:])
        )
    else:
        return np.hstack(
            (np.reshape(geometry[:, 0], (-1, 1)), 1 / ang_to_bohr * geometry[:, 1:])
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
    # df['Geometry_bohr'] = df.apply(lambda x: x['Geometry'], axis=1)
    # df['Geometry'] = df.apply(lambda x: make_bohr(x['Geometry'], False), axis=1)
    params = src.paramsTable.paramsDict()["HF"]
    if False:
        d4_dimers, d4_mons, d4_diffs = [], [], []
        r4r2_ls = src.r4r2.r4r2_vals_ls()
        for n, row in df.iterrows():
            print(n)
            ma = row["monAs"]
            mb = row["monBs"]
            charges = row["charges"]
            geom_bohr = row["Geometry_bohr"]
            C6s_dimer = row["C6s"]
            C6s_mA = row["C6_A"]
            C6s_mB = row["C6_B"]

            d4_dimer, d4_mons_individually = src.locald4.compute_bj_with_different_C6s(
                geom_bohr,
                ma,
                mb,
                charges,
                C6s_dimer,
                C6s_mA,
                C6s_mB,
                params,
            )
            diff = d4_dimer - d4_mons_individually

            d4_dimers.append(d4_dimer)
            d4_mons.append(d4_mons_individually)
            d4_diffs.append(diff)

        df["d4_C6s_dimer"] = d4_dimers
        df["d4_C6s_monomers"] = d4_mons
        df["d4_C6s_diff"] = d4_diffs
        df["d4_C6s_diff_abs"] = abs(df["d4_C6s_diff"])
        print(
            df[
                ["d4_C6s_dimer", "d4_C6s_monomers", "d4_C6s_diff", "d4_C6s_diff_abs"]
            ].describe()
        )
        df.to_pickle(selected)
    df["d4_C6s_dimer_IE"] = pd.to_numeric(
        df["Benchmark"] - (df["HF_adz"] + df["d4_C6s_dimer"])
    )
    df["d4_C6s_monomer_IE"] = pd.to_numeric(
        df["Benchmark"] - (df["HF_adz"] + df["d4_C6s_monomers"])
    )
    print(df[["d4_C6s_dimer_IE", "d4_C6s_monomer_IE"]].describe())
    investigate_pre = [
        "d4_C6s_dimer",
        "d4_C6s_monomers",
        "d4_C6s_diff",
        "d4_C6s_diff_abs",
        "HF_adz",
        "DB",
        # "monAs",
        # "monBs",
        "Benchmark",
    ]
    if True:
        investigate = []
        for i in investigate_pre:
            if i in df.columns.values:
                investigate.append(i)

        print(df[investigate].describe())
        print()
        df["d4_C6s_diff_abs"] = abs(df["d4_C6s_diff"])
        df2 = df.sort_values(by="d4_C6s_diff_abs", inplace=False, ascending=False)

        def print_rows(df, break_after=9000):
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
            for n, r in df2.iterrows():
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
        cnt = 0
        for n, r in df2.iterrows():
            print(n, r["charges"][0], "angstrom")
            tools.print_cartesians_dimer(
                r["Geometry"], r["monAs"], r["monBs"], r["charges"]
            )
            cnt += 1
            if cnt > 3:
                break
    if True:
        idx = 3014
        r = df.iloc[idx]
        print("FINAL:", idx)
        print(r["charges"][0], "angstrom")
        tools.print_cartesians_dimer(
            r["Geometry"], r["monAs"], r["monBs"], r["charges"]
        )
        # print(r['charges'][0], 'bohr')
        # tools.print_cartesians_dimer(r["Geometry_bohr"], r['monAs'], r['monBs'], r['charges'])

    return df


def charge_comparison():
    df, selected = df_names(0)
    print(df.columns.values)
    def_charge = [[0, 1] for i in range(3)]
    cnt_correct = 0
    cnt_wrong = 0
    for n, r in df.iterrows():
        line = f"{n} {r['charges']} {r['HF INTERACTION ENERGY']:.4f} {r['HF_jdz']:.4f}"
        e_diff = abs(r["HF INTERACTION ENERGY"] - r["HF_jdz"])

        if n < 0:
            print(line)
        elif not np.all(r["charges"] == def_charge):
            if e_diff < 0.001:
                cnt_correct += 1
            else:
                cnt_wrong += 1
                print(line)
    print(cnt_correct, cnt_wrong)


def main():
    src.misc.main()
    return
    # gather_data("schr")
    df, selected = df_names(6)
    print(df.columns.values)


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

    # compute_ie_differences(0)
    # compute_ie_differences(5)

    # opt()
    # grimme_test_atm()
    src.plotting.plotting_setup(df_names(6), False, "plots/plot2.pkl")
    return


if __name__ == "__main__":
    main()
