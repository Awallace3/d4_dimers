import numpy as np
import pandas as pd
import src
import qcelemental as qcel
import tqdm
from qm_tools_aw import tools
import dispersion

# TODO: Plot Violin plots for each basis set
#       - collect basis set

# from pandarallel import pandarallel
# pandarallel.initialize(use_memory_fs=True)
# from parallel_pandas import ParallelPandas
# ParallelPandas.initialize(
#     n_cpu=8, split_factor=4, show_vmem=True, disable_pr_bar=False
# )

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
    D3={
        "powell": True,
    },
    D4={
        "powell": True,
        "least_squares": False,
        "powell_ATM_TT": True,
    },
    ATM=False,
    extra="",
    use_2B_C6s=False,
) -> None:
    # Optimize parameters through 5-fold cross validation
    # params = src.paramsTable.paramsDict()[start_params_d4_key][1:]
    params = src.paramsTable.get_params(start_params_d4_key)
    dispersion.omp_set_num_threads(8)
    print(f"Starting Key: {start_params_d4_key}")
    subset = [
        "Geometry_bohr",
        *bases,
        "D3Data",
        "Benchmark",
        "charges",
        "monAs",
        "monBs",
        "C6s",
        "C6_A",
        "C6_B",
        "C6_ATM",
        "C6_ATM_A",
        "C6_ATM_B",
    ]
    df = df[subset].copy()
    if use_2B_C6s:
        print("Using 2B, charge scaled C6s!")
        df["C6_ATM"] = df["C6s"]
        df["C6_ATM_A"] = df["C6_A"]
        df["C6_ATM_B"] = df["C6_B"]

    for i in bases:
        extra_added = extra
        print(i)
        if D3["powell"]:
            print("D3 powell")
            version = {
                "method": "powell",
                "compute_energy": "jeff_d3",
                "compute_stats": "jeff_d3",
            }
            src.optimization.opt_cross_val(
                df,
                nfolds=5,
                start_params=start_params_d3,
                hf_key=i,
                output_l_marker="D3_" + extra_added,
                version=version,
            )
        if D4["powell"]:
            print("D4 powell")
            if ATM:
                print("ATM ON")
                compute_energy = "compute_int_energy_DISP"
                extra_added += "ATM_"
                # params.append(1.0)
            else:
                print("ATM OFF")
                # TODO: need to ensure s9 is 0.0
                compute_energy = "compute_int_energy_DISP"
                extra_added += "2B_"
            version = {
                "method": "powell",
                "compute_energy": compute_energy,
                "compute_stats": "compute_int_energy_stats_DISP",
            }

            src.optimization.opt_cross_val(
                df,
                nfolds=5,
                start_params=params,
                hf_key=i,
                # output_l_marker=f"{extra_added}",
                output_l_marker="D4_" + extra_added,
                version=version,
                force_ATM_on=ATM,
            )
        if D4["powell_ATM_TT"]:
            print("D4 powell ATM TT")
            if ATM:
                print("ATM ON")
                extra_added += "ATM_"
            else:
                print("ATM OFF")
                extra_added += "2B_"
            version = {
                "method": "powell",
                "compute_energy": "compute_int_energy_DISP_TT",
                "compute_stats": "compute_int_energy_stats_DISP_TT",
            }

            src.optimization.opt_cross_val(
                df,
                nfolds=5,
                start_params=params,
                hf_key=i,
                # output_l_marker=f"{extra_added}",
                output_l_marker="TT_" + extra_added,
                version=version,
            )
        if D4["least_squares"]:
            print("D4 least_squares")
            version = {
                "method": "powell",
                "compute_energy": "compute_int_energy_least_squares",
                "compute_stats": "compute_int_energy_stats",
            }
            src.optimization.opt_cross_val(
                df,
                nfolds=5,
                start_params=params,
                hf_key=i,
                output_l_marker="least",
                version=version,
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
        "data/schr_dft_charges.pkl",
        "data/schr_dft2_SR.pkl",
        "plots/basis.pkl",
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


def make_geometry_bohr_column(i):
    df, selected = df_names(i)
    tools.print_cartesians(df.iloc[0]["Geometry"])
    df["Geometry_bohr"] = df.apply(lambda x: make_bohr(x["Geometry"], True), axis=1)
    print()
    tools.print_cartesians(df.iloc[0]["Geometry_bohr"])
    print(df.columns.values)
    df.to_pickle(selected)
    return


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
    return


def merge_SAPT0_results_into_df():
    df, selected = df_names(6)
    print(df)
    df2 = pd.read_pickle("data/schr_sapt0.pkl")
    print(df2.columns.values)
    copy_SAPT0_cols = [
        "id",
        "SAPT0_dz",
        "SAPT0_jdz",
        "SAPT0_adz",
        "SAPT0_tz",
        "SAPT0_mtz",
        "SAPT0_jtz",
        "SAPT0_atz",
    ]
    for i in copy_SAPT0_cols:
        df[i] = df2[i]
    for i in copy_SAPT0_cols:
        print(df[i])
    for i in [
        j
        for j in df.columns.values
        if "SAPT0_" in j
        if j not in ["SAPT0", "SAPT0_aqz"]
        if "_IE" not in j
    ]:
        print(f'"{i}_3_IE",')
        df[i + "_3_IE"] = df.apply(lambda r: sum(r[i][1:4]), axis=1)
        df[i + "_IE"] = df.apply(lambda r: r[i][0], axis=1)
    df.to_pickle(selected)
    return


def main():
    # TODO: plot -D4 2B with Grimme parameters
    # TODO: plot -D3 ATM
    # df, selected = df_names(8)
    # src.dftd3.compute_dftd3(*df_names(4), "Geometry", param_label="D3MBJ")
    # src.dftd3.compute_dftd3(*df_names(4), "Geometry", param_label="D3MBJ ATM")
    # merge_SAPT0_results_into_df()
    df, selected = df_names(6)
    # df.to_pickle(selected)

    bases = [
        "SAPT0_adz_3_IE",
        # "SAPT0_jdz_3_IE",
        # "SAPT0_mtz_3_IE",
        # "SAPT0_jtz_3_IE",
        # "SAPT0_dz_3_IE",
        # "SAPT0_atz_3_IE",
        # "SAPT0_tz_3_IE",
    ]

    def opt(bases):
        optimize_paramaters(
            df,
            bases,
            start_params_d4_key="HF_ATM_CHG_OPT_START",
            # D3={"powell": True},
            D4={"powell": True, "least_squares": False, "powell_ATM_TT": False},
            # start_params_d4_key="HF_ATM_TT_OPT_START",
            D3={"powell": False},
            # D4={"powell": False, "least_squares": False, "powell_ATM_TT": True},
            ATM=True,
            # extra="",
            extra="2B_C6s_uncharged",
            use_2B_C6s=False,
        )

    # opt(bases)
    # return
    # opt(["HF_qz"])
    # opt(["HF_adz"])

    def SR_testing():
        import dispersion

        src.sr.generate_SR_data_ATM(
            *df_names(6),
            params_key="HF_ATM_SHARED",
            # params_key="SAPT0_adz_3_IE_ATM",
        )
        return

    SR_testing()
    return
    # return
    # src.misc.sensitivity_analysis(df)
    # src.misc.examine_ATM_TT(df)
    if True:
        df, _ = df_names(9)
        src.plotting.plot_basis_sets_d4(
            df,
            True,
        )
        df, _ = df_names(9)
        src.plotting.plot_basis_sets_d3(
            df,
            True,
        )
    if True:
        src.plotting.plotting_setup(
            df_names(9),
            True,
        )
    return


if __name__ == "__main__":
    main()
