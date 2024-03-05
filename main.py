import pandas as pd
import src
import dispersion
import os
import subprocess


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
        "powell_C6_only": True,
    },
    ATM=False,
    extra="",
    use_2B_C6s=False,
    drop_na=True,
    five_fold=False,
    omp_threads=20,
) -> None:
    """
    Optimize the parameters for the D3 and D4 dispersion models.

    """

    params = src.paramsTable.get_params(start_params_d4_key)
    dispersion.omp_set_num_threads(omp_threads)
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
    if drop_na:
        df = df[df[bases].notna().all(axis=1)].copy()
        print(f"Dropped NaNs, new size: {len(df)}")
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
            else:
                print("ATM OFF")
                compute_energy = "compute_int_energy_DISP"
                extra_added += "2B_"
            version = {
                "method": "powell",
                "compute_energy": compute_energy,
                "compute_stats": "compute_int_energy_stats_DISP",
            }

            if five_fold:
                src.optimization.opt_cross_val(
                    df,
                    nfolds=5,
                    start_params=params,
                    hf_key=i,
                    output_l_marker="D4_" + extra_added,
                    version=version,
                    force_ATM_on=ATM,
                )
            else:
                src.optimization.opt_val_no_folds(
                    df,
                    start_params=params,
                    hf_key=i,
                    version=version,
                    force_ATM_on=ATM,
                )

        if D4["powell_C6_only"]:
            print("D4 powell")
            if ATM:
                print("ATM ON")
                extra_added += "2B_C6_ATM_"
            else:
                print("ATM OFF")
                extra_added += "2B_C6"
            version = {
                "method": "powell",
                "compute_energy": "compute_int_energy_DISP_C6_only",
                "compute_stats": "compute_int_energy_stats_DISP_C6_only",
            }

            if five_fold:
                src.optimization.opt_cross_val(
                    df,
                    nfolds=5,
                    start_params=params,
                    hf_key=i,
                    output_l_marker="D4_" + extra_added,
                    version=version,
                    force_ATM_on=ATM,
                )
            else:
                src.optimization.opt_val_no_folds(
                    df,
                    start_params=params,
                    hf_key=i,
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

            if five_fold:
                src.optimization.opt_cross_val(
                    df,
                    nfolds=5,
                    start_params=params,
                    hf_key=i,
                    output_l_marker="D4_" + extra_added,
                    version=version,
                    force_ATM_on=ATM,
                )
            else:
                src.optimization.opt_val_no_folds(
                    df,
                    start_params=params,
                    hf_key=i,
                    version=version,
                    force_ATM_on=ATM,
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


def df_names(i):
    names = [
        "dfs/schr_dft2.pkl",
        "dfs/schr_dft2_SR.pkl",
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
        "data/schr_saptdft.pkl"
        "plots/basis_study.pkl",
    ]
    if i == 0:
        if not os.path.exists("dfs/schr_dft2.pkl"):
            print("Cannot find ./dfs/schr_dft2.pkl, creating it now...")
            subprocess.call("cat dfs/schr_dft2.pkl_part* > dfs/schr_dft2.pkl.tar.gz", shell=True)
            subprocess.call("tar -xzf dfs/schr_dft2.pkl.tar.gz", shell=True)
            subprocess.call("rm dfs/schr_dft2.pkl.tar.gz", shell=True)
            subprocess.call("mv schr_dft2.pkl dfs/schr_dft2.pkl", shell=True)
    selected = names[i]
    print(f"Selected: {selected} for df")
    df = pd.read_pickle(selected)
    return df, selected


def main():
    df, selected = df_names(0)

    bases = [
        "SAPT0_adz_3_IE",
        # "SAPT_DFT_adz_3_IE",
        # "SAPT_DFT_atz_3_IE",
        # "SAPT0_jdz_3_IE",
        # "SAPT0_mtz_3_IE",
        # "SAPT0_jtz_3_IE",
        # "SAPT0_dz_3_IE",
        # "SAPT0_atz_3_IE",
        # "SAPT0_tz_3_IE",
    ]

    def opt(bases, start_params_d4_key="SAPT_DFT_OPT_START4"):
        optimize_paramaters(
            df,
            bases,
            start_params_d4_key=start_params_d4_key,
            D4={"powell": False,
                 "least_squares": False,
                 "powell_ATM_TT": True,
                 "powell_C6_only": False,
            },

            D3={"powell": False},
            ATM=True,
            extra="",
            # extra="SAPT_DFT_",
            use_2B_C6s=False,
            five_fold=False,
        )

    # opt(bases, "SAPT_DFT_OPT_START4")
    # opt(bases, "SAPT_DFT_OPT_START5")
    # opt(bases, "SAPT_DFT_OPT_START3")
    # opt(bases, "SAPT0_adz_BJ_ATM")
    opt(bases, "SAPT0_adz_BJ_ATM_TT_5p")
    # opt(bases, "SAPT_DFT_atz_ATM_TT_OPT_START_2p")
    return


if __name__ == "__main__":
    main()
