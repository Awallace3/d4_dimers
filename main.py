import pandas as pd
import src
import dispersion
import os
import subprocess
import argparse

# TODO: implement powell_2B_BJ_ATM_TT and powell_2B_TT_ATM_TT in optimzation.py

def optimize_paramaters(
    df,
    level_theories,
    start_params_d3=[0.7683276390453782, 0.09699087897359535, 3.6407701963142745],
    start_params_d4_key="HF",
    D3={
        "powell": True,
    },
    D4={
        "powell": False,
        "least_squares": False,
        "powell_2B_BJ_ATM_TT": False,
        "powell_2B_TT_ATM_TT": False,
        "powell_C6_only": False,
    },
    ATM=False,
    extra="",
    use_2B_C6s=False,
    drop_na=False,
    five_fold=False,
    omp_threads=18,
) -> None:
    """
    Optimize the parameters for the D3 and D4 dispersion models.

    """
    print(f"{D4 = }")

    params = src.paramsTable.get_params(start_params_d4_key)
    dispersion.omp_set_num_threads(omp_threads)
    print(f"Starting Key: {start_params_d4_key}")
    subset = [
        "Geometry_bohr",
        *level_theories,
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
        df = df[df[level_theories].notna().all(axis=1)].copy()
        print(f"Dropped NaNs, new size: {len(df)}")
    if use_2B_C6s:
        print("Using 2B, charge scaled C6s!")
        df["C6_ATM"] = df["C6s"]
        df["C6_ATM_A"] = df["C6_A"]
        df["C6_ATM_B"] = df["C6_B"]

    print(f"Level Theories: {level_theories}")
    for i in level_theories:
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
            extra_added = extra
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
            extra_added = extra

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
            extra_added = extra

        if D4["powell_2B_BJ_ATM_TT"]:
            print("D4 powell 2B BJ ATM TT")
            extra_added += "powell_2B_TT_ATM_TT_"
            if ATM:
                print("ATM ON")
                extra_added += "ATM_"
            else:
                print("ATM OFF")
                extra_added += "2B_"
            version = {
                "method": "powell",
                "compute_energy": "compute_int_energy_DISP_2B_BJ_ATM_TT",
                "compute_stats": "compute_int_energy_stats_DISP_2B_BJ_ATM_TT",
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
                    output_marker="powell_2B_BJ_ATM_TT",
                    force_ATM_on=ATM,
                )
            extra_added = extra

        if D4["powell_2B_TT_ATM_TT"]:
            print("D4 powell 2B TT ATM TT")
            extra_added += "powell_2B_TT_ATM_TT_"
            if ATM:
                print("ATM ON")
                extra_added += "ATM_"
            else:
                print("ATM OFF")
                extra_added += "2B_"
            version = {
                "method": "powell",
                "compute_energy": "compute_int_energy_DISP_2B_TT_ATM_TT",
                "compute_stats": "compute_int_energy_DISP_2B_TT_ATM_TT",
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
                    output_marker="powell_2B_TT_ATM_TT",
                    force_ATM_on=ATM,
                )
            extra_added = extra

        if D4["powell_2B_BJ_supramolecular"]:
            print("D4 powell 2B BJ supramolecular")
            extra_added += "powell_2B_BJ_supramolecular_"
            if ATM:
                print("ATM ON")
                extra_added += "ATM_"
            else:
                print("ATM OFF")
                extra_added += "2B_"
            version = {
                "method": "powell",
                "compute_energy": "compute_int_energy_DISP_2B_BJ_supra",
                "compute_stats": "compute_int_energy_stats_DISP_2B_BJ_supra",
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
                    output_marker="powell_2B_BJ",
                    force_ATM_on=ATM,
                )
            extra_added = extra

        if D4["powell_2B_TT_supramolecular"]:
            print("D4 powell 2B TT supramolecular")
            extra_added += "powell_2B_TT_supramolecular_"
            if ATM:
                print("ATM ON")
                extra_added += "ATM_"
            else:
                print("ATM OFF")
                extra_added += "2B_"
            version = {
                "method": "powell",
                "compute_energy": "compute_int_energy_DISP_2B_TT_supra",
                "compute_stats": "compute_int_energy_stats_DISP_2B_TT_supra",
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
                    output_marker="powell_2B_TT",
                    force_ATM_on=ATM,
                )
            extra_added = extra

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
            extra_added = extra
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
        "data/schr_saptdft.pkl",
        "plots/basis_study.pkl",
    ]
    if i == 0:
        if not os.path.exists("dfs/schr_dft2.pkl"):
            print("Cannot find ./dfs/schr_dft2.pkl, creating it now...")
            subprocess.call(
                "cat dfs/schr_dft2.pkl_part* > dfs/schr_dft2.pkl.tar.gz", shell=True
            )
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

    parser = argparse.ArgumentParser(
        description="main.py for optimizing D3 and D4 dispersion parameters."
    )
    parser.add_argument(
        "--level_theories",
        type=str,
        help="Pandas Column Name for the level of theory to optimize for (Default: SAPT0_adz_3_IE)",
        nargs="+",
        default="SAPT0_adz_3_IE",
    )
    parser.add_argument(
        "--start_params_d4_key",
        type=str,
        help="Key for the start parameters for the D4 optimization. Find available options in src/paramsTable.py:paramsDict() (Default: SAPT_DFT_OPT_START4)",
        default="SAPT_DFT_OPT_START4",
    )
    parser.add_argument(
        "--powell",
        help="Flag for using Powell optimization (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--least_squares",
        help="Flag for using least_squares optimization (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--powell_2B_BJ_ATM_TT",
        help="Flag for using Powell 2B BJ ATM TT optimization (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--powell_C6_only",
        help="Flag for using Powell C6 only optimization (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--powell_2B_TT_ATM_TT",
        help="Flag for using Powell 2B TT ATM TT optimization (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ATM",
        help="Flag for using ATM (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_2B_C6s",
        help="Flag for using 2B C6s (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--D3",
        help="Flag for using D3 (Default: False)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--five_fold",
        help="Flag for using 5-fold cross validation (Default: False)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--extra_label",
        type=str,
        help="Extra label to add to the output file from 5-fold (Default: None)",
        default="",
    )
    parser.add_argument(
        "--supramolecular_BJ",
        help="Flag for supramolecular dipsersion interaction energy only using dimer C6s (Default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--supramolecular_TT",
        help="Flag for supramolecular dipsersion interaction energy only using dimer C6s (Default: False)",
        action="store_true",
        default=False,
    )


    args = parser.parse_args()
    print(args)
    optimize_paramaters(
        df=df,
        level_theories=args.level_theories,
        start_params_d4_key=args.start_params_d4_key,
        D4={
            "powell": args.powell,
            "least_squares": args.least_squares,
            "powell_2B_BJ_ATM_TT": args.powell_2B_BJ_ATM_TT,
            "powell_C6_only": args.powell_C6_only,
            "powell_2B_TT_ATM_TT": args.powell_2B_TT_ATM_TT,
            "powell_2B_BJ_supramolecular": args.supramolecular_BJ,
            "powell_2B_TT_supramolecular": args.supramolecular_TT,
        },
        D3={"powell": args.D3},
        ATM=args.ATM,
        extra=args.extra_label,
        use_2B_C6s=args.use_2B_C6s,
        five_fold=args.five_fold,
    )
    return


if __name__ == "__main__":
    main()
