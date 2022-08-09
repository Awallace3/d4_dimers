import pandas as pd
from src.setup import (
    gather_data4,
    gather_data5,
    gather_data2_testing_mol,
    inpsect_master_regen,
    assign_charges,
)
from src.optimization import optimization, opt_cross_val, HF_only, find_max_e
from src.jobs import (
    create_hf_binding_energies_jobs,
    run_sapt0_example,
    fix_hf_charges_energies_jobs,
)
from src.harvest import ssi_bfdb_data, harvest_data
from src.compare import error_stats_method, analyze_diffs

"""
dftd4/src/dftd4/param.f90
    case(p_hf)
      param = dftd_param ( & ! (SAW190103)
         &  s6=1.0000_wp, s8=1.61679827 _wp, a1=0.44959224 _wp, a2=3.35743605 _wp )
      !  Fitset: MD= -0.02597 MAD= 0.34732 RMSD= 0.49719
]
"""

# TODO: use --pair-resolved to sum over fragment pieces to acquire comparison for compute_bj_pairs
# CP HF_jdz IE = 14.829109041411 kcal/mol for 1466
# IE (pairs)   = 14.788696708055493
# CP HF_jdz IE = 27.231587582085 kcal/mol for 7265
# IE (pairs)   = 27.24627373445821

# look at System names to assign charges


def get_params():
    return {
        # "HF_jdz": [0.57791015, 0.67223747, 0.98740773],
        # "HF_adz": [0.5971246, 0.64577916, 1.17106229],
        # "HF_jdz": [1.61679827, 0.44959224, 3.35743605],
        # "HF_adz": [1.61679827, 0.44959224, 3.35743605],
        # "HF_jdz": [1.61679827, 0.44959224, 3.35743605],
        "HF_jdz": [0.86639457, 0.77313591, 0.79705373],
        "HF_adz": [0.87605397, 0.74188409, 0.98700505],
        "HF_dz": [0.83376653, 0.71675200, 1.01439706],
        "HF_tz": [0.85959583, 0.69363123, 1.21911775],
    }


def analyze_max_errors(
    df,
    count: int = 5,
) -> None:
    """
    analyze_max_errors looks at largest max errors
    """
    params_dc = get_params()
    for k, v in params_dc.items():
        find_max_e(df, v, k, count)
    return


# TODO: Show analyze_max_errors in comparison to jeff.out
# TODO: error is in the C6s before dimer splits...


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # gather_data5(output_path="opt5.pkl", from_master=False, HF_columns=["HF_tz"])
    # df = pd.read_pickle("base.pkl")
    # df = ssi_bfdb_data(df)
    # compute_values()
    # compute_values(7265)

    df = pd.read_pickle("opt6.pkl")
    # assign_charges(df)
    # df.to_pickle("opt6.pkl")

    # print(df["HF_tz"].isna().sum())
    # print(379 + 575)
    # print(379 + 575 - 205)
    # df = df[df["DB"].isin(["SSI", "BBI"])]
    # df = df[df["DB"].isin(["SSI"])]
    # # BBI doesn't have any charged
    # df = df[["HF_jdz","HF_tz"]]
    # print(df["HF_tz"].isna().sum())
    # for idx, i in df.iterrows():
    #     print(i)
    # inpsect_master_regen()
    # analyze_max_errors(df)
    # analyze_diffs(df, get_params(), hf_cols=["HF_jdz"])
    # analyze_diffs(df, params_dc, hf_cols=["HF_jdz", "HF_adz", "HF_dz"])

    # s0, mae_s0, rmse_s0, max_e_s0 = error_stats_method(df, method="SAPT0")
    print(df.DB.unique())
    return
    # df = df[df["DB"].isin(["S66by10", "S22by7"])]
    # print(df["HF_tz"].isna().sum())
    # basis_set = "adz"
    # hf_key = "HF_%s" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # params = [
    #     1.833767,
    #     0.716752,
    #     1.014397,
    # ]
    # opt_cross_val(df, nfolds=5, start_params=params, hf_key=hf_key)

    bases = ["dz", "tz"]
    # bases = ["atz"]
    fix_hf_charges_energies_jobs("opt6.pkl", bases)
    # create_hf_binding_energies_jobs(
    #     "base1.pkl",
    #     bases,
    #     "calc",
    #     "dimer",
    #     "4gb",
    #     10,
    #     6,
    #     "99:00:00",
    # )
    return


def compute_values(id=1466) -> None:
    """
    compute_values ...
    """
    df = pd.read_pickle("opt.pkl")
    mol = df.loc[id]
    print(mol)
    gather_data2_testing_mol(mol)
    return


if __name__ == "__main__":
    main()
