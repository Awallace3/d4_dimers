import pandas as pd
from src.setup import (
    gather_data4,
    gather_data5,
    gather_data2_testing_mol,
    inpsect_master_regen,
    assign_charges,
    expand_opt_df,
)
from src.optimization import (
    optimization,
    opt_cross_val,
    HF_only,
    find_max_e,
    optimization_least_squares,
    compute_int_energy_stats,
    compute_int_energy_stats_dftd4_key,
)
from src.jobs import (
    create_hf_binding_energies_jobs,
    create_hf_dftd4_ie_jobs,
    run_sapt0_example,
    fix_hf_charges_energies_jobs,
)
from src.harvest import ssi_bfdb_data, harvest_data
from src.compare import error_stats_method, analyze_diffs
import numpy as np
from src.grimme_setup import (
    gather_BLIND_geoms,
    create_Grimme_db,
    create_grimme_s22s66blind,
)
from src.tools import print_cartesians
import pickle

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
        "HF_jdz": [1.61679827, 0.44959224, 3.35743605],
        "HF_adz": [1.61679827, 0.44959224, 3.35743605],
        "HF_dz": [1.61679827, 0.44959224, 3.35743605],
        "HF_tz": [1.61679827, 0.44959224, 3.35743605],
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
    # gather_data5(
    #     output_path="opt8.pkl",
    #     from_master=True,
    #     # HF_columns=["HF_atz"],
    #     HF_columns=["HF_dz", "HF_jdz", "HF_adz", "HF_tz", "HF_jdz_dftd4"],
    #     # HF_columns=["HF_jdz_dftd4"],
    #     # HF_columns=["HF_dz", "HF_tz"],
    #     overwrite=True,
    # )

    # df = pd.read_pickle("opt8.pkl")
    df = pd.read_pickle("tests/diffs.pkl")
    print(df['C6s'])
    # print(df1.iloc[6165]['HF_jdz_dftd4'])
    # print(df.iloc[6165]['HF_jdz_dftd4'])
    # df = pd.read_pickle("opt6.pkl")
    # df = pd.read_pickle("tests/diffs.pkl")
    # compute_int_energy_stats_dftd4_key(df, hf_key='HF_jdz')
    df.to_pickle("tests/diffs.pkl")
    print(df['HF_jdz_d4'])
    print(len(df))
    print(df.columns.values)
    # print(df['HF_diff'].to_list())
    df["HF_diff_abs"] = df["HF_diff"].abs()
    df = df.sort_values("HF_diff_abs", ascending=False)
    mu = df['HF_diff'].abs().mean()
    print("MAE:", mu)

    for idx, r in df.iterrows():
        if abs(r["HF_diff"]) > 1e-1:
            print(idx, r['DB'], r['HF_jdz_d4_sum'], r["HF_jdz_dftd4"], r["HF_diff"])
    #     if r['diff'] < 1e-1:
    #         continue
    #     else:
    #         print(idx, r['diff'], r['d4_jdz'], r['HF_jdz'], r['HF_jdz_dftd4'])

    #
    # create_grimme_s22s66blind()
    # def merge_col(geom):
    #     geom = np.around(geom, decimals=3)
    #     s = sorted(np.array2string(geom))
    #     s = "".join(s).strip()
    #     ban = "-[]+e. "
    #     s = "".join([i for i in s if i not in ban])
    #     return s
    # print(df)
    # analyze_max_errors(df)

    # df = pd.read_pickle("grimme_db.pkl")
    # #
    # print(df)
    # basis_set = "dz"
    # hf_key = "HF_%s" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # mae, rmse, max_e, mad = compute_int_energy_stats(params, df, hf_key)
    # print("\nStats\n")
    # print("        1. MAE  = %.4f" % mae)
    # print("        2. RMSE = %.4f" % rmse)
    # print("        3. MAX  = %.4f" % max_e)
    # print("        4. MAD  = %.4f" % mad)
    #
    # df = pd.read_pickle("data/grimme_fitset_db.pkl")
    # basis_set = "jdz"
    # hf_key = "HF_%s_no_cp" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # print("HF_jdz NO CP")
    # optimization_least_squares(df, params, hf_key=hf_key)
    # basis_set = "jdz"
    # hf_key = "HF_%s" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # print("HF_jdz CP")
    # optimization_least_squares(df, params, hf_key=hf_key)
    # opt_cross_val(df, nfolds=5, start_params=params, hf_key=hf_key)

    # bases = ["jdz"]
    # create_hf_dftd4_ie_jobs(
    #     # df_p="./tests/td4.pkl",
    #     df_p="opt6.pkl",
    #     bases=bases,
    #     data_dir="calc",
    #     in_file="dimer",
    #     memory="4gb",
    #     nodes=20,
    #     cores=1,
    #     ppn=1,
    #     walltime="40:00:00",
    #     params=[0.44959224, 3.35743605, 16.0, 1.0, 1.61679827, 0.0],
    # )
    # bases = ["dz", "tz"]
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
    # fix_hf_charges_energies_jobs('opt6.pkl')

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
