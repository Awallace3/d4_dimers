import pandas as pd
from src.setup import (
    gather_data4,
    gather_data5,
    gather_data6,
    gather_data2_testing_mol,
    inpsect_master_regen,
    assign_charges,
    expand_opt_df,
    compute_pairwise_dispersion,
)
from src.optimization import (
    optimization,
    opt_cross_val,
    HF_only,
    find_max_e,
    optimization_least_squares,
    compute_int_energy_stats,
    compute_int_energy_stats_dftd4_key,
    compute_dftd4_values,
    compute_stats_dftd4_values_fixed,
)
from src.jobs import (
    create_hf_binding_energies_jobs,
    create_hf_dftd4_ie_jobs,
    run_sapt0_example,
    fix_hf_charges_energies_jobs,
    fix_heavy_element_basis_sets,
    fix_heavy_element_basis_sets_dftd4,
)
from src.harvest import ssi_bfdb_data, harvest_data
from src.compare import error_stats_method, analyze_diffs
import numpy as np
from src.grimme_setup import (
    gather_BLIND_geoms,
    create_Grimme_db,
    create_grimme_s22s66blind,
    gather_grimme_from_db,
)
from src.tools import print_cartesians, stats_to_latex_row
import pickle
from src.jeff import compute_error_stats_d3, d3data_stats, optimization_d3

"""
/theoryfs2/ds/amwalla3/miniconda3/envs/pysr_psi4/lib/python3.9/site-packages/psi4/__init__.py
dftd4/src/dftd4/param.f90
    case(p_hf)
      param = dftd_param ( & ! (SAW190103)
         &  s6=1.0000_wp, s8=1.61679827 _wp, a1=0.44959224 _wp, a2=3.35743605 _wp )
      !  Fitset: MD= -0.02597 MAD= 0.34732 RMSD= 0.49719
]
"""


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


def fix_heavy() -> None:
    """
    fix_heavy
    """
    inds = []
    df = pd.read_pickle("opt8.pkl")
    for idx, i in df.iterrows():
        g = df.iloc[idx]["Geometry"]
        if np.any(g[:, 0] > 10):
            inds.append(idx)
    df2 = df.iloc[inds]
    fix_heavy_element_basis_sets(df2)
    fix_heavy_element_basis_sets_dftd4(df2)


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # TODO: add D3 params to overall .pkl
    # ms = inpsect_master_regen()
    # return
    # print('Disp20', ms['Disp20'].isna().sum())
    # ms = ms[['Benchmark', 'System', 'System #', 'Disp20', 'SAPT DISP ENERGY']]
    # ms['B'] = str(ms['Benchmark'])
    # ms['m'] = ms['System'] + str(ms['System #']) + ms['B']

    # gather_data5(
    #     output_path="opt6.pkl",
    #     from_master=True,
    #     # HF_columns=["HF_jdz"],
    #     HF_columns=["HF_dz", "HF_jdz", "HF_adz", "HF_tz", "HF_jdz_dftd4"],
    #     # HF_columns=["HF_jdz_dftd4"],
    #     # HF_columns=["HF_tz"],
    #     # HF_columns=["HF_atz"],
    #     overwrite=False,
    # )
    # gather_data6(
    #     output_path="sr2.pkl",
    #     from_master=False,
    #     # HF_columns=["HF_jdz"],
    #     # HF_columns=["HF_dz", "HF_jdz", "HF_adz", "HF_tz", "HF_jdz_dftd4"],
    #     # HF_columns=["HF_jdz_dftd4"],
    #     HF_columns=["HF_atz", "HF_jtz"],
    #     # HF_columns=["HF_atz"],
    #     overwrite=True,
    # )
    #
    df = pd.read_pickle("sr2.pkl")
    # print('atz', df['HF_atz'].isna().sum())
    # print('jtz', df['HF_jtz'].isna().sum())

    # d3data_stats(df)

    # df = pd.read_pickle("opt9.pkl")
    # df['B'] = str(df['Benchmark'])
    # df['m'] = df['System'] + str(df['System #']) + df['B']
    # df.merge(ms, on='m')
    # df.to_pickle("opt9.pkl")
    # print(df)

    # print(df['HF_atz'])
    # # print(len(df))
    # df['d'] = df['HF_jdz'] - df['HF INTERACTION ENERGY']
    # df = df.sort_values(['d'], ascending=False)
    # # df = df[df['d'].abs() > 1e-3]
    # print(df['d'], df['d'].max(), df['d'].mean(), sep='\n')
    # # df = pd.read_pickle("tests/diffs.pkl")
    # gather_grimme_from_db()
    # df = pd.read_pickle("data/grimme_fitset_db2.pkl")
    # compute_dftd4_values(df, s9="0.0", key="dftd4_disp_ie_grimme_params")
    # compute_dftd4_values(df, s9="1.0", key="dftd4_disp_ie_grimme_params_ATM")
    # df.to_pickle("sr2.pkl")
    # print(df['Disp20'].isna().sum())
    # return
    # pdi, pa, pb, pdisp = [], [], [],[]
    # for i, r in df.iterrows():
    #     pairs, pairs_a, pairs_b, disp = compute_pairwise_dispersion(r)
    #     pdi.append(pairs)
    #     pa.append(pairs_a)
    #     pb.append(pairs_b)
    #     pdisp.append(disp)
    # df['pairs'] = pdi
    # df['pa'] = pa
    # df['pb'] = pb
    # df['pdisp'] = pdisp
    #
    # df.to_pickle("data/grimme_fitset_db3.pkl")
    # return
    # df.to_pickle("opt5_d4.pkl")
    # compute_stats_dftd4_values_fixed(df, fixed_col="dftd4_disp_ie_grimme_params")
    # compute_stats_dftd4_values_fixed(df, fixed_col='dftd4_disp_ie_grimme_params_ATM')

    # # df = pd.read_pickle("tests/diffs_grimme.pkl")
    # compute_int_energy_stats_dftd4_key(df, hf_key='HF_jdz')
    # # print(df.columns.values)
    # print(df)

    # df.to_pickle("tests/diffs_grimme.pkl")
    # df.to_pickle("tests/diffs.pkl")
    # print(df.columns.values)
    # df["HF_diff_abs"] = df["HF_diff"].abs()
    # df = df.sort_values("HF_diff_abs", ascending=False)
    # mu = df["HF_diff"].abs().mean()
    # print("MAE:", mu)

    # for idx, r in df.iterrows():
    #     if abs(r["HF_diff"]) > 1e-6:
    #         print(idx, r["DB"], r["HF_jdz_d4_sum"], r["HF_jdz_dftd4"], r["HF_diff"]) # kcal / mol

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
    # basis_set = "qz_no_df"
    # hf_key = "HF_%s" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # mae, rmse, max_e, mad, mean_diff = compute_int_energy_stats(params, df, hf_key)
    # print("\nStats\n")
    # print("        1. MAE  = %.4f" % mae)
    # print("        2. RMSE = %.4f" % rmse)
    # print("        3. MAX  = %.4f" % max_e)
    # print("        4. MAD  = %.4f" % mad)
    # print("        5. MD   = %.4f" % mean_diff)
    #
    # basis_set = "jdz"
    # hf_key = "HF_%s" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # print("HF_jdz CP")
    # optimization_least_squares(df, params, hf_key=hf_key)
    # df = pd.read_pickle("data/grimme_fitset_db.pkl")
    # df = pd.read_pickle("opt8.pkl")
    # df = df.iloc[[0, 1, 3, 4]]
    # df = pd.read_pickle("data/grimme_fitset_db.pkl")
    # df = pd.read_pickle("opt8.pkl")
    # # print(df.columns.values)


    return
    bases = [
        # "HF_dz",
        # "HF_jdz",
        # "HF_adz",
        # "HF_tz",
        "HF_atz",
        # "HF_jdz_no_cp",
        # "HF_dz_no_cp",
        # "HF_qz",
        # "HF_qz_no_cp",
        # "HF_qz_no_df",
        # "HF_qz_conv_e_4",
    ]

    for i in bases:
        print(i)
        # params = [1.61679827, 0.44959224, 3.35743605]
        params = [0.713190, 0.079541, 3.627854]
        opt_cross_val(
            df,
            nfolds=5,
            start_params=params,
            hf_key=i,
            output_l_marker="D3_",
            optimizer_func=optimization_d3,
            compute_int_energy_stats_func=compute_error_stats_d3,
            opt_type="Powell",
        )
        # opt_cross_val(df, nfolds=5, start_params=params, hf_key=i, output_l_marker="G_", optimizer_func=optimization)
        # opt_cross_val(df, nfolds=5, start_params=params, hf_key=i, output_l_marker="least", optimizer_func=optimization_least_squares)
    # basis_set = "atz"
    # hf_key = "HF_%s" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # # print("HF_dz CP")
    # # optimization_least_squares(df, params, hf_key=hf_key)
    # opt_cross_val(df, nfolds=5, start_params=params, hf_key=hf_key, output_l_marker="")

    # bases = ["jdz"]
    # bases = ["jdz"]
    # create_hf_dftd4_ie_jobs(
    #     # df_p="./tests/td4.pkl",
    #     df_p="data/grimme_fitset_db.pkl",
    #     bases=bases,
    #     data_dir="calcgrimme",
    #     in_file="dimer",
    #     memory="4gb",
    #     nodes=20,
    #     cores=1,
    #     ppn=1,
    #     walltime="40:00:00",
    #     params=[0.44959224, 3.35743605, 16.0, 1.0, 1.61679827, 0.0],
    # )
    # create_hf_binding_energies_jobs(
    #     "opt6.pkl",
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


if __name__ == "__main__":
    main()
