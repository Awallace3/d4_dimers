import numpy as np
import pandas as pd
from pprint import pprint as pp
import qcelemental as qcel
from glob import glob

h2kcalmol = qcel.constants.hartree2kcalmol


keep_columns = [
    "id",
    "benchmark ref energy",
    "DB",
    "system_id",
    "Geometry",
    "monAs",
    "monBs",
    "coordinates",
    "atomic_numbers",
    "dimer_charge",
    "dimer_multiplicity",
    "monA_charge",
    "monA_multiplicity",
    "monB_charge",
    "monB_multiplicity",
    "Benchmark",
    "subset",
    "System Label",
    "R",
]


def elimination(df=None):
    if df is None:
        df = pd.read_pickle("./plots/LoS_ddft.pkl")
    # filter out columns that have *b2plyp* and *wb97* in their name, case insensitive
    filter_list = [
        "b2plyp",
        "wb97",
        "[pbe0] dmp2",
        "d4 dmp2",
        "dlpno",
        "b97",
        "n_oelect",
        "nbf",
        "schr_id",
        "original_id",
        "pcm pol",
        "dipole",
        "scf iteration",
        "scf total",
        "scs-mp2",
        "geometry_bohr",
        "geometry",
        "sapt(dft)+d4",
        "disp_a",
        "disp_b",
        "disp_c",
        "disp_d",
        "ref_",
        "mp2 opposite-spin",
        "mp2 singles",
        "ssapt0",
        "dedispdt",
        "dft-d4",
        "sapt(dft)-d4",
        "two-electron",
        "error",
        "sapt2+dmp2 ",
        "sapt2+3 ",
        "sapt2 ",
        "sapt2+ ",
        "sapt ",
        "sapt2+(3)",
        "hf ",
        "mp2 ",
        "sapt2+(ccd) ",
        "sapt2+3dmp2 ",
        "scs-",
        # 'ccsd_t_ie_adz'
        "mp2_atqz_diff",
        "homo",
        "_cation_c6_a",
        "c6_a",
        "c6_atm",
        "c6_b",
        "c6s",
        "d3data",
        "_diff",
        "_pre_d4",
        "plus_d4",
        "no_damping",
        "unnamed",
    ]
    filtered_cols = [
        c for c in df.columns if not any(f in c.lower() for f in filter_list)
    ]
    df = df[filtered_cols]
    df.drop(
        columns=[
            "SAPT0_adz",
            "SAPT0_atz",
            "SAPT0_aqz",
            "CCSD_T_IE_adz",
            "SAPT_DFT_pbe0_adz",
            "SAPT_DFT_pbe0_atz",
            "SAPT_DFT_pbe0_aqz",
            "SAPT_DFT_b3lyp_adz",
            "SAPT_DFT_b3lyp_atz",
            "SAPT_DFT_b3lyp_aqz",
            "pbe0_adz_A",
            "pbe0_adz_B",
            "pbe0_adz_cation_A",
            "pbe0_adz_cation_B",
            "pbe0_adz_delta_DFT",
            "pbe0_adz_delta_HF",
            "pbe0_atz_A",
            "pbe0_atz_B",
            "pbe0_atz_cation_A",
            "pbe0_atz_cation_B",
            "pbe0_atz_delta_DFT",
            "pbe0_atz_delta_HF",
            "b3lyp_adz_A",
            "b3lyp_adz_B",
            "b3lyp_adz_cation_A",
            "b3lyp_adz_cation_B",
            "b3lyp_adz_delta_DFT",
            "b3lyp_adz_delta_HF",
            "SAPT_DFT_adz",
            "SAPT_DFT_pbe0_adz_d4_total",
            "LoS_subset",
            "SAPT_DFT_D4_pbe0_adz_total",
            "SAPT_DFT_D4_pbe0_atz_total",
            "SAPT_DFT_D4_pbe0_aqz_total",
            "SAPT_DFT_D4_b3lyp_adz_total",
            "SAPT_DFT_D4_b3lyp_atz_total",
            "SAPT_DFT_D4_b3lyp_aqz_total",
            "-D4 (SAPT0_adz_3_IE)",
            "-D4 (SAPT0_atz_3_IE)",
            "-D4 (SAPT0_aqz_3_IE)",
        ],
        inplace=True,
    )
    df.rename(
        columns={
            "SAPT0-D4 (I) TOTAL ENERGY adz": "SAPT0-D4 INTER TOTAL ENERGY adz",
            "SAPT0-D4 (I) TOTAL ENERGY atz": "SAPT0-D4 INTER TOTAL ENERGY atz",
            "SAPT0-D4 (I) TOTAL ENERGY aqz": "SAPT0-D4 INTER TOTAL ENERGY aqz",
            "SAPT0-D4 (I) DISP ENERGY adz": "SAPT0-D4 INTER DISP ENERGY adz",
            "SAPT0-D4 (I) DISP ENERGY atz": "SAPT0-D4 INTER DISP ENERGY atz",
            "SAPT0-D4 (I) DISP ENERGY aqz": "SAPT0-D4 INTER DISP ENERGY aqz",
        },
        inplace=True,
    )
    # assert df['subset'] == df['LoS_subset']
    # print(len(df.columns))
    pp(df)
    pp(df.columns.to_list())
    conv_col = [
        "benchmark ref energy",
        "B3LYP-D3 IE adz",
        "B3LYP-D3 IE atz",
        "B3LYP-D3 IE aqz",
        "B3LYP-D4 IE adz",
        "B3LYP-D4 IE atz",
        "B3LYP-D4 IE aqz",
        # PBE0
        "PBE0-D3 IE adz",
        "PBE0-D3 IE atz",
        "PBE0-D3 IE aqz",
        "PBE0-D4 IE adz",
        "PBE0-D4 IE atz",
        "PBE0-D4 IE aqz",
        "E_R_eq_elst_aqz",
        "E_R_eq_elst_atz",
        "E_R_eq_exch_aqz",
        "E_R_eq_exch_atz",
        "E_R_eq_ind_aqz",
        "E_R_eq_ind_atz",
        # misc
        "SAPT(B3LYP)-D4 INTER DISP ENERGY",
        "SAPT(B3LYP)-D3 SUPER DISP ENERGY",
        "SAPT(DFT) [PBE0] Sum",
        "SAPT(PBE0)-D4 INTER DISP ENERGY",
        # sapt
        "SAPT0-D4/aDZ",
        "SAPT0_adz_3_IE",
        "SAPT0_adz_d4",
        "SAPT0_adz_disp",
        "SAPT0_adz_elst",
        "SAPT0_adz_exch",
        "SAPT0_adz_indu",
        "SAPT0_adz_total",
        "SAPT0_aqz_3_IE",
        "SAPT0_aqz_d4",
        "SAPT0_aqz_disp",
        "SAPT0_aqz_elst",
        "SAPT0_aqz_exch",
        "SAPT0_aqz_indu",
        "SAPT0_aqz_total",
        "SAPT0_atz_3_IE",
        "SAPT0_atz_d4",
        "SAPT0_atz_disp",
        "SAPT0_atz_elst",
        "SAPT0_atz_exch",
        "SAPT0_atz_indu",
        "SAPT0_atz_total",
        # sapt_dft, screen out totals
        "SAPT_DFT_b3lyp_adz_3_IE",
        "SAPT_DFT_b3lyp_adz_D3_IE",
        "SAPT_DFT_b3lyp_adz_D4_IE",
        "SAPT_DFT_b3lyp_adz_DFT_IE",
        "SAPT_DFT_b3lyp_adz_d4_disp",
        "SAPT_DFT_b3lyp_adz_dDFT",
        "SAPT_DFT_b3lyp_adz_dHF",
        "SAPT_DFT_b3lyp_adz_disp",
        "SAPT_DFT_b3lyp_adz_elst",
        "SAPT_DFT_b3lyp_adz_exch",
        "SAPT_DFT_b3lyp_adz_indu",
        "SAPT_DFT_b3lyp_aqz_3_IE",
        "SAPT_DFT_b3lyp_aqz_D3_IE",
        "SAPT_DFT_b3lyp_aqz_D4_IE",
        "SAPT_DFT_b3lyp_aqz_DFT_IE",
        "SAPT_DFT_b3lyp_aqz_d4_disp",
        "SAPT_DFT_b3lyp_aqz_dDFT",
        "SAPT_DFT_b3lyp_aqz_dHF",
        "SAPT_DFT_b3lyp_aqz_disp",
        "SAPT_DFT_b3lyp_aqz_elst",
        "SAPT_DFT_b3lyp_aqz_exch",
        "SAPT_DFT_b3lyp_aqz_indu",
        "SAPT_DFT_b3lyp_atz_3_IE",
        "SAPT_DFT_b3lyp_atz_D3_IE",
        "SAPT_DFT_b3lyp_atz_D4_IE",
        "SAPT_DFT_b3lyp_atz_DFT_IE",
        "SAPT_DFT_b3lyp_atz_d4_disp",
        "SAPT_DFT_b3lyp_atz_dDFT",
        "SAPT_DFT_b3lyp_atz_dHF",
        "SAPT_DFT_b3lyp_atz_disp",
        "SAPT_DFT_b3lyp_atz_elst",
        "SAPT_DFT_b3lyp_atz_exch",
        "SAPT_DFT_b3lyp_atz_indu",
        "SAPT_DFT_pbe0_adz_3_IE",
        "SAPT_DFT_pbe0_adz_D3_IE",
        "SAPT_DFT_pbe0_adz_D4_IE",
        "SAPT_DFT_pbe0_adz_DFT_IE",
        "SAPT_DFT_pbe0_adz_d4_disp",
        "SAPT_DFT_pbe0_adz_dDFT",
        "SAPT_DFT_pbe0_adz_dHF",
        "SAPT_DFT_pbe0_adz_disp",
        "SAPT_DFT_pbe0_adz_elst",
        "SAPT_DFT_pbe0_adz_exch",
        "SAPT_DFT_pbe0_adz_indu",
        "SAPT_DFT_pbe0_aqz_3_IE",
        "SAPT_DFT_pbe0_aqz_D3_IE",
        "SAPT_DFT_pbe0_aqz_D4_IE",
        "SAPT_DFT_pbe0_aqz_DFT_IE",
        "SAPT_DFT_pbe0_aqz_d4_disp",
        "SAPT_DFT_pbe0_aqz_dDFT",
        "SAPT_DFT_pbe0_aqz_dHF",
        "SAPT_DFT_pbe0_aqz_disp",
        "SAPT_DFT_pbe0_aqz_elst",
        "SAPT_DFT_pbe0_aqz_exch",
        "SAPT_DFT_pbe0_aqz_indu",
        "SAPT_DFT_pbe0_atz_3_IE",
        "SAPT_DFT_pbe0_atz_D3_IE",
        "SAPT_DFT_pbe0_atz_D4_IE",
        "SAPT_DFT_pbe0_atz_DFT_IE",
        "SAPT_DFT_pbe0_atz_d4_disp",
        "SAPT_DFT_pbe0_atz_dDFT",
        "SAPT_DFT_pbe0_atz_dHF",
        "SAPT_DFT_pbe0_atz_disp",
        "SAPT_DFT_pbe0_atz_elst",
        "SAPT_DFT_pbe0_atz_exch",
        "SAPT_DFT_pbe0_atz_indu",
        "-D4 (HF)",
        "-D4 (HF_ATM)",
        "-D4 (SAPT0_adz_3_IE)",
        "-D4 (SAPT0_adz_3_IE_2B_BJ_inter)",
        "-D4 (SAPT0_aqz_3_IE)",
        "-D4 (SAPT0_atz_3_IE)",
        "-D4 (SAPT_DFT_b3lyp_adz_3_IE_inter)",
        "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
        "-D4 (SAPT_DFT_pbe0_adz_3_IE_inter)",
        "-D4 (SAPT_DFT_pbe0_aqz_3_IE)",
        "-D4 (SAPT_DFT_pbe0_atz_3_IE)",
    ]
    for c in conv_col:
        df[c] = df[c] / h2kcalmol
    # print(
    #     df[
    #         [
    #             "PBE0-D4 IE aqz",
    #             "benchmark ref energy",
    #         ]
    #     ]
    # )
    # df.to_pickle(f"./si_data/LoS_full_{len(df)}.pkl")
    df.to_csv(f"./si_data/LoS_full_{len(df)}.csv", index=True)
    df_subset = df[df["subset"]]
    # df_subset.to_pickle(f"./si_data/LoS_si_{len(df_subset)}.pkl")
    df_subset.to_csv(f"./si_data/LoS_si_{len(df_subset)}.csv", index=True)
    # print(df)
    # print(df_subset)
    pp(df.iloc[0].to_dict())
    return df


def main():
    df_los_ddft = pd.read_pickle("./plots/LoS_ddft.pkl")
    pkl_files = glob("./dfs/LoS_*.pkl")
    full_dfs = []
    subset_dfs = []
    for pkl in pkl_files:
        if "_si_" in pkl:
            continue
        print(f"Processing {pkl}...")
        d = pd.read_pickle(pkl)
        print(d)
        for i in d:
            df = i["df"]
            df = elimination()
            if i["basis"] == "aug-cc-pVDZ":
                bs_str = " adz"
            elif i["basis"] == "aug-cc-pVTZ":
                bs_str = " atz"
            elif i["basis"] == "aug-cc-pVQZ":
                bs_str = " aqz"
            # rename all columns to have bs_str at the end
            df.rename(columns={c: c + bs_str for c in df.columns}, inplace=True)
            if "full" in pkl:
                full_dfs.append(df)
            elif "subset" in pkl:
                subset_dfs.append(df)
    full_df = pd.concat(full_dfs, axis=1)
    pp(full_df)
    pp(full_df.columns.to_list())
    # full_df.to_pickle(f"./si_data/LoS_full_{len(full_df)}.pkl")
    # subset_df = pd.concat(subset_dfs, axis=1)
    # pp(subset_df)
    # pp(subset_df.columns.to_list())
    # subset_df.to_pickle(f"./si_data/LoS_si_{len(subset_df)}.pkl")
    return


if __name__ == "__main__":
    elimination()
    df = pd.read_csv("./si_data/LoS_full_4558.csv")
    print(df)
    pp(df.iloc[0].to_dict())
    # main()
