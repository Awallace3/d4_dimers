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


def elimnation():
    df = pd.read_pickle("./plots/LoS_ddft.pkl")
    # filter out columns that have *b2plyp* and *wb97* in their name, case insensitive
    print(len(df.columns))
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
    ]
    filtered_cols = [
        c for c in df.columns if not any(f in c.lower() for f in filter_list)
    ]
    df = df[filtered_cols]
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
    print(len(df.columns))
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
    ]
    for c in conv_col:
        df[c] = df[c] / h2kcalmol
    print(df[["PBE0-D4 IE aqz", "benchmark ref energy"]])
    return


def main():
    df_los_ddft = pd.read_pickle("./plots/LoS_ddft.pkl")
    pkl_files = glob("./dfs/LoS_*.pkl")
    full_dfs = []
    subset_dfs = []
    for pkl in pkl_files:
        print(f"Processing {pkl}...")
        d = pd.read_pickle(pkl)
        for i in d:
            df = i["df"]
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
    subset_df = pd.concat(subset_dfs, axis=1)
    pp(subset_df)
    pp(subset_df.columns.to_list())
    return


if __name__ == "__main__":
    elimnation()
    # main()
