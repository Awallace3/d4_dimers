import src
import os
import subprocess
import pandas as pd
from pprint import pprint as pp
from qcelemental import constants
import numpy as np

hartree2kcalmol = constants.hartree2kcalmol

def main():
    df_name = "./plots/ddft_study.pkl"
    if not os.path.exists(df_name):
        print("Cannot find ./plots/ddft_study.pkl, creating it now...")
        subprocess.call(
            "cat plots/ddft_study-* > plots/ddft_study.pkl.tar.gz", shell=True
        )
        subprocess.call("tar -xzf plots/ddft_study.pkl.tar.gz", shell=True)
        subprocess.call("rm plots/ddft_study.pkl.tar.gz", shell=True)
        subprocess.call("mv ddft_study.pkl plots/ddft_study.pkl", shell=True)
    regen = False
    if regen:
        if os.path.exists("./dfs/LoS_total_full_dfs_D3-ML.pkl"):
            os.remove("./dfs/LoS_total_full_dfs_D3-ML.pkl")
            os.remove("./dfs/LoS_total_subset_dfs_D3-ML.pkl")
            os.remove("./dfs/LoS_components_full_dfs_D3-ML.pkl")
            os.remove("./dfs/LoS_components_subset_dfs_D3-ML.pkl")

    df = src.plotting.plotting_setup_dft_ddft(
        # df_name,
        "./plots/LoS.pkl",
        build_df=regen,
        # build_df=False,
        df_out="./plots/LoS_ddft.pkl",
        original_plot=False,
    )
    # return
    # src.plotting.plot_LoS_saptdft(
    #     df,
    #     presentation=True,
    # )
    # src.plotting_saptdft.plot_LoS_saptdft(
    #     df,
    #     presentation=True,
    # )

    # df = pd.read_pickle("./dfs/LoS_components_full_dfs.pkl")[1]['df']
    pp(df.columns.tolist())
    df = df[df['DB'] == 's66x8']
    system_ids = [
        '01_Water-Water_1.00',
    ]
    df = df[df["system_id"].isin(system_ids)]
    df['SAPT(PBE0)-D3 INTER DISP ENERGY atz'] = df['SAPT(PBE0)-D3 INTER DISP ENERGY atz'].astype(float) * hartree2kcalmol
    df['SAPT2+3(CCD) DISP ENERGY atz'] = df['SAPT2+3(CCD) DISP ENERGY atz'].astype(float) * hartree2kcalmol
    print(df[['system_id', 'SAPT(PBE0)-D3 INTER DISP ENERGY atz', 'SAPT2+3(CCD) DISP ENERGY atz']])
    # np print for array as list 6 decimal places
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    np.set_printoptions(precision=6, suppress=True)
    # no truncation
    np.set_printoptions(threshold=np.inf)
    print(df.iloc[0]['D3Data'])
    print(df.iloc[0]['qcel_molecule'])
    return


if __name__ == "__main__":
    main()
