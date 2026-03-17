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
    src.plotting_saptdft.plot_LoS_saptdft(
        df,
        presentation=True,
    )
    return


if __name__ == "__main__":
    main()
