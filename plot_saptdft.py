import pandas as pd
import src
import subprocess, os


def main():
    df_name = "dfs/los_saptdft_atz.pkl"
    src.plotting.plotting_setup_dft_ddft(
        df_name,
        build_df=True,
        split_components=True,
    )
    return


if __name__ == "__main__":
    main()
