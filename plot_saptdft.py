import pandas as pd
import src
import subprocess, os


def main():
    df_name = "dfs/los_saptdft.pkl"
    df = pd.read_pickle(df_name)
    df.rename(columns={'SAPT_DFT_adz': 'SAPT_DFT_pbe0_adz'}, inplace=True)
    df.to_pickle(df_name)
    if 'Geometry_bohr' not in df.columns:
        df = src.misc.make_geometry_bohr_column_df(df)
        df.to_pickle(df_name)
    print(df.columns.values)
    src.plotting.plotting_setup_dft_ddft(
        df,
        df_name,
    )
    return


if __name__ == "__main__":
    main()
