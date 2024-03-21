import pandas as pd
import src
import subprocess, os


def main():
    df_name = "dfs/los_saptdft.pkl"
    df_name = "dfs/los_saptdft_atz.pkl"
    df_name = "dfs/los_saptdft_aqz.pkl"
    df = pd.read_pickle(df_name)
    print(df.columns.values)
    # df = df[df['SAPT_DFT_adz'].notnull()]
    # print(df)
    # df.rename(columns={'SAPT_DFT_adz': 'SAPT_DFT_pbe0_adz'}, inplace=True)
    # print(df.columns.values)
    # df.to_pickle(df_name)
    # if 'Geometry_bohr' not in df.columns:
    #     df = src.misc.make_geometry_bohr_column_df(df)
    #     df.to_pickle(df_name)
    # if 'C6' not in df.columns:
    #     df = src.setup.generate_D4_data(df)
    #     df.to_pickle(df_name)
    # print(df.columns.values)
    print(len(df))
    src.plotting.plotting_setup_dft_ddft(
        df,
        df_name,
    )
    return


if __name__ == "__main__":
    main()
