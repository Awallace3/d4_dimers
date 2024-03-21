import pandas as pd
import src
import subprocess, os


def main():
    df_name = "dfs/los_saptdft.pkl"
    df = pd.read_pickle(df_name)
    if 'Geometry_bohr' not in df.columns:
        df = src.misc.make_geometry_bohr_column_df(df)
        df.to_pickle(df_name)
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
