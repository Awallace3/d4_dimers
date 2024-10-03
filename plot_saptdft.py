import src
import pandas as pd
import numpy as np
import qcelemental as qcel
from pprint import pprint as pp

h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")


def merge_basis_study():
    df = pd.read_pickle("./plots/basis_study.pkl")
    print(df.columns.values)
    df2 = pd.read_pickle("./plots/los_saptdft_atz_2.pkl")
    return


def check_c6s(df):
    if "C6s" not in df.columns.values:
        df = src.setup.generate_D4_data(df)
        return df
    if df.iloc[0]["C6s"] is None:
        df = src.setup.generate_D4_data(df)
    return df


def main():
    # df_name = "./dfs/los_adz_candidacy_s0atz.pkl"
    # df_name = "./dfs/los_saptdft_adz_3.pkl"
    # df_name = "./dfs/los_all.pkl"
    df_name = "./dfs/los_all.pkl"
    # df_name = "./dfs/ddft_study.pkl"
    # df = pd.read_pickle(df_name)
    # df = check_c6s(df)
    # df = src.misc.make_geometry_bohr_column_df(df)
    # df.to_pickle(df_name)
    # assert df['C6s'].notnull().all()

    # pp(df.columns.values.tolist())
    df = src.plotting.plotting_setup_dft_ddft(
        df_name,
        build_df=False,
        split_components=True,
        original_plot=False,
    )
    # return
    # src.plotting.plot_components_sapt0_saptdft(df)
    # return
    # TODO: plot dispersion as pull apart
    src.plotting.plot_LoS_saptdft(df, presentation=False)
    return


if __name__ == "__main__":
    main()
