import src
import pandas as pd
import numpy as np
import qcelemental as qcel

h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

def merge_basis_study():
    df = pd.read_pickle("./plots/basis_study.pkl")
    print(df.columns.values)
    df2 = pd.read_pickle("./plots/los_saptdft_atz_2.pkl")
    return

def check_c6s(df):
    if df.iloc[0]['C6s'] is None:
        df = src.setup.generate_D4_data(df)
    return df

def main():
    # merge_basis_study()
    # return
    # df_name = "./dfs/los_adz_candidacy_s0atz.pkl"
    df_name = "./dfs/los_saptdft_adz_3.pkl"
    df_name = "./dfs/los_adz_candidacy_s0atz.pkl"
    df = pd.read_pickle(df_name)
    print(df.columns.values)
    df = check_c6s(df)
    df_name = "./dfs/los_adz_candidacy_s0atz.pkl"
    # df.to_pickle(df_name)
    # print('null c6s:', df['C6s'].isnull().count())
    assert df['C6s'].notnull().all()

    df_name = "./dfs/los_adz_candidacy_s0atz.pkl"
    # df = check_c6s(df)
    # df.to_pickle(df_name)
    print(f"{df_name = }")
    # df = src.misc.make_geometry_bohr_column_df(df)
    df.to_pickle(df_name)
    df.dropna(subset=['SAPT_DFT_pbe0_adz', 'SAPT_DFT_pbe0_atz', "C6s"], inplace=True)
    # return
        
    src.plotting.plotting_setup_dft_ddft(
        df_name,
        build_df=True,
        split_components=True,
    )
    return


if __name__ == "__main__":
    main()
