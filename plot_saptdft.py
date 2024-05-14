import src
import pandas as pd

def merge_basis_study():
    df = pd.read_pickle("./plots/basis_study.pkl")
    print(df.columns.values)
    df2 = pd.read_pickle("./plots/los_saptdft_atz_2.pkl")
    return

def check_c6s(df):
    # if df.iloc[0]['C6s'] is None:
    df = src.setup.generate_D4_data(df)
    return df

def main():
    # merge_basis_study()
    # return
    df_name = "./dfs/los_saptdft_atz_fixed.pkl"
    df = pd.read_pickle(df_name)
    df['C6s'] = df['C6']
    print(df['C6s'])
    print(len(df))
    print(df['DB'].unique())
    print(df['C6s'])
    # print number of null C6s
    print('null c6s:', df['C6s'].isnull().count())
    assert df['C6s'].notnull().all()

    # df = check_c6s(df)
    # print(df['C6s'])
    # print(f"{df_name = }")
    # df.to_pickle(df_name)
    # df = src.misc.make_geometry_bohr_column_df(df)
    # df.to_pickle(df_name)
    df.dropna(subset=['SAPT_DFT_pbe0_adz', 'SAPT_DFT_pbe0_atz', "C6s"], inplace=True)

        
    src.plotting.plotting_setup_dft_ddft(
        df_name,
        build_df=True,
        split_components=True,
    )
    return


if __name__ == "__main__":
    main()
