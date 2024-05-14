import src
import pandas as pd

def merge_basis_study():
    df = pd.read_pickle("./plots/basis_study.pkl")
    print(df.columns.values)
    df2 = pd.read_pickle("./plots/los_saptdft_atz_2.pkl")
    return

def check_c6s(df):
    if df.iloc[0]['C6s'] is None:
        src.setup.generate_D4_data(df)
    return df

def main():
    # merge_basis_study()
    # return
    df_name = "./dfs/los_saptdft_atz_fixed.pkl"
    df = pd.read_pickle(df_name)
    df = check_c6s(df)
    df.to_pickle(df_name)
    df["Geometry_bohr"] = df.apply(lambda x: src.misc.make_bohr(x["Geometry"], True), axis=1)
    df.to_pickle(df_name)
        
    src.plotting.plotting_setup_dft_ddft(
        df_name,
        build_df=True,
        split_components=True,
    )
    return


if __name__ == "__main__":
    main()
