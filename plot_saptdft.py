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
    # if df.iloc[0]['C6s'] is None:
    df = src.setup.generate_D4_data(df)
    return df

def main():
    # merge_basis_study()
    # return
    # df_name = "./dfs/los_saptdft_atz_fixed.pkl"
    df_name = "./dfs/los_adz_candidacy_s0atz.pkl"
    df = pd.read_pickle(df_name)
    # df['C6s'] = df['C6']
    # assert df['C6s'].notnull().all()
    # df.dropna(subset=['SAPT_DFT_pbe0_adz_total', "C6s"], inplace=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # df['diff'] = abs(df['Benchmark'] - df['SAPT_DFT_pbe0_adz_total'])
    # print(df[["system_id", 'SAPT_DFT_pbe0_adz_total', 'Benchmark', "diff"]])
#     df.sort_values(by='diff', inplace=True, 
# ascending=False
#                    )
#     print(df[["system_id", 'Benchmark', "diff"]].head(50))
#     print(df.columns.values)
    # return
    df['SAPT0_adz'] = df.apply(
            lambda x: np.array([
                x['SAPT0 ELST ENERGY adz']+
                x['SAPT0 EXCH ENERGY adz']+
                x['SAPT0 IND ENERGY adz']+
                x['SAPT0 DISP ENERGY adz'],
                x['SAPT0 ELST ENERGY adz'],
                x['SAPT0 EXCH ENERGY adz'],
                x['SAPT0 IND ENERGY adz'],
                x['SAPT0 DISP ENERGY adz'],
            ])* h2kcalmol, axis=1)
    df['SAPT0_atz'] = df.apply(
            lambda x: np.array([
                x['SAPT0 ELST ENERGY atz']+
                x['SAPT0 EXCH ENERGY atz']+
                x['SAPT0 IND ENERGY atz']+
                x['SAPT0 DISP ENERGY atz'],
                x['SAPT0 ELST ENERGY atz'],
                x['SAPT0 EXCH ENERGY atz'],
                x['SAPT0 IND ENERGY atz'],
                x['SAPT0 DISP ENERGY atz'],
            ]) * h2kcalmol, axis=1)
    df['SAPT0_atz_IE'] = df.apply(
        lambda x: x['SAPT0_atz'][0], axis=1) 
    df['SAPT0_adz_IE'] = df.apply(
        lambda x: x['SAPT0_adz'][0], axis=1) 
    df['SAPT0_atz_3_IE'] = df.apply(
        lambda x: sum(x['SAPT0_atz'][1:4]), axis=1) 
    # df['SAPT0_adz_3_IE'] = df.apply(
    #         lambda x: 
    #             (x['SAPT0 ELST ENERGY adz']+
    #             x['SAPT0 EXCH ENERGY adz']+
    #             x['SAPT0 IND ENERGY adz']) * h2kcalmol, axis=1) 
    df['SAPT0_adz_3_IE'] = df.apply(
        lambda x: sum(x['SAPT0_adz'][1:4]), axis=1) 
    # df = src.plotting.compute_d4_from_opt_params(
    #     df,
    #     bases=[
    #         [
    #             "SAPT0_adz_IE",
    #             "SAPT0_adz_3_IE",
    #             "SAPT0_adz_3_IE_2B",
    #             "SAPT0_adz_3_IE",
    #         ],
    #         [
    #             "SAPT0_atz_IE",
    #             "SAPT0_atz_3_IE",
    #             "SAPT0_atz_3_IE_2B",
    #             "SAPT0_atz_3_IE",
    #         ],
    #     ],
    # )
    df['Benchmark'] = df['benchmark ref energy']
    df.to_pickle(df_name)
    # print(df.columns.values)
    src.plotting.plotting_setup_dft_ddft(
        df_name,
        build_df=True,
        split_components=True,
    )
    return


if __name__ == "__main__":
    main()
