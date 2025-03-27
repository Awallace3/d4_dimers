import src
import pandas as pd
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
    df_sys = df[df['System Label'] == '50_Benzene-Ethyne'].copy()
    print(df_sys[['R', '-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)']])
    pp(df.columns.values.tolist())
    print(
        df[
            [
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)",
                "-D4 (HF)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)",
            ]
        ]
    )
    print(
        df[
            [
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)",
                "-D4 (HF)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)",
            ]
        ].describe()
    )
    print(df["SAPT_DFT_pbe0_adz"])
    df["PBE0(SAPT)_adz_DIFF_REF"] = df.apply(
        lambda r: sum(r["SAPT_DFT_pbe0_adz"][1:4])
        + r["SAPT_DFT_pbe0_adz_dDFT"]
        - r["SAPT_DFT_pbe0_adz_dHF"]
        - r["benchmark ref energy"],
        axis=1,
    )
    print(df["PBE0(SAPT)_adz_DIFF_REF"])
    df.to_pickle("./plots/train.pkl")
    return
    # df = src.plotting.prep_saptdft_components(df, "pbe0", "adz")
    # pp(df.columns.values.tolist())
    # df["SAPT(DFT) [PBE0] Sum"] = df.apply(
    #     lambda r: r["SAPT(DFT) [PBE0] ELST ENERGY adz"]
    #     + r["SAPT(DFT) [PBE0] EXCH ENERGY adz"]
    #     + r["SAPT(DFT) [PBE0] IND ENERGY adz"]
    #     + r['SAPT_DFT_pbe0_adz_DFT_IE'],
    #     axis=1,
    # )
    # df.to_pickle("./plots/ddft_study.pkl")
    # return
    src.plotting.plot_LoS_saptdft(df, presentation=False)
    return


if __name__ == "__main__":
    main()
