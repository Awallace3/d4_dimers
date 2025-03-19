import src
import pandas as pd
import qcelemental as qcel

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
    print(
        df[
            [
                # "-D4 (HF)",
                # "-D4 (HF_ATM)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
            ]
        ]
    )
    print(
        df[
            [
                # "-D4 (HF)",
                # "-D4 (HF_ATM)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)",
                "-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)",
            ]
        ]
    )
    # print(df.columns.values.tolist())
    df['SAPT2+3(CCD) DISP ENERGY atz kcal'] = df['SAPT2+3(CCD) DISP ENERGY atz'] * h2kcalmol
    print(
        df[
            [
                "-D4 (HF)",
                "-D4 (HF_ATM)",
                "SAPT2+3(CCD) DISP ENERGY atz kcal"
                # "-D4 (SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING)",
                # "-D4 (SAPT_DFT_pbe0_adz_3_IE)",
            ]
        ]
    )
    df['SAPT(DFT)-D4 SUPRA DISP ENERGY'] = df['-D4 (SAPT_DFT_pbe0_adz_3_IE_supra)']
    df['SAPT(DFT)-D4 SUPRA ND DISP ENERGY'] = df['-D4 (SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING)']
    # return
    # pp(df.columns.tolist())
    # return
    # src.plotting.plot_components_sapt0_saptdft(df)
    # return
    # TODO: plot dispersion as pull apart
    src.plotting.plot_LoS_saptdft(df, presentation=False)
    return


if __name__ == "__main__":
    main()
