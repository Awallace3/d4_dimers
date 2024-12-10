import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from qcelemental import constants as qcc


def dmp2_correlation_plots(df):
    # create subplot (3x2) that has [df['SAPT MP2(2) ENERGY adz'] vs [df['SAPT2+3 EXCH ENERGY adz'],  df['SAPT2+3 IND ENERGY adz'], df['SAPT2+3 DISP ENERGY adz']]] for the first row and [df['SAPT MP2(3) ENERGY adz'] vs [df['SAPT2+3 EXCH ENERGY adz'],  df['SAPT2+3 IND ENERGY adz'], df['SAPT2+3 DISP ENERGY adz']]] for the second row
    fig, axs = plt.subplots(4, 4, figsize=(10, 10), dpi=400)
    fig.suptitle("DMP2 Correlation Plots")
    # Compute correlation between the SAPT component and the MP2(2) or MP2(3) energy
    df = df[
        [
            "SAPT MP2(2) ENERGY adz",
            "SAPT MP2(3) ENERGY adz",
            "SAPT2+3 ELST ENERGY adz",
            "SAPT2+3 EXCH ENERGY adz",
            "SAPT2+3 IND ENERGY adz",
            "SAPT2+3 DISP ENERGY adz",
            "SAPT MP2(2) ENERGY atz",
            "SAPT MP2(3) ENERGY atz",
            "SAPT2+3 ELST ENERGY atz",
            "SAPT2+3 EXCH ENERGY atz",
            "SAPT2+3 IND ENERGY atz",
            "SAPT2+3 DISP ENERGY atz",
        ]
    ].copy()
    # Convert units to kcal/mol for all columns
    df = df.apply(lambda x: x * qcc.hartree2kcalmol, axis=0)
    corr = df.corr()
    pd.set_option("display.max_columns", None)
    print(corr)
    # row 1
    axs[0, 0].scatter(
        df["SAPT2+3 ELST ENERGY adz"],
        df["SAPT MP2(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 1].scatter(
        df["SAPT2+3 EXCH ENERGY adz"],
        df["SAPT MP2(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 2].scatter(
        df["SAPT2+3 IND ENERGY adz"],
        df["SAPT MP2(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 3].scatter(
        df["SAPT2+3 DISP ENERGY adz"],
        df["SAPT MP2(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 0].set_ylabel("SAPT MP2(2) ENERGY adz")
    axs[0, 0].set_xlabel("SAPT2+3 ELST ENERGY adz")
    axs[0, 1].set_xlabel("SAPT2+3 EXCH ENERGY adz")
    axs[0, 2].set_xlabel("SAPT2+3 IND ENERGY adz")
    axs[0, 3].set_xlabel("SAPT2+3 DISP ENERGY adz")
    # row 2
    axs[1, 0].scatter(
        df["SAPT2+3 ELST ENERGY adz"],
        df["SAPT MP2(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 1].scatter(
        df["SAPT2+3 EXCH ENERGY adz"],
        df["SAPT MP2(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 2].scatter(
        df["SAPT2+3 IND ENERGY adz"],
        df["SAPT MP2(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 3].scatter(
        df["SAPT2+3 DISP ENERGY adz"],
        df["SAPT MP2(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 0].set_ylabel("SAPT MP2(3) ENERGY adz")
    axs[1, 0].set_xlabel("SAPT2+3 ELST ENERGY adz")
    axs[1, 1].set_xlabel("SAPT2+3 EXCH ENERGY adz")
    axs[1, 2].set_xlabel("SAPT2+3 IND ENERGY adz")
    axs[1, 3].set_xlabel("SAPT2+3 DISP ENERGY adz")
    # row 3
    axs[2, 0].scatter(
        df["SAPT2+3 ELST ENERGY atz"],
        df["SAPT MP2(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 1].scatter(
        df["SAPT2+3 EXCH ENERGY atz"],
        df["SAPT MP2(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 2].scatter(
        df["SAPT2+3 IND ENERGY atz"],
        df["SAPT MP2(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 3].scatter(
        df["SAPT2+3 DISP ENERGY atz"],
        df["SAPT MP2(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 0].set_ylabel("SAPT MP2(2) ENERGY atz")
    axs[2, 0].set_xlabel("SAPT2+3 ELST ENERGY atz")
    axs[2, 1].set_xlabel("SAPT2+3 EXCH ENERGY atz")
    axs[2, 2].set_xlabel("SAPT2+3 IND ENERGY atz")
    axs[2, 3].set_xlabel("SAPT2+3 DISP ENERGY atz")
    # row 4
    axs[3, 0].scatter(
        df["SAPT2+3 ELST ENERGY atz"],
        df["SAPT MP2(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 1].scatter(
        df["SAPT2+3 EXCH ENERGY atz"],
        df["SAPT MP2(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 2].scatter(
        df["SAPT2+3 IND ENERGY atz"],
        df["SAPT MP2(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 3].scatter(
        df["SAPT2+3 DISP ENERGY atz"],
        df["SAPT MP2(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 0].set_ylabel("SAPT MP2(3) ENERGY atz")
    axs[3, 0].set_xlabel("SAPT2+3 ELST ENERGY atz")
    axs[3, 1].set_xlabel("SAPT2+3 EXCH ENERGY atz")
    axs[3, 2].set_xlabel("SAPT2+3 IND ENERGY atz")
    axs[3, 3].set_xlabel("SAPT2+3 DISP ENERGY atz")
    for i in range(4):
        for j in range(4):
            axs[i, j].set_xlim(axs[i, j].get_xlim()[0], axs[i, j].get_xlim()[1])
            axs[i, j].set_ylim(axs[i, j].get_ylim()[0], axs[i, j].get_ylim()[1])
            x = np.linspace(axs[i, j].get_xlim()[0], axs[i, j].get_xlim()[1], 100)
            # get MP2(N) index
            corr_y = i if i < 2 else i + 4
            # get SAPT component index
            corr_x = j + 2 if i < 2 else j + 8
            print(i, j, corr_x, corr_y)
            axs[i, j].plot(
                x,
                np.poly1d(np.polyfit(df.iloc[:, corr_x], df.iloc[:, corr_y], 1))(x),
                color="red",
                linestyle="--",
                label=f"Corr. {corr.iloc[corr_x, corr_y]:.2f}",
            )
            axs[i, j].legend()
    plt.tight_layout()
    plt.savefig("./delta_plots/dmp2_correlation_plots.png")
    return


def dhf_correlation_plots(df):
    # create subplot (3x2) that has [df['SAPT MP2(2) ENERGY adz'] vs [df['SAPT2+3 EXCH ENERGY adz'],  df['SAPT2+3 IND ENERGY adz'], df['SAPT2+3 DISP ENERGY adz']]] for the first row and [df['SAPT MP2(3) ENERGY adz'] vs [df['SAPT2+3 EXCH ENERGY adz'],  df['SAPT2+3 IND ENERGY adz'], df['SAPT2+3 DISP ENERGY adz']]] for the second row
    fig, axs = plt.subplots(4, 4, figsize=(10, 10), dpi=400)
    fig.suptitle("DHF Correlation Plots")
    # Compute correlation between the SAPT component and the MP2(2) or MP2(3) energy
    df = df[
        [
            "SAPT HF(2) ENERGY adz",
            "SAPT HF(3) ENERGY adz",
            "SAPT2+3 ELST ENERGY adz",
            "SAPT2+3 EXCH ENERGY adz",
            "SAPT2+3 IND ENERGY adz",
            "SAPT2+3 DISP ENERGY adz",
            "SAPT HF(2) ENERGY atz",
            "SAPT HF(3) ENERGY atz",
            "SAPT2+3 ELST ENERGY atz",
            "SAPT2+3 EXCH ENERGY atz",
            "SAPT2+3 IND ENERGY atz",
            "SAPT2+3 DISP ENERGY atz",
        ]
    ].copy()
    # Convert units to kcal/mol for all columns
    df = df.apply(lambda x: x * qcc.hartree2kcalmol, axis=0)
    corr = df.corr()
    pd.set_option("display.max_columns", None)
    print(corr)
    # row 1
    axs[0, 0].scatter(
        df["SAPT2+3 ELST ENERGY adz"],
        df["SAPT HF(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 1].scatter(
        df["SAPT2+3 EXCH ENERGY adz"],
        df["SAPT HF(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 2].scatter(
        df["SAPT2+3 IND ENERGY adz"],
        df["SAPT HF(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 3].scatter(
        df["SAPT2+3 DISP ENERGY adz"],
        df["SAPT HF(2) ENERGY adz"],
        s=0.8,
    )
    axs[0, 0].set_ylabel("SAPT HF(2) ENERGY adz")
    axs[0, 0].set_xlabel("SAPT2+3 ELST ENERGY adz")
    axs[0, 1].set_xlabel("SAPT2+3 EXCH ENERGY adz")
    axs[0, 2].set_xlabel("SAPT2+3 IND ENERGY adz")
    axs[0, 3].set_xlabel("SAPT2+3 DISP ENERGY adz")
    # row 2
    axs[1, 0].scatter(
        df["SAPT2+3 ELST ENERGY adz"],
        df["SAPT HF(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 1].scatter(
        df["SAPT2+3 EXCH ENERGY adz"],
        df["SAPT HF(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 2].scatter(
        df["SAPT2+3 IND ENERGY adz"],
        df["SAPT HF(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 3].scatter(
        df["SAPT2+3 DISP ENERGY adz"],
        df["SAPT HF(3) ENERGY adz"],
        s=0.8,
    )
    axs[1, 0].set_ylabel("SAPT HF(3) ENERGY adz")
    axs[1, 0].set_xlabel("SAPT2+3 ELST ENERGY adz")
    axs[1, 1].set_xlabel("SAPT2+3 EXCH ENERGY adz")
    axs[1, 2].set_xlabel("SAPT2+3 IND ENERGY adz")
    axs[1, 3].set_xlabel("SAPT2+3 DISP ENERGY adz")
    # row 3
    axs[2, 0].scatter(
        df["SAPT2+3 ELST ENERGY atz"],
        df["SAPT HF(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 1].scatter(
        df["SAPT2+3 EXCH ENERGY atz"],
        df["SAPT HF(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 2].scatter(
        df["SAPT2+3 IND ENERGY atz"],
        df["SAPT HF(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 3].scatter(
        df["SAPT2+3 DISP ENERGY atz"],
        df["SAPT HF(2) ENERGY atz"],
        s=0.8,
    )
    axs[2, 0].set_ylabel("SAPT HF(2) ENERGY atz")
    axs[2, 0].set_xlabel("SAPT2+3 ELST ENERGY atz")
    axs[2, 1].set_xlabel("SAPT2+3 EXCH ENERGY atz")
    axs[2, 2].set_xlabel("SAPT2+3 IND ENERGY atz")
    axs[2, 3].set_xlabel("SAPT2+3 DISP ENERGY atz")
    # row 4
    axs[3, 0].scatter(
        df["SAPT2+3 ELST ENERGY atz"],
        df["SAPT HF(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 1].scatter(
        df["SAPT2+3 EXCH ENERGY atz"],
        df["SAPT HF(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 2].scatter(
        df["SAPT2+3 IND ENERGY atz"],
        df["SAPT HF(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 3].scatter(
        df["SAPT2+3 DISP ENERGY atz"],
        df["SAPT HF(3) ENERGY atz"],
        s=0.8,
    )
    axs[3, 0].set_ylabel("SAPT HF(3) ENERGY atz")
    axs[3, 0].set_xlabel("SAPT2+3 ELST ENERGY atz")
    axs[3, 1].set_xlabel("SAPT2+3 EXCH ENERGY atz")
    axs[3, 2].set_xlabel("SAPT2+3 IND ENERGY atz")
    axs[3, 3].set_xlabel("SAPT2+3 DISP ENERGY atz")
    for i in range(4):
        for j in range(4):
            axs[i, j].set_xlim(axs[i, j].get_xlim()[0], axs[i, j].get_xlim()[1])
            axs[i, j].set_ylim(axs[i, j].get_ylim()[0], axs[i, j].get_ylim()[1])
            x = np.linspace(axs[i, j].get_xlim()[0], axs[i, j].get_xlim()[1], 100)
            # get HF(N) index
            corr_y = i if i < 2 else i + 4
            # get SAPT component index
            corr_x = j + 2 if i < 2 else j + 8
            print(i, j, corr_x, corr_y)
            axs[i, j].plot(
                x,
                np.poly1d(np.polyfit(df.iloc[:, corr_x], df.iloc[:, corr_y], 1))(x),
                color="red",
                linestyle="--",
                label=f"Corr. {corr.iloc[corr_x, corr_y]:.2f}",
            )
            axs[i, j].legend()
    plt.tight_layout()
    plt.savefig("./delta_plots/dhf_correlation_plots.png")
    return


def delta_correlation_plots(
    df,
    label="delta HF",
    delta_terms_bs1=["SAPT HF(2) ENERGY adz", "SAPT HF(3) ENERGY adz"],
    delta_terms_bs2=["SAPT HF(2) ENERGY atz", "SAPT HF(3) ENERGY atz"],
):
    # create subplot (3x2) that has [df['SAPT MP2(2) ENERGY adz'] vs [df['SAPT2+3 EXCH ENERGY adz'],  df['SAPT2+3 IND ENERGY adz'], df['SAPT2+3 DISP ENERGY adz']]] for the first row and [df['SAPT MP2(3) ENERGY adz'] vs [df['SAPT2+3 EXCH ENERGY adz'],  df['SAPT2+3 IND ENERGY adz'], df['SAPT2+3 DISP ENERGY adz']]] for the second row
    fig, axs = plt.subplots(4, 4, figsize=(10, 10), dpi=400)
    fig.suptitle(f"{label} Correlation Plots")
    # Compute correlation between the SAPT component and the MP2(2) or MP2(3) energy
    df = df[
        [
            delta_terms_bs1[0],
            delta_terms_bs1[1],
            "SAPT2+3 ELST ENERGY adz",
            "SAPT2+3 EXCH ENERGY adz",
            "SAPT2+3 IND ENERGY adz",
            "SAPT2+3 DISP ENERGY adz",
            delta_terms_bs2[0],
            delta_terms_bs2[1],
            "SAPT2+3 ELST ENERGY atz",
            "SAPT2+3 EXCH ENERGY atz",
            "SAPT2+3 IND ENERGY atz",
            "SAPT2+3 DISP ENERGY atz",
        ]
    ].copy()
    df = df.apply(lambda x: x * qcc.hartree2kcalmol, axis=0)
    corr = df.corr()
    pd.set_option("display.max_columns", None)
    print(corr)
    # row 1
    axs[0, 0].scatter(
        df["SAPT2+3 ELST ENERGY adz"],
        df[delta_terms_bs1[0]],
        s=0.8,
    )
    axs[0, 1].scatter(
        df["SAPT2+3 EXCH ENERGY adz"],
        df[delta_terms_bs1[0]],
        s=0.8,
    )
    axs[0, 2].scatter(
        df["SAPT2+3 IND ENERGY adz"],
        df[delta_terms_bs1[0]],
        s=0.8,
    )
    axs[0, 3].scatter(
        df["SAPT2+3 DISP ENERGY adz"],
        df[delta_terms_bs1[0]],
        s=0.8,
    )
    axs[0, 0].set_ylabel(delta_terms_bs1[0])
    axs[0, 0].set_xlabel("SAPT2+3 ELST ENERGY adz")
    axs[0, 1].set_xlabel("SAPT2+3 EXCH ENERGY adz")
    axs[0, 2].set_xlabel("SAPT2+3 IND ENERGY adz")
    axs[0, 3].set_xlabel("SAPT2+3 DISP ENERGY adz")
    # row 2
    axs[1, 0].scatter(
        df["SAPT2+3 ELST ENERGY adz"],
        df[delta_terms_bs1[1]],
        s=0.8,
    )
    axs[1, 1].scatter(
        df["SAPT2+3 EXCH ENERGY adz"],
        df[delta_terms_bs1[1]],
        s=0.8,
    )
    axs[1, 2].scatter(
        df["SAPT2+3 IND ENERGY adz"],
        df[delta_terms_bs1[1]],
        s=0.8,
    )
    axs[1, 3].scatter(
        df["SAPT2+3 DISP ENERGY adz"],
        df[delta_terms_bs1[1]],
        s=0.8,
    )
    axs[1, 0].set_ylabel(delta_terms_bs1[1])
    axs[1, 0].set_xlabel("SAPT2+3 ELST ENERGY adz")
    axs[1, 1].set_xlabel("SAPT2+3 EXCH ENERGY adz")
    axs[1, 2].set_xlabel("SAPT2+3 IND ENERGY adz")
    axs[1, 3].set_xlabel("SAPT2+3 DISP ENERGY adz")
    # row 3
    axs[2, 0].scatter(
        df["SAPT2+3 ELST ENERGY atz"],
        df[delta_terms_bs2[0]],
        s=0.8,
    )
    axs[2, 1].scatter(
        df["SAPT2+3 EXCH ENERGY atz"],
        df[delta_terms_bs2[0]],
        s=0.8,
    )
    axs[2, 2].scatter(
        df["SAPT2+3 IND ENERGY atz"],
        df[delta_terms_bs2[0]],
        s=0.8,
    )
    axs[2, 3].scatter(
        df["SAPT2+3 DISP ENERGY atz"],
        df[delta_terms_bs2[0]],
        s=0.8,
    )
    axs[2, 0].set_ylabel(delta_terms_bs2[0])
    axs[2, 0].set_xlabel("SAPT2+3 ELST ENERGY atz")
    axs[2, 1].set_xlabel("SAPT2+3 EXCH ENERGY atz")
    axs[2, 2].set_xlabel("SAPT2+3 IND ENERGY atz")
    axs[2, 3].set_xlabel("SAPT2+3 DISP ENERGY atz")
    # row 4
    axs[3, 0].scatter(
        df["SAPT2+3 ELST ENERGY atz"],
        df[delta_terms_bs2[1]],
        s=0.8,
    )
    axs[3, 1].scatter(
        df["SAPT2+3 EXCH ENERGY atz"],
        df[delta_terms_bs2[1]],
        s=0.8,
    )
    axs[3, 2].scatter(
        df["SAPT2+3 IND ENERGY atz"],
        df[delta_terms_bs2[1]],
        s=0.8,
    )
    axs[3, 3].scatter(
        df["SAPT2+3 DISP ENERGY atz"],
        df[delta_terms_bs2[1]],
        s=0.8,
    )
    axs[3, 0].set_ylabel(delta_terms_bs2[1])
    axs[3, 0].set_xlabel("SAPT2+3 ELST ENERGY atz")
    axs[3, 1].set_xlabel("SAPT2+3 EXCH ENERGY atz")
    axs[3, 2].set_xlabel("SAPT2+3 IND ENERGY atz")
    axs[3, 3].set_xlabel("SAPT2+3 DISP ENERGY atz")
    for i in range(4):
        for j in range(4):
            axs[i, j].set_xlim(axs[i, j].get_xlim()[0], axs[i, j].get_xlim()[1])
            axs[i, j].set_ylim(axs[i, j].get_ylim()[0], axs[i, j].get_ylim()[1])
            x = np.linspace(axs[i, j].get_xlim()[0], axs[i, j].get_xlim()[1], 100)
            # get HF(N) index
            corr_y = i if i < 2 else i + 4
            # get SAPT component index
            corr_x = j + 2 if i < 2 else j + 8
            axs[i, j].plot(
                x,
                np.poly1d(np.polyfit(df.iloc[:, corr_x], df.iloc[:, corr_y], 1))(x),
                color="red",
                linestyle="--",
                label=f"Corr. {corr.iloc[corr_x, corr_y]:.2f}",
            )
            axs[i, j].legend()
    plt.tight_layout()
    plt.savefig(f"./delta_plots/{label.replace(' ', '_')}_correlation_plots.png")
    return


def main():
    df = pd.read_pickle("./plots/ddft_study.pkl")
    # delta_correlation_plots(
    #     df,
    #     label="delta MP2",
    #     delta_terms_bs1=["SAPT MP2(2) ENERGY adz", "SAPT MP2(3) ENERGY adz"],
    #     delta_terms_bs2=["SAPT MP2(2) ENERGY atz", "SAPT MP2(3) ENERGY atz"],
    # )
    # delta_correlation_plots(
    #     df,
    #     label="delta HF",
    #     delta_terms_bs1=["SAPT HF(2) ENERGY adz", "SAPT HF(3) ENERGY adz"],
    #     delta_terms_bs2=["SAPT HF(2) ENERGY atz", "SAPT HF(3) ENERGY atz"],
    # )
    # drop extra duplicate columns but keep one
    tmp1 = df['SAPT_DFT_pbe0_adz_dDFT'].tolist()
    df.drop(columns=['SAPT_DFT_pbe0_adz_dDFT'], inplace=True)
    df['SAPT_DFT_pbe0_adz_dDFT'] = tmp1

    # df['SAPT_DFT_pbe0_adz_dDFT_dHF'] = df.apply(lambda r: (r['SAPT_DFT_pbe0_adz_dDFT'] - 1.5*r['SAPT_DFT_pbe0_adz_dHF']) / qcc.hartree2kcalmol, axis=1)
    # df['SAPT_DFT_pbe0_atz_dDFT_dHF'] = df.apply(lambda r: (r['SAPT_DFT_pbe0_atz_dDFT'] - 1.5*r['SAPT_DFT_pbe0_atz_dHF']) / qcc.hartree2kcalmol, axis=1)
    df['SAPT_DFT_pbe0_adz_dDFT_dHF'] = df.apply(lambda r: (r['SAPT_DFT_pbe0_adz_dDFT'] - r['SAPT_DFT_pbe0_adz_dHF']) / qcc.hartree2kcalmol, axis=1)
    df['SAPT_DFT_pbe0_atz_dDFT_dHF'] = df.apply(lambda r: (r['SAPT_DFT_pbe0_atz_dDFT'] - r['SAPT_DFT_pbe0_atz_dHF']) / qcc.hartree2kcalmol, axis=1)
    df['SAPT_DFT_pbe0_adz_dDFT'] = df.apply(lambda r: (r['SAPT_DFT_pbe0_adz_dDFT']) / qcc.hartree2kcalmol, axis=1)
    df['SAPT_DFT_pbe0_atz_dDFT'] = df.apply(lambda r: (r['SAPT_DFT_pbe0_atz_dDFT']) / qcc.hartree2kcalmol, axis=1)
    print(df[['SAPT_DFT_pbe0_adz_dDFT', 'SAPT_DFT_pbe0_adz_dDFT_dHF']])
    delta_correlation_plots(
        df,
        label="delta DFT",
        delta_terms_bs1=["SAPT_DFT_pbe0_adz_dDFT", "SAPT_DFT_pbe0_adz_dDFT_dHF"],
        delta_terms_bs2=["SAPT_DFT_pbe0_atz_dDFT", "SAPT_DFT_pbe0_atz_dDFT_dHF"],
    )
    return


if __name__ == "__main__":
    main()
