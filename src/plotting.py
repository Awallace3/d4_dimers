import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import src
from qm_tools_aw import tools
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

kcal_per_mol = "$\mathrm{kcal\cdot mol^{-1}}$"

# colors = [
#         "blue",
#         "red",
#         "purple",
#         "brown",
#         "green",
#         "orange",
#         "pink",
#         "grey",
#         "yellow",
#         "teal",
#         "cyan",
#         "navy",
#         "magenta",
#         "lime",
#         "maroon",
#         "olive",
#         "indigo",
#         "gold",
#         "orchid",
#         "salmon",
#         "tan",
#         "turquoise",
#     ]


def compute_D3_D4_values_for_params_for_plotting(
    df: pd.DataFrame,
    label: str,
    compute_d3: bool = True,
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    """
    params_dict = src.paramsTable.paramsDict()
    params_d4 = params_dict["sadz"]
    params_d3 = params_dict["sdadz"][1:4]
    params_d4_ATM_G = params_dict["HF_ATM_OPT_START"]
    params_d4_ATM = params_dict["HF_ATM_OPT_OUT"]

    if compute_d3:
        print(f"Computing D3 values for {label}...")
        df[f"-D3 ({label})"] = df.apply(
            lambda r: src.jeff.compute_bj(params_d3, r["D3Data"]),
            axis=1,
        )
    print(f"Computing D4 2B values for {label}...")
    params_2B, params_ATM = src.paramsTable.generate_2B_ATM_param_subsets(params_d4)

    df[f"-D4 ({label})"] = df.apply(
        lambda row: src.locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B,
            params_ATM,
        ),
        axis=1,
    )
    print(f"Computing D4-ATM values for {label}...")
    params_2B_2, params_ATM_2 = src.paramsTable.generate_2B_ATM_param_subsets(
        params_d4_ATM_G
    )
    df[f"-D4 ({label}) ATM G"] = df.apply(
        lambda row: src.locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B_2,
            params_ATM_2,
        ),
        axis=1,
    )
    print(f"Computing D4 2B values (ATM PARAMS) for {label}...")
    df[f"-D4 2B@ATM_params ({label}) G"] = df.apply(
        lambda row: src.locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B_2,
            params_ATM,
        ),
        axis=1,
    )
    print(params_2B, params_2B_2, sep="\n")
    print(params_ATM, params_ATM_2, sep="\n")
    params_2B_2, params_ATM_2 = src.paramsTable.generate_2B_ATM_param_subsets(
        params_d4_ATM
    )
    df[f"-D4 ({label}) ATM"] = df.apply(
        lambda row: src.locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B_2,
            params_ATM_2,
        ),
        axis=1,
    )

    return df


def compute_d4_from_opt_params(
    df: pd.DataFrame,
    bases=[
        [
            "SAPT_DFT_adz_IE",
            "SAPT_DFT_adz_3_IE_ATM",
            "SAPT_DFT_OPT_ATM_END3",
            "SAPT_DFT_adz_3_IE",
        ],
        [
            "SAPT_DFT_adz_IE",
            "SAPT_DFT_adz_3_IE",
            "SAPT_DFT_OPT_END3",
            "SAPT_DFT_adz_3_IE",
        ],
        [
            "SAPT_DFT_adz_3_IE",
            "SAPT_DFT_adz_3_IE_no_disp",
            "SAPT_DFT_OPT_END3",
            "SAPT_DFT_adz_3_IE",
        ],
        # "DF_col_for_IE": "PARAMS_NAME"
        ["SAPT0_dz_IE", "SAPT0_dz_3_IE", "SAPT0_dz_3_IE_2B", "SAPT0_dz_3_IE"],
        ["SAPT0_jdz_IE", "SAPT0_jdz_3_IE", "SAPT0_jdz_3_IE_2B", "SAPT0_jdz_3_IE"],
        ["SAPT0_adz_IE", "SAPT0_adz_3_IE", "SAPT0_adz_3_IE_2B", "SAPT0_adz_3_IE"],
        [
            "SAPT0_adz_3_IE",
            "SAPT0_adz_3_IE_no_disp",
            "SAPT0_adz_3_IE_2B",
            "SAPT0_adz_3_IE",
        ],
        ["SAPT0_tz_IE", "SAPT0_tz_3_IE", "SAPT0_tz_3_IE_2B", "SAPT0_tz_3_IE"],
        ["SAPT0_mtz_IE", "SAPT0_mtz_3_IE", "SAPT0_mtz_3_IE_2B", "SAPT0_mtz_3_IE"],
        ["SAPT0_jtz_IE", "SAPT0_jtz_3_IE", "SAPT0_jtz_3_IE_2B", "SAPT0_jtz_3_IE"],
        ["SAPT0_atz_IE", "SAPT0_atz_3_IE", "SAPT0_atz_3_IE_2B", "SAPT0_atz_3_IE"],
    ],
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    """
    params_dict = src.paramsTable.paramsDict()
    plot_vals = {}
    for i in bases:
        params_d4 = params_dict[i[2]]
        params_2B, params_ATM = src.paramsTable.generate_2B_ATM_param_subsets(params_d4)
        df[f"-D4 ({i[1]})"] = df.apply(
            lambda row: src.locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
                row,
                params_2B,
                params_ATM,
            ),
            axis=1,
        )
        diff = f"{i[1]}_diff"
        d4_diff = f"{i[1]}_d4_diff"
        df[diff] = df["Benchmark"] - df[i[0]]
        df[d4_diff] = df["Benchmark"] - df[i[3]] - df[f"-D4 ({i[1]})"]
        print(f'"{diff}",')
        print(f'"{d4_diff}",')
    return df


def compute_d4_from_opt_params_TT(
    df: pd.DataFrame,
    bases=[
        # "DF_col_for_IE": "PARAMS_NAME"
        ["SAPT0_adz_IE", "SAPT0_adz_3_IE_TT", "HF_ATM_TT_OPT_START", "SAPT0_adz_3_IE"],
        ["SAPT0_adz_IE", "SAPT0_adz_3_IE_TT_OPT", "HF_ATM_OPT_OUT", "SAPT0_adz_3_IE"],
    ],
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    """
    params_dict = src.paramsTable.paramsDict()
    plot_vals = {}
    for i in bases:
        params_d4 = params_dict[i[2]]
        params_2B, params_ATM = src.paramsTable.generate_2B_ATM_param_subsets(params_d4)
        df[f"-D4(ATM TT) ({i[1]})"] = df.apply(
            lambda row: src.locald4.compute_disp_2B_BJ_ATM_TT_dimer(
                row,
                params_2B,
                params_ATM,
            ),
            axis=1,
        )
        d4_diff = f"{i[1]}_d4_diff"
        df[d4_diff] = df["Benchmark"] - df[i[3]] - df[f"-D4(ATM TT) ({i[1]})"]
        print(f'"{d4_diff}",')
    return df


def correlation_plot(df_subset, pfn="plots/correlation.png") -> None:
    """
    correlation_plot
    """
    # TODO: plot by dataset?
    cor = df_subset.corr()
    print(cor)
    sns.heatmap(
        cor,
        annot=True,
        cmap="vlag",
        fmt=".4g",
        annot_kws={"size": 8},
    )
    plt.title(f"Correlation Plot")
    plt.savefig(f"{pfn}", bbox_inches="tight")
    plt.clf()
    return


def plot_dbs(df, df_col, title_name, pfn, color="blue") -> None:
    """
    plot_dbs
    """
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    for d in dbs:
        df2 = df[df["DB"] == d]
        vData.append(df2[df_col].to_list())
        vLabels.append(d)

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)

    for n, pc in enumerate(vplot["bodies"], 1):
        if n % 2 != 0:
            pc.set_facecolor(color)
        else:
            pc.set_facecolor(color)
        pc.set_alpha(0.5)
        # pc.set_edgecolor("black")

    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        # label="+-1 kcal/mol",
        linewidth=0.8,
        label=r"$\pm$1 kcal/mol",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        alpha=0.5,
        linewidth=0.5,
        label="0 kcal/mol",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        linewidth=0.8,
        # label="+-1 kcal/mol",
        zorder=0,
    )
    ax.set_xticks(xs)
    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="5")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel("Error (kcal/mol)")
    plt.title(f"{title_name}")
    plt.savefig(f"plots/{pfn}_dbs_violin.png", bbox_inches="tight")
    plt.clf()
    return


def plot_dbs_d3_d4_two(df, c1, c2, l1, l2, title_name, pfn, first=True) -> None:
    """ """
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    if first:
        # dbs = [dbs[i] for i in range(len(dbs)//2)]
        dbs = dbs[: len(dbs) // 2]
    else:
        dbs = dbs[len(dbs) // 2 :]
    print(dbs)

    for d in dbs:
        df2 = df[df["DB"] == d]
        vData.append(df2[c1].to_list())
        vData.append(df2[c2].to_list())
        vLabels.append(f"{d} - {l1}")
        vLabels.append(f"{d} - {l2}")

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)

    for n, pc in enumerate(vplot["bodies"], 1):
        if n % 2 != 0:
            pc.set_facecolor("blue")
        else:
            pc.set_facecolor("red")
        pc.set_alpha(0.5)
        # pc.set_edgecolor("black")

    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        label=r"$\pm$1" + kcal_per_mol,
        zorder=0,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        # label=r"0 $kcal\cdot mol^{-1}$",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        # label="+-1 kcal/mol",
        zorder=0,
    )
    ax.set_xticks(xs)
    plt.setp(ax.set_xticklabels(vLabels), rotation=50, fontsize="7")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim((-12, 6))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    ax.grid(color="gray", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):
        if n % 2 != 0:
            xtick.set_color("blue")
        else:
            xtick.set_color("red")

    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=0.25)
    # plt.show()
    plt.savefig(f"plots/{pfn}_dbs_violin.png", bbox_inches="tight")
    plt.clf()
    return


def get_charged_df(df) -> pd.DataFrame:
    df = df.copy()
    def_charge = np.array([[0, 1] for i in range(3)])
    inds = []
    for i, row in df.iterrows():
        if np.all(row["charges"] == def_charge):
            inds.append(i)
    df = df.drop(inds)
    return df


def plot_basis_sets_d4(df, build_df=False, df_out: str = "basis"):
    df_out = f"plots/{df_out}.pkl"
    if build_df:
        df = compute_d4_from_opt_params(df)
        df = compute_d4_from_opt_params(
            df,
            bases=[
                # "DF_col_for_IE": "PARAMS_NAME"
                [
                    "SAPT0_dz_IE",
                    "SAPT0_dz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_dz_3_IE",
                ],
                [
                    "SAPT0_jdz_IE",
                    "SAPT0_jdz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_jdz_3_IE",
                ],
                [
                    "SAPT0_adz_IE",
                    "SAPT0_adz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_adz_3_IE",
                ],
                [
                    "SAPT0_tz_IE",
                    "SAPT0_tz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_tz_3_IE",
                ],
                [
                    "SAPT0_mtz_IE",
                    "SAPT0_mtz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_mtz_3_IE",
                ],
                [
                    "SAPT0_jtz_IE",
                    "SAPT0_jtz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_jtz_3_IE",
                ],
                [
                    "SAPT0_atz_IE",
                    "SAPT0_atz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_atz_3_IE",
                ],
            ],
        )
        df.to_pickle(df_out)
    else:
        df = pd.read_pickle(df_out)
    plot_violin_d3_d4_ALL(
        df,
        {
            "0/DZ": "SAPT0_dz_3_IE_diff",
            "0-D4/DZ": "SAPT0_dz_3_IE_d4_diff",
            "0/jDZ": "SAPT0_jdz_3_IE_diff",
            "0-D4/jDZ": "SAPT0_jdz_3_IE_d4_diff",
            "0/aDZ": "SAPT0_adz_3_IE_diff",
            "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
            "0/TZ": "SAPT0_tz_3_IE_diff",
            "0-D4/TZ": "SAPT0_tz_3_IE_d4_diff",
            "0/mTZ": "SAPT0_mtz_3_IE_diff",
            "0-D4/mTZ": "SAPT0_mtz_3_IE_d4_diff",
            "0/jTZ": "SAPT0_jtz_3_IE_diff",
            "0-D4/jTZ": "SAPT0_jtz_3_IE_d4_diff",
            "0/aTZ": "SAPT0_atz_3_IE_diff",
            "0-D4/aTZ": "SAPT0_atz_3_IE_d4_diff",
        },
        # f"{len(df)} Dimers With Different Basis Sets (D4)",
        # f"All Dimers ({len(df)})",
        f"Basis Set Comparison Across All Dimers ({len(df)})",
        f"basis_set_d4",
        bottom=0.30,

    )
    plot_violin_d3_d4_ALL(
        df,
        {
            "DZ (aDZ)": "SAPT0_dz_3_IE_ADZ_d4_diff",
            "DZ (OPT)": "SAPT0_dz_3_IE_d4_diff",
            "jDZ (aDZ)": "SAPT0_jdz_3_IE_ADZ_d4_diff",
            "jDZ (OPT)": "SAPT0_jdz_3_IE_d4_diff",
            "aDZ (aDZ)": "SAPT0_adz_3_IE_ADZ_d4_diff",
            "aDZ (OPT)": "SAPT0_adz_3_IE_d4_diff",
            "TZ (aDZ)": "SAPT0_tz_3_IE_ADZ_d4_diff",
            "TZ (OPT)": "SAPT0_tz_3_IE_d4_diff",
            "mTZ (aDZ)": "SAPT0_mtz_3_IE_ADZ_d4_diff",
            "mTZ (OPT)": "SAPT0_mtz_3_IE_d4_diff",
            "jTZ (aDZ)": "SAPT0_jtz_3_IE_ADZ_d4_diff",
            "jTZ (OPT)": "SAPT0_jtz_3_IE_d4_diff",
            "aTZ (aDZ)": "SAPT0_atz_3_IE_ADZ_d4_diff",
            "aTZ (OPT)": "SAPT0_atz_3_IE_d4_diff",
        },
        # f"{len(df)} Dimers With Different Basis Sets (D4)",
        f"All Dimers ({len(df)})",
        f"basis_set_d4_opt_vs_adz",
        ylim=[-15, 14],
        bottom=0.35,
    )
    return


def compute_d3_from_opt_params(
    df: pd.DataFrame,
    bases=[
        # "DF_col_for_IE": "PARAMS_NAME"
        [
            "SAPT0_dz_IE",
            "SAPT0_dz_3_IE",
            "SAPT0_dz_3_IE_2B_D3",
            "SAPT0_dz_3_IE",
        ],
        [
            "SAPT0_jdz_IE",
            "SAPT0_jdz_3_IE",
            "SAPT0_jdz_3_IE_2B_D3",
            "SAPT0_jdz_3_IE",
        ],
        [
            "SAPT0_adz_IE",
            "SAPT0_adz_3_IE",
            "SAPT0_adz_3_IE_2B_D3",
            "SAPT0_adz_3_IE",
        ],
        [
            "SAPT0_tz_IE",
            "SAPT0_tz_3_IE",
            "SAPT0_tz_3_IE_2B_D3",
            "SAPT0_tz_3_IE",
        ],
        [
            "SAPT0_mtz_IE",
            "SAPT0_mtz_3_IE",
            "SAPT0_mtz_3_IE_2B_D3",
            "SAPT0_mtz_3_IE",
        ],
        [
            "SAPT0_jtz_IE",
            "SAPT0_jtz_3_IE",
            "SAPT0_jtz_3_IE_2B_D3",
            "SAPT0_jtz_3_IE",
        ],
        [
            "SAPT0_atz_IE",
            "SAPT0_atz_3_IE",
            "SAPT0_atz_3_IE_2B_D3",
            "SAPT0_atz_3_IE",
        ],
    ],
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    """
    params_dict = src.paramsTable.paramsDict()
    plot_vals = {}
    for i in bases:
        params_d3 = params_dict[i[2]][0][1:4]
        print(params_d3)
        df[f"-D3 ({i[1]})"] = df.apply(
            lambda row: src.jeff.compute_BJ_CPP(
                params_d3,
                row["D3Data"],
            ),
            axis=1,
        )
        print(df[f"-D3 ({i[1]})"])
        diff = f"{i[1]}_diff"
        d3_diff = f"{i[1]}_d3_diff"
        df[diff] = df["Benchmark"] - df[i[0]]
        df[d3_diff] = df["Benchmark"] - df[i[3]] - df[f"-D3 ({i[1]})"]
        print(f'"{diff}",')
        print(f'"{d3_diff}",')
    return df


def plot_basis_sets_d3(df, build_df=False, df_out: str = "basis"):
    df_out = f"plots/{df_out}.pkl"
    if build_df:
        df = compute_d3_from_opt_params(df)
        df = compute_d3_from_opt_params(
            df,
            bases=[
                [
                    "SAPT0_dz_IE",
                    "SAPT0_dz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B_D3",
                    "SAPT0_dz_3_IE",
                ],
                [
                    "SAPT0_jdz_IE",
                    "SAPT0_jdz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B_D3",
                    "SAPT0_jdz_3_IE",
                ],
                [
                    "SAPT0_adz_IE",
                    "SAPT0_adz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B_D3",
                    "SAPT0_adz_3_IE",
                ],
                [
                    "SAPT0_tz_IE",
                    "SAPT0_tz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B_D3",
                    "SAPT0_tz_3_IE",
                ],
                [
                    "SAPT0_mtz_IE",
                    "SAPT0_mtz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B_D3",
                    "SAPT0_mtz_3_IE",
                ],
                [
                    "SAPT0_jtz_IE",
                    "SAPT0_jtz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B_D3",
                    "SAPT0_jtz_3_IE",
                ],
                [
                    "SAPT0_atz_IE",
                    "SAPT0_atz_3_IE_ADZ",
                    "SAPT0_adz_3_IE_2B_D3",
                    "SAPT0_atz_3_IE",
                ],
            ],
        )
        print(df.columns.values)
        df.to_pickle(df_out)
    else:
        df = pd.read_pickle(df_out)
    # TODO: simplify plot labels to be like -D/adz, and rotate labels back
    plot_violin_d3_d4_ALL(
        df,
        {
            "0/DZ": "SAPT0_dz_3_IE_diff",
            "0-D3/DZ": "SAPT0_dz_3_IE_d3_diff",
            "0/jDZ": "SAPT0_jdz_3_IE_diff",
            "0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
            "0/aDZ": "SAPT0_adz_3_IE_diff",
            "0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
            "0/TZ": "SAPT0_tz_3_IE_diff",
            "0-D3/TZ": "SAPT0_tz_3_IE_d3_diff",
            "0/mTZ": "SAPT0_mtz_3_IE_diff",
            "0-D3/mTZ": "SAPT0_mtz_3_IE_d3_diff",
            "0/jTZ": "SAPT0_jtz_3_IE_diff",
            "0-D3/jTZ": "SAPT0_jtz_3_IE_d3_diff",
            "0/aTZ": "SAPT0_atz_3_IE_diff",
            "0-D3/aTZ": "SAPT0_atz_3_IE_d3_diff",
        },
        f"{len(df)} Dimers With Different Basis Sets (D3)",
        f"basis_set_d3",
        bottom=0.30,
    )
    plot_violin_d3_d4_ALL(
        df,
        {
            "DZ (ADZ)": "SAPT0_dz_3_IE_ADZ_d3_diff",
            "DZ (OPT)": "SAPT0_dz_3_IE_d3_diff",
            "jDZ (ADZ)": "SAPT0_jdz_3_IE_ADZ_d3_diff",
            "jDZ (OPT)": "SAPT0_jdz_3_IE_d3_diff",
            "aDZ (ADZ)": "SAPT0_adz_3_IE_ADZ_d3_diff",
            "aDZ (OPT)": "SAPT0_adz_3_IE_d3_diff",
            "TZ (ADZ)": "SAPT0_tz_3_IE_ADZ_d3_diff",
            "TZ (OPT)": "SAPT0_tz_3_IE_d3_diff",
            "mTZ (ADZ)": "SAPT0_mtz_3_IE_ADZ_d3_diff",
            "mTZ (OPT)": "SAPT0_mtz_3_IE_d3_diff",
            "jTZ (ADZ)": "SAPT0_jtz_3_IE_ADZ_d3_diff",
            "jTZ (OPT)": "SAPT0_jtz_3_IE_d3_diff",
            "aTZ (ADZ)": "SAPT0_atz_3_IE_ADZ_d3_diff",
            "aTZ (OPT)": "SAPT0_atz_3_IE_d3_diff",
        },
        f"{len(df)} Dimers With Different Basis Sets (D3)",
        f"basis_set_d3_opt_vs_adz",
        bottom=0.35,
        ylim=[-15, 15],
    )

    return


def plotting_setup(df, build_df=False, df_out: str = "plots/plot.pkl", compute_d3=True):
    df, selected = df
    selected = selected.split("/")[-1].split(".")[0]
    df_out = f"plots/{selected}.pkl"
    if build_df:
        # print(df.columns.values)
        # for i in [
        #     j
        #     for j in df.columns.values
        #     if "SAPT0_" in j
        #     if j not in ["SAPT0", "SAPT0_jdz", "SAPT0_aqz"]
        #     if "_IE" not in j
        # ]:
        #     df[i + "_IE"] = df.apply(lambda r: r[i][0], axis=1)
        #     df[i + "_diff"] = df["Benchmark"] - df[i + "_IE"]
        df = compute_d4_from_opt_params(
            df,
            bases=[["SAPT0_dz_IE", "SAPT0_dz_3_IE_ATM_SHARED", "HF_ATM_SHARED", "SAPT0_dz_3_IE"]],
        )

        df = compute_D3_D4_values_for_params_for_plotting(df, "adz", compute_d3)
        df = compute_D3_D4_values_for_params_for_plotting(df, "jdz", compute_d3)
        df = compute_d4_from_opt_params(df)
        df = compute_d4_from_opt_params_TT(df)


        df["SAPT0-D4/aug-cc-pVDZ"] = df.apply(
            lambda row: row["SAPT0_adz_3_IE"] + row["-D4 (adz)"],
            axis=1,
        )
        df["SAPT0-D4(ATM)/aug-cc-pVDZ"] = df.apply(
            lambda row: row["SAPT0_adz_3_IE"] + row["-D4 (adz) ATM"],
            axis=1,
        )
        df["SAPT0-D4/jun-cc-pVDZ"] = df.apply(
            lambda row: row["SAPT0_jdz_3_IE"] + row["-D4 (jdz)"],
            axis=1,
        )
        df["SAPT0-D4(ATM)/jun-cc-pVDZ"] = df.apply(
            lambda row: row["SAPT0_jdz_3_IE"] + row["-D4 (jdz) ATM"],
            axis=1,
        )
        df["adz_diff_d4"] = df["Benchmark"] - df["SAPT0-D4/aug-cc-pVDZ"]
        df["adz_diff_d4_ATM"] = df["Benchmark"] - df["SAPT0-D4(ATM)/aug-cc-pVDZ"]
        df["adz_diff_d4_ATM_G"] = df["Benchmark"] - (
            df["SAPT0_adz_3_IE"] + df["-D4 (adz) ATM G"]
        )
        df["jdz_diff_d4_ATM_G"] = df["Benchmark"] - (
            df["SAPT0_jdz_3_IE"] + df["-D4 (jdz) ATM G"]
        )
        df["jdz_diff_d4"] = df["Benchmark"] - df["SAPT0-D4/jun-cc-pVDZ"]
        df["jdz_diff_d4_ATM"] = df["Benchmark"] - df["SAPT0-D4(ATM)/jun-cc-pVDZ"]
        df["SAPT0_jdz_diff"] = df["Benchmark"] - df["SAPT0"]
        df["SAPT0_jdz_diff"] = df["Benchmark"] - df["SAPT0"]

        if compute_d3:
            df["adz_diff_d4_2B@ATM_G"] = df["Benchmark"] - (
                df["SAPT0_adz_3_IE"] + df["-D4 2B@ATM_params (adz) G"]
            )
            df["jdz_diff_d4_2B@ATM_G"] = df["Benchmark"] - (
                df["SAPT0_jdz_3_IE"] + df["-D4 2B@ATM_params (adz) G"]
            )
            df["SAPT0-D3/jun-cc-pVDZ"] = df.apply(
                lambda row: row["SAPT0_jdz_3_IE"] + row["-D3 (jdz)"],
                axis=1,
            )
            df["SAPT0-D3/aug-cc-pVDZ"] = df.apply(
                lambda row: row["SAPT0_adz_3_IE"] + row["-D3 (adz)"],
                axis=1,
            )
            df["SAPT0-D3/aug-cc-pVDZ"] = df.apply(
                lambda row: row["SAPT0_adz_3_IE"] + row["-D3 (adz)"],
                axis=1,
            )
        df["adz_diff_d3"] = df["Benchmark"] - df["SAPT0-D3/aug-cc-pVDZ"]
        df["jdz_diff_d3"] = df["Benchmark"] - df["SAPT0-D3/jun-cc-pVDZ"]

        # D3 binary results
        df["jdz_diff_d3mbj"] = df["Benchmark"] - (df["SAPT0_jdz_3_IE"] + df["D3MBJ"])
        df["adz_diff_d3mbj"] = df["Benchmark"] - (df["SAPT0_adz_3_IE"] + df["D3MBJ"])
        df["jdz_diff_d3mbj_atm"] = df["Benchmark"] - (
            df["SAPT0_jdz_3_IE"] + df["D3MBJ ATM"]
        )
        df["adz_diff_d3mbj_atm"] = df["Benchmark"] - (
            df["SAPT0_adz_3_IE"] + df["D3MBJ ATM"]
        )
        df.to_pickle(df_out)

    else:
        df = pd.read_pickle(df_out)
    # Non charged
    plot_violin_d3_d4_ALL(
        df,
        {
            "0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
            "0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
            "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
            "0-D4(ATM)/aDZ": "SAPT0_dz_3_IE_ATM_SHARED_d4_diff",
            "0-D4(2B ATM)/aDZ": "adz_diff_d4_ATM",
            "0-D4(2B@G ATM)/aDZ": "adz_diff_d4_2B@ATM_G",
            "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
            # "0-D4(ATM TT)/aDZ": "SAPT0_adz_3_IE_TT_OPT_d4_diff",
            "0/jDZ": "SAPT0_jdz_3_IE_diff",
            "0/aDZ": "SAPT0_adz_3_IE_diff",
            "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
            "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
            "SAPT(DFT)-D4/aTZ": "SAPT_DFT_adz_3_IE_d4_diff",
            "SAPT(DFT)/aTZ": "SAPT_DFT_adz_3_IE_diff",
        },
        f"",
        f"{selected}_ATM",
        bottom=0.45,
        ylim=[-18, 22],
        # figure_size=(6, 6),
        dpi=1200,
        pdf=False,
    )
    if True:
        # plot_violin_d3_d4_ALL(
        #     df,
        #     {
        #         "0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
        #         # "0-D3MBJ(ATM)/jDZ": "jdz_diff_d3mbj_atm",
        #         "0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
        #         # "0-D3MBJ(ATM)/aDZ": "adz_diff_d3mbj_atm",
        #         "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
        #         "0-D4(ATM)/aDZ": "SAPT0_dz_3_IE_ATM_SHARED_d4_diff",
        #         "0-D4(2B ATM)/aDZ": "adz_diff_d4_ATM",
        #         "0-D4(2B@G ATM)/aDZ": "adz_diff_d4_2B@ATM_G",
        #         "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
        #         "0-D4(ATM TT)/aDZ": "SAPT0_adz_3_IE_TT_OPT_d4_diff",
        #         "0/jDZ": "SAPT0_jdz_3_IE_diff",
        #         "0/aDZ": "SAPT0_adz_3_IE_diff",
        #         "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
        #         "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
        #     },
        #     f"All Dimers (8299)",
        #     f"{selected}_ATM",
        #     bottom=0.45,
        #     ylim=[-18, 22],
        #     figure_size=(6, 6),
        # )
        plot_dbs_d3_d4(
            df,
            "adz_diff_d4",
            "adz_diff_d4_ATM_G",
            "-D4",
            "-D4(ATM)",
            bottom=0.35,
            # title_name=f"DB Breakdown SAPT0-D4/aug-cc-pVDZ ({selected})",
            title_name=f"-D4 Two-Body versus Three-Body (ATM)",
            pfn=f"db_breakdown_2B_ATM",
        )
        # Basis Set Performance: SAPT0
        plot_violin_d3_d4_ALL(
            df,
            {
                "0-D4/DZ": "SAPT0_dz_3_IE_d4_diff",
                "0-D4/jDZ": "SAPT0_jdz_3_IE_d4_diff",
                "0/jDZ": "SAPT0_jdz_3_IE_diff",
                "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
                "0/aDZ": "SAPT0_adz_3_IE_diff",
                "0/TZ": "SAPT0_tz_3_IE_diff",
                "0/mTZ": "SAPT0_mtz_3_IE_diff",
                "0/jTZ": "SAPT0_jtz_3_IE_diff",
                "0/aTZ": "SAPT0_atz_3_IE_diff",
            },
            # f"All Dimers with SAPT0 ({selected})",
            "All Dimers (8299)",
            f"{selected}_basis_set",
        )
        # charged
        df_charged = get_charged_df(df)
        plot_violin_d3_d4_ALL(
            df_charged,
            {
                "0-D3/jDZ": "jdz_diff_d3",
                "0-D3/aDZ": "adz_diff_d3",
                "0-D3MBJ(ATM)/jDZ": "jdz_diff_d3mbj_atm",
                "0-D3MBJ(ATM)/aDZ": "adz_diff_d3mbj_atm",
                "0-D4/jVDZ": "jdz_diff_d4",
                "0-D4/aDZ": "adz_diff_d4",
                # "0-D4(2B@G ATM)/jDZ": "jdz_diff_d4_2B@ATM_G",
                # "0-D4(2B@G ATM)/aDZ": "adz_diff_d4_2B@ATM_G",
                "0-D4(2B@G ATM@G)/jDZ": "jdz_diff_d4_ATM_G",
                "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
                "0/jDZ": "SAPT0_jdz_3_IE_diff",
                "0/aDZ": "SAPT0_adz_3_IE_diff",
                "0-D4(ATM TT)/aDZ": "SAPT0_adz_3_IE_TT_OPT_d4_diff",
            },
            f"Charged Dimers ({len(df_charged)})",
            f"{selected}_charged",
            bottom=0.42,
            ylim=[-10, 10],
        )
    return


def select_element(vals, ind):
    if vals is not None:
        return vals[ind]
    else:
        return np.nan


def plotting_setup_dft(
    df, build_df=False, df_out: str = "plots/plot.pkl", compute_d3=True
):
    df, selected = df
    selected = selected.split("/")[-1].split(".")[0]
    df_out = f"plots/{selected}.pkl"
    if build_df:
        # df = compute_d4_from_opt_params(df)
        # Need to get components
        df["SAPT0_adz_elst"] = df.apply(lambda x: x["SAPT0_adz"][1], axis=1)
        df["SAPT0_adz_exch"] = df.apply(lambda x: x["SAPT0_adz"][2], axis=1)
        df["SAPT0_adz_indu"] = df.apply(lambda x: x["SAPT0_adz"][3], axis=1)
        df["SAPT0_adz_disp"] = df.apply(lambda x: x["SAPT0_adz"][4], axis=1)
        df["SAPT_DFT_adz_elst"] = df.apply(
            lambda x: select_element(x["SAPT_DFT_adz"], 1), axis=1
        )
        df["SAPT_DFT_adz_exch"] = df.apply(
            lambda x: select_element(x["SAPT_DFT_adz"], 2), axis=1
        )
        df["SAPT_DFT_adz_indu"] = df.apply(
            lambda x: select_element(x["SAPT_DFT_adz"], 3), axis=1
        )
        df["SAPT_DFT_adz_disp"] = df.apply(
            lambda x: select_element(x["SAPT_DFT_adz"], 4), axis=1
        )
        df.to_pickle(df_out)
    else:
        df = pd.read_pickle(df_out)
    plot_violin_SAPT0_DFT_components(df, pfn=f"{selected}_saptdft_sapt0_components")
    plot_violin_d3_d4_ALL(
        df,
        {
            "0/aDZ": "SAPT0_adz_3_IE_diff",
            "0/aDZ no disp": "SAPT0_adz_3_IE_no_disp_diff",
            "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
            "DFT/aDZ": "SAPT_DFT_adz_3_IE_diff",
            "DFT/aDZ no disp": "SAPT_DFT_adz_3_IE_no_disp_diff",
            "DFT-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
            "DFT-D4(ATM)/aDZ": "SAPT_DFT_adz_3_IE_ATM_d4_diff",
        },
        f"All Dimers with SAPT0 and SAPT-DFT",
        f"{selected}_saptdft_sapt0",
        bottom=0.45,
        ylim=[-16, 30],
        transparent=False,
    )
    return


def plotting_setup_G(
    df,
    build_df=False,
    df_out: str = "plots/plot.pkl",
    compute_d3=False,
):
    df, selected = df
    selected = selected.split("/")[-1].split(".")[0]
    df_out = f"plots/{selected}.pkl"
    if build_df:
        df = compute_D3_D4_values_for_params_for_plotting(df, "qz", compute_d3)

        df["SAPT0-D4/aug-cc-pVDZ"] = df.apply(
            lambda row: row["HF_qz"] + row["-D4 (qz)"],
            axis=1,
        )
        df["SAPT0-D4(ATM)/aug-cc-pVDZ"] = df.apply(
            lambda row: row["HF_qz"] + row["-D4 (qz) ATM"],
            axis=1,
        )
        df["qz_diff_d4"] = df["Benchmark"] - df["SAPT0-D4/aug-cc-pVDZ"]
        df["qz_diff_d4_ATM"] = df["Benchmark"] - df["SAPT0-D4(ATM)/aug-cc-pVDZ"]
        df["qz_diff_d4_ATM_G"] = df["Benchmark"] - (df["HF_qz"] + df["-D4 (qz) ATM G"])
        df["qz_diff_d4_2B@ATM_G"] = df["Benchmark"] - (
            df["HF_qz"] + df["-D4 2B@ATM_params (qz) G"]
        )
        # D3 binary results
        df["qz_diff_d3mbj"] = df["Benchmark"] - (df["HF_qz"] + df["D3MBJ"])
        df["qz_diff_d3mbj_atm"] = df["Benchmark"] - (df["HF_qz"] + df["D3MBJ ATM"])
        df.to_pickle(df_out)
    else:
        df = pd.read_pickle(df_out)
    # Non charged
    plot_dbs_d3_d4(
        df,
        "qz_diff_d4",
        "qz_diff_d4_ATM_G",
        "-D4 (2B)",
        "-D4 (ATM_G)",
        # title_name=f"DB Breakdown SAPT0-D4/aug-cc-pVDZ ({selected})",
        title_name=f"-D4 Two-Body version Three-Body",
        pfn=f"{selected}_db_breakdown_2B_ATM",
    )
    df_charged = get_charged_df(df)
    # TODO: remove (2B) and take of _G for ATM
    plot_violin_d3_d4_ALL(
        df,
        {
            "-D3(ATM)/aug-cc-pVDZ": "qz_diff_d3mbj_atm",
            "-D4/aug-cc-pVDZ": "qz_diff_d4",
            "-D4(ATM)/aug-cc-pVDZ": "qz_diff_d4_ATM",
            "-D4(2B@ATM_params_G)/aug-cc-pVDZ": "qz_diff_d4_2B@ATM_G",
            "-D4(ATM_G)/aug-cc-pVDZ": "qz_diff_d4_ATM_G",
        },
        f"All Dimers with SAPT0 ({selected})",
        f"{selected}_qz_d3_d4_total_sapt0",
    )
    return


def plot_violin_d3_d4_ALL(
    df,
    vals: {},
    title_name: str,
    pfn: str,
    bottom: float = 0.4,
    ylim=[-15, 35],
    transparent=True,
    widths=0.85,
    figure_size=None,
    set_xlable=False,
    dpi=1200,
    pdf=False,
) -> None:
    """ """
    print(f"Plotting {pfn}")
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []

    annotations = []  # [(x, y, text), ...]
    cnt = 1
    plt.rcParams["text.usetex"] = True
    for k, v in vals.items():
        df[v] = pd.to_numeric(df[v])
        df_sub = df[df[v].notna()].copy()
        vData.append(df_sub[v].to_list())
        k_label = "\\textbf{" + k + "}"
        vLabels.append(k_label)
        m = df_sub[v].max()
        rmse = df_sub[v].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[v].apply(lambda x: abs(x)).mean()
        max_error = df_sub[v].apply(lambda x: abs(x)).max()
        text = r"$\mathit{%.2f}$" % mae
        text += "\n"
        text += r"$\mathbf{%.2f}$" % rmse
        text += "\n"
        text += r"$\mathrm{%.2f}$" % max_error
        annotations.append((cnt, m, text))
        cnt += 1

    pd.set_option("display.max_columns", None)
    # print(df[vals.values()].describe(include="all"))
    # transparent figure
    fig = plt.figure(dpi=dpi)
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    ax = plt.subplot(111)
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        quantiles=[[0.05, 0.95] for i in range(len(vData))],
        widths=widths,
    )
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)
    quantile_color = "red"
    quantile_style = "-"
    quantile_linewidth = 0.8
    for n, partname in enumerate(["cquantiles"]):
        vp = vplot[partname]
        vp.set_edgecolor(quantile_color)
        vp.set_linewidth(quantile_linewidth)
        vp.set_linestyle(quantile_style)
        vp.set_alpha(1)

    colors = ["blue" if i % 2 == 0 else "green" for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        pc.set_alpha(0.6)

    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        label=r"$\pm$1 $\mathrm{kcal\cdot mol^{-1}}$",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        # label=r"Reference Energy",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        [],
        [],
        linestyle=quantile_style,
        color=quantile_color,
        linewidth=quantile_linewidth,
        label=r"5-95th Percentile",
    )
    # TODO: fix minor ticks to be between
    navy_blue = (0.0, 0.32, 0.96)
    ax.set_xticks(xs)
    # minor_yticks = np.arange(ylim[0], ylim[1], 2)
    # ax.set_yticks(minor_yticks, minor=True)

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="8")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc="upper left", edgecolor="black", fontsize="8")
    # lg.get_frame().set_alpha(None)
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k")
    ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k")
    # ax.grid(color="gray", which="major", linewidth=0.5, alpha=0.3)
    # ax.grid(color="gray", which="minor", linewidth=0.5, alpha=0.3)

    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)
    # Annotations of RMSE
    for x, y, text in annotations:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y + 0.1),
            color="black",
            fontsize="8",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    if title_name is not None:
        plt.title(f"{title_name}")
    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)

    if pdf:
        fn_pdf = f"plots/{pfn}_dbs_violin.pdf"
        fn_png = f"plots/{pfn}_dbs_violin.png"
        plt.savefig(
            fn_pdf, transparent=transparent, bbox_inches="tight", dpi=dpi,
        )
        if os.path.exists(fn_png):
            os.system(f"rm {fn_png}")
        os.system(f"pdftoppm -png -r 400 {fn_pdf} {fn_png}")
        if os.path.exists(f"{fn_png}-1.png"):
            os.system(f"mv {fn_png}-1.png {fn_png}")
        else:
            print(f"Error: {fn_png}-1.png does not exist")
    else:
        plt.savefig(
            f"plots/{pfn}_dbs_violin.png", transparent=transparent, bbox_inches="tight", dpi=dpi,
        )
    plt.clf()
    return


def collect_component_data(df, vals):
    vLabels, vData = [], []
    annotations = []  # [(x, y, text), ...]
    cnt = 1
    plt.rcParams["text.usetex"] = True
    print(vals)
    for k, v in vals["vals"].items():
        print(k, v)
        df[v] = pd.to_numeric(df[v])
        df[f"{v}_diff"] = df[vals["reference"][1]] - df[v]
        df_sub = df[df[f"{v}_diff"].notna()].copy()
        print(df_sub[[vals["reference"][1], v, f"{v}_diff"]].describe())
        vData.append(df_sub[f"{v}_diff"].to_list())
        vLabels.append(k)
        m = df_sub[f"{v}_diff"].max()
        rmse = df_sub[f"{v}_diff"].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[f"{v}_diff"].apply(lambda x: abs(x)).mean()
        max_error = df_sub[f"{v}_diff"].apply(lambda x: abs(x)).max()
        text = r"$\mathit{%.2f}$" % mae
        text += "\n"
        text += r"$\mathbf{%.2f}$" % rmse
        text += "\n"
        text += r"$\mathrm{%.2f}$" % max_error
        annotations.append((cnt, m, text))
        cnt += 1
    return vData, vLabels, annotations


def create_minor_y_ticks(ylim):
    diff = abs(ylim[1] - ylim[0])
    if diff > 100:
        inc = 10
    if diff > 20:
        inc = 5
    elif diff > 10:
        inc = 2.5
    else:
        inc = 1
    lower_bound = int(ylim[0])
    while lower_bound % inc != 0:
        lower_bound -= 1
    upper_bound = int(ylim[1])
    while upper_bound % inc != 0:
        upper_bound += 1
    upper_bound += inc
    minor_yticks = np.arange(lower_bound, upper_bound, inc)
    return minor_yticks


def plot_component_violin(
    ax, vData, vLabels, annotations, title_name, ylabel, widths=0.85
):
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        quantiles=[[0.05, 0.95] for i in range(len(vData))],
        widths=widths,
    )
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)
    quantile_color = "red"
    quantile_style = "-"
    quantile_linewidth = 0.8
    for n, partname in enumerate(["cquantiles"]):
        vp = vplot[partname]
        vp.set_edgecolor(quantile_color)
        vp.set_linewidth(quantile_linewidth)
        vp.set_linestyle(quantile_style)
        vp.set_alpha(1)

    colors = ["blue" if i % 2 == 0 else "green" for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        pc.set_alpha(0.6)

    # plt automatically make extra ylimits for annotations above violin plot error bar
    # so we need to add extra space to the ylim
    ylim = ax.get_ylim()
    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_ylim((ylim[0], int(ylim[1] + abs(ylim[1] - ylim[0]) * 0.2)))
    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        label=r"$\pm$1 $\mathrm{kcal\cdot mol^{-1}}$",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        # label=r"Reference Energy",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        [],
        [],
        linestyle=quantile_style,
        color=quantile_color,
        linewidth=quantile_linewidth,
        label=r"5-95th Percentile",
    )
    # TODO: fix minor ticks to be between
    navy_blue = (0.0, 0.32, 0.96)
    ax.set_xticks(xs)
    # plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="8")
    plt.setp(ax.set_xticklabels(vLabels), rotation=45, fontsize="5")
    ax.set_xlim((0, len(vLabels)))
    # lg = ax.legend(loc="upper left", edgecolor="black", fontsize="8")
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    ylabel = f"{ylabel} Error" + r" ($\mathrm{kcal\cdot mol^{-1}}$)"
    ax.set_ylabel(ylabel, color="k")
    # set minor ticks to be between major ticks

    ax.grid(color="grey", which="major", linewidth=0.5, alpha=0.3)
    ax.grid(color="grey", which="minor", linewidth=0.5, alpha=0.3)
    # Set subplot title
    ax.set_ylabel(ylabel, color="k", fontsize="8")
    title_color = "k"
    if title_name == "Electrostatics":
        title_color = "red"
    elif title_name == "Exchange":
        title_color = "blue"
    elif title_name == "Induction":
        title_color = "green"
    elif title_name == "Dispersion":
        title_color = "orange"
    ax.set_title(title_name, color=title_color, fontsize="10")

    # Annotations of RMSE
    for x, y, text in annotations:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y + 0.1),
            color="black",
            fontsize="8",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)
    return ax


def plot_violin_SAPT0_DFT_components(
    df,
    elst_vals={
        "name": "Electrostatics",
        "reference": ["SAPT0 Ref.", "SAPT0_adz_elst"],
        "vals": {
            "SAPT(DFT)": "SAPT_DFT_adz_elst",
        },
    },
    exch_vals={
        "name": "Exchange",
        "reference": ["SAPT0 Ref.", "SAPT0_adz_exch"],
        "vals": {
            "SAPT(DFT)": "SAPT_DFT_adz_exch",
        },
    },
    indu_vals={
        "name": "Induction",
        "reference": ["SAPT0 Ref.", "SAPT0_adz_indu"],
        "vals": {
            "SAPT(DFT)": "SAPT_DFT_adz_indu",
        },
    },
    disp_vals={
        "name": "Dispersion",
        "reference": ["SAPT0 Ref.", "SAPT0_adz_disp"],
        "vals": {
            "SAPT(DFT)": "SAPT_DFT_adz_disp",
            "-D4/aDZ (SAPT0_2B)": "-D4 (SAPT0_adz_3_IE)",
            "-D4/aDZ (SAPT_DFT_2B)": "-D4 (SAPT_DFT_adz_3_IE)",
            "-D4/aDZ (SAPT_DFT_ATM)": "-D4 (SAPT_DFT_adz_3_IE_ATM)",
        },
    },
    pfn: str = "sapt0_dft_components",
    bottom: float = 0.4,
    transparent=False,
    widths=0.95,
) -> None:
    """ """
    print(f"Plotting {pfn}")
    # create subplots

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=1000)
    elst_ax = axs[0, 0]
    exch_ax = axs[0, 1]
    indu_ax = axs[1, 0]
    disp_ax = axs[1, 1]
    # add extra space for subplot titles
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Component Data
    elst_data, elst_labels, elst_annotations = collect_component_data(df, elst_vals)
    exch_data, exch_labels, exch_annotations = collect_component_data(df, exch_vals)
    indu_data, indu_labels, indu_annotations = collect_component_data(df, indu_vals)
    disp_data, disp_labels, disp_annotations = collect_component_data(df, disp_vals)

    # Plot violins
    plot_component_violin(
        elst_ax,
        elst_data,
        elst_labels,
        elst_annotations,
        elst_vals["name"],
        elst_vals["reference"][0],
        widths,
    )
    plot_component_violin(
        exch_ax,
        exch_data,
        exch_labels,
        exch_annotations,
        exch_vals["name"],
        exch_vals["reference"][0],
        widths,
    )
    plot_component_violin(
        indu_ax,
        indu_data,
        indu_labels,
        indu_annotations,
        indu_vals["name"],
        indu_vals["reference"][0],
        widths,
    )
    plot_component_violin(
        disp_ax,
        disp_data,
        disp_labels,
        disp_annotations,
        disp_vals["name"],
        disp_vals["reference"][0],
        widths,
    )

    # plt add space at bottom of figure

    plt.savefig(f"plots/{pfn}.png", transparent=transparent, bbox_inches="tight")
    plt.clf()
    return


def plot_dbs_d3_d4(
    df,
    c1,
    c2,
    l1,
    l2,
    title_name,
    pfn,
    outlier_cutoff=3,
    bottom=0.3,
    transparent=True,
    dpi=1200,
    pdf=False,
) -> None:
    print(f"Plotting {pfn}")
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData, vDataErrors = [], [], []
    annotations = []  # [(x, y, text), ...]
    for d in dbs:
        df2 = df[df["DB"] == d]
        vData.append(df2[c1].to_list())
        vData.append(df2[c2].to_list())
        df3 = df2[abs(df2[c1]) > outlier_cutoff]
        if len(df3) > 0:
            vDataErrors.append(df3[c1].to_list())
        else:
            vDataErrors.append([])
        df3 = df2[abs(df2[c2]) > outlier_cutoff]
        if len(df3) > 0:
            vDataErrors.append(df3[c2].to_list())
            for n, r in df3.iterrows():
                print(
                    f"\n{r['DB']}, {r['System']}\nid: {r['id']}, error: {c2}={r[c2]:.2f}, {c1}={r[c1]:.2f}"
                )
                tools.print_cartesians(r["Geometry"])

        else:
            vDataErrors.append([])
        d_str = d.replace(" ", "")
        # vLabels.append(f"{d_str} -{l1}")
        # vLabels.append(f"{d_str} -{l2}")
        vLabels.append(f"\\textbf{{{d_str}-{l1}}}")
        vLabels.append(f"\\textbf{{{d_str}-{l2}}}")

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)

    for n, pc in enumerate(vplot["bodies"], 1):
        if n % 2 != 0:
            pc.set_facecolor("blue")
        else:
            pc.set_facecolor("red")
        pc.set_alpha(0.5)
        # pc.set_edgecolor("black")

    xs, ys = [], []
    for n, y in enumerate(vDataErrors):
        if len(y) > 0:
            xs.extend([n + 1 for j in range(len(y))])
            ys.extend(y)
    print(xs, ys)
    ax.scatter(xs, ys, color="orange", s=8.0, label="Outliers")

    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        label=r"$\pm$1 $\mathrm{kcal\cdot mol^{-1}}$",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        # label=r"0 $kcal\cdot mol^{-1}$",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        # label="+-1 kcal/mol",
        zorder=0,
    )
    ax.set_xticks(xs)

    # Minor ticks
    # ax.yaxis.set_major_locator(MultipleLocator(20))
    # ax.yaxis.set_major_formatter('{y:.0f}')
    # For the minor ticks, use no labels; default NullFormatter.
    # ax.yaxis.set_minor_locator(MultipleLocator(2))
    # ax.tick_params(which='minor', length=2, color='black', labelsize=5)

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="5")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="lower left")
    ax.set_xlabel("Database")
    # ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
    ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    ax.grid(color="gray", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):
        if n % 2 != 0:
            xtick.set_color("blue")
        else:
            xtick.set_color("red")

    plt.minorticks_on()
    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)
    if pdf:
        fn_pdf = f"plots/{pfn}_dbs_violin.pdf"
        fn_png = f"plots/{pfn}_dbs_violin.png"
        plt.savefig(
            fn_pdf, transparent=transparent, bbox_inches="tight", dpi=dpi,
        )
        if os.path.exists(fn_png):
            os.system(f"rm {fn_png}")
        os.system(f"pdftoppm -png -r 400 {fn_pdf} {fn_png}")
        if os.path.exists(f"{fn_png}-1.png"):
            os.system(f"mv {fn_png}-1.png {fn_png}")
        else:
            print(f"Error: {fn_png}-1.png does not exist")
    else:
        plt.savefig(
            f"plots/{pfn}_dbs_violin.png", transparent=transparent, bbox_inches="tight", dpi=dpi,
        )
    plt.clf()
    return
