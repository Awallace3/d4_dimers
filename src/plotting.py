import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
from qm_tools_aw import tools
import warnings
from . import paramsTable
from . import locald4
from . import jeff
import qcelemental as qcel

h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# plt.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
        # "text.usetex": True,
        # "font.family": "Arial",
        # "font.sans-serif": "Arial",
        # "mathtext.fontset": "custom",
    }
)


def compute_D3_D4_values_for_params_for_plotting(
    df: pd.DataFrame,
    label: str,
    compute_d3: bool = True,
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    """
    params_dict = paramsTable.paramsDict()
    params_d4 = params_dict["sadz"]
    params_d3 = params_dict["sdadz"][1:4]
    params_d4_ATM_G = params_dict["HF_ATM_OPT_START"]
    params_d4_ATM = params_dict["HF_ATM_OPT_OUT"]
    params_d4_ATM_OPT_ALL = params_dict["SAPT0_adz_BJ_ATM_OUT"]

    if compute_d3:
        print(f"Computing D3 values for {label}...")
        df[f"-D3 ({label})"] = df.apply(
            lambda r: jeff.compute_bj(params_d3, r["D3Data"]),
            axis=1,
        )
    print(f"Computing D4 2B values for {label}...")
    params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params_d4)

    df[f"-D4 ({label})"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B,
            params_ATM,
        ),
        axis=1,
    )
    print(f"Computing D4-ATM values for {label}...")
    params_2B_2, params_ATM_2 = paramsTable.generate_2B_ATM_param_subsets(
        params_d4_ATM_G
    )
    df[f"-D4 ({label}) ATM G"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B_2,
            params_ATM_2,
        ),
        axis=1,
    )
    print(f"Computing D4 2B values (ATM PARAMS) for {label}...")
    df[f"-D4 2B@ATM_params ({label}) G"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B_2,
            params_ATM,
        ),
        axis=1,
    )
    print(params_2B, params_2B_2, sep="\n")
    print(params_ATM, params_ATM_2, sep="\n")
    params_2B_2, params_ATM_2 = paramsTable.generate_2B_ATM_param_subsets(params_d4_ATM)
    df[f"-D4 ({label}) ATM"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
            row,
            params_2B_2,
            params_ATM_2,
        ),
        axis=1,
    )

    params_2B_2, params_ATM_2 = paramsTable.generate_2B_ATM_param_subsets(
        params_d4_ATM_OPT_ALL
    )
    df[f"-D4 ({label}) ATM ALL"] = df.apply(
        lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
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
        [
            "SAPT_DFT_atz_IE",
            "SAPT_DFT_atz_3_IE",
            "SAPT_DFT_OPT_END3",
            "SAPT_DFT_atz_3_IE",
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
    benchmark_label="Benchmark",
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    each bases element should be a list of 4 strings:
    [[
        df_column_for_IE_method_diff,
        df_column_for_label,
        params_name,
        df_column_for_elst_exch_indu_sum
    ]
    ...
    ]
    """
    params_dict = paramsTable.paramsDict()
    plot_vals = {}
    for i in bases:
        params_d4 = params_dict[i[2]]
        params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params_d4)
        df[f"-D4 ({i[1]})"] = df.apply(
            lambda row: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(
                row,
                params_2B,
                params_ATM,
            ),
            axis=1,
        )
        diff = f"{i[1]}_diff"
        d4_diff = f"{i[1]}_d4_diff"
        df[diff] = df[benchmark_label] - df[i[0]]
        df[d4_diff] = df[benchmark_label] - df[i[3]] - df[f"-D4 ({i[1]})"]
        print(f'"{diff}",')
        print(f'"{d4_diff}",')
    return df


def compute_d4_from_opt_params_TT(
    df: pd.DataFrame,
    bases=[
        # "DF_col_for_IE": "PARAMS_NAME"
        [
            "SAPT0_adz_IE",
            "SAPT0_adz_3_IE_TT_ALL",
            "SAPT0_adz_BJ_ATM_TT_5p",
            "SAPT0_adz_3_IE",
        ],
        # ["SAPT0_adz_IE", "SAPT0_adz_3_IE_TT", "HF_ATM_TT_OPT_START", "SAPT0_adz_3_IE"],
        # ["SAPT0_adz_IE", "SAPT0_adz_3_IE_TT_OPT", "HF_ATM_OPT_OUT", "SAPT0_adz_3_IE"],
    ],
) -> pd.DataFrame:
    """
    compute_D3_D4_values_for_params
    """
    params_dict = paramsTable.paramsDict()
    plot_vals = {}
    for i in bases:
        params_d4 = params_dict[i[2]]
        params_2B, params_ATM = paramsTable.generate_2B_ATM_param_subsets(params_d4)
        params_2B[-1] = 1.0
        params_ATM[-1] = 1.0
        print(params_2B, params_ATM, sep="\n")
        df[f"-D4(ATM TT) ({i[1]})"] = df.apply(
            lambda row: locald4.compute_disp_2B_BJ_ATM_TT_dimer(
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
    ax.legend(loc="upper left", fontsize="9")
    ax.set_xlabel("Database", fontsize="12")
    ax.set_ylabel("Error (kcal/mol)", fontsize="14")
    if title_name is not None:
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
    ax.legend(loc="upper left", fontsize="9")
    ax.set_xlabel("Database", fontsize="12")
    # ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", fontsize="14")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    ax.grid(color="gray", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):
        if n % 2 != 0:
            xtick.set_color("blue")
        else:
            xtick.set_color("red")

    if title_name is not None:
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


def plot_basis_sets_d4(df, build_df=False, df_out: str = "basis_study", df_name=""):
    selected = df_out
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
                [
                    "SAPT0_adz_IE",
                    "SAPT0_adz_BJ_ATM",
                    "SAPT0_adz_BJ_ATM",
                    "SAPT0_adz_3_IE",
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
        # f"Basis Set Comparison Across All Dimers ({len(df)})",
        None,
        f"{selected}_d4",
        bottom=0.30,
    )
    plot_violin_d3_d4_ALL_zoomed_min_max(
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
        "",  # f"All Dimers (8299)",
        f"{selected}_d4_zoomed",
        bottom=0.45,
        ylim=[-5, 5],
        legend_loc="lower right",
        transparent=True,
        # figure_size=(6, 6),
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
        # f"All Dimers ({len(df)})",
        None,
        f"{selected}_d4_opt_vs_adz",
        ylim=[-15, 14],
        bottom=0.35,
    )
    return df


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
    params_dict = paramsTable.paramsDict()
    plot_vals = {}
    for i in bases:
        params_d3 = params_dict[i[2]][0][1:4]
        print(params_d3)
        df[f"-D3 ({i[1]})"] = df.apply(
            lambda row: jeff.compute_BJ_CPP(
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


def plot_basis_sets_d3(df, build_df=False, df_out: str = "basis_study"):
    selected = df_out
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
        # f"{len(df)} Dimers With Different Basis Sets (D3)",
        None,
        f"{selected}_d3",
        bottom=0.30,
    )
    plot_violin_d3_d4_ALL_zoomed_min_max(
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
        "",  # f"All Dimers (8299)",
        f"{selected}_d3_zoomed",
        bottom=0.45,
        ylim=[-5, 5],
        legend_loc="lower right",
        transparent=True,
        # figure_size=(6, 6),
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
        # f"{len(df)} Dimers With Different Basis Sets (D3)",
        None,
        f"{selected}_d3_opt_vs_adz",
        bottom=0.35,
        ylim=[-15, 15],
    )

    return df


def plot_ie_curve(
    df,
    elst_col,
    exch_col,
    indu_col,
    disp_col,
    db="NBC10",
    system_num=0,
):
    df_sys = df[(df["DB"] == db) & (df["System #"] == system_num)]
    print(df_sys.columns.values)
    df_sys.sort_values(by="R", inplace=True)
    df.reset_index(drop=True, inplace=True)
    pd.set_option("display.max_columns", None)
    tools.print_cartesians(df_sys.iloc[0]["Geometry"])
    print()
    print(df_sys["R"].to_list())
    tools.print_cartesians(df_sys.iloc[len(df_sys) - 1]["Geometry"])
    print(df_sys)
    return


def plotting_setup(
    df, build_df=False, df_out: str = "plots/basis_study.pkl", compute_d3=True
):
    df, selected = df
    selected = selected.split("/")[-1].split(".")[0]
    df_out = f"plots/{selected}.pkl"
    if build_df:
        df = compute_D3_D4_values_for_params_for_plotting(df, "adz", compute_d3)
        df = compute_D3_D4_values_for_params_for_plotting(df, "jdz", compute_d3)
        df = compute_d4_from_opt_params(df)
        df["SAPT0-D4/aug-cc-pVDZ"] = df.apply(
            lambda row: row["SAPT0_adz_3_IE"] + row["-D4 (adz)"],
            axis=1,
        )
        df["SAPT0-D4(ATM)/aug-cc-pVDZ"] = df.apply(
            lambda row: row["SAPT0_adz_3_IE"] + row["-D4 (adz) ATM"],
            axis=1,
        )
        df["SAPT0-D4(ATM ALL)/aug-cc-pVDZ"] = df.apply(
            lambda row: row["SAPT0_adz_3_IE"] + row["-D4 (adz) ATM ALL"],
            axis=1,
        )
        df["SAPT0_adz_ATM_opt_all_diff"] = (
            df["Benchmark"] - df["SAPT0-D4(ATM ALL)/aug-cc-pVDZ"]
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
    # plot_violin_d3_d4_ALL(
    #     df,
    #     {
    #         "0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
    #         "0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
    #         "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
    #         "0-D4(ATM)/aDZ": "SAPT0_dz_3_IE_ATM_SHARED_d4_diff",
    #         "0-D4(2B ATM)/aDZ": "adz_diff_d4_ATM",
    #         "0-D4(2B@G ATM)/aDZ": "adz_diff_d4_2B@ATM_G",
    #         "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
    #         # "0-D4(ATM TT)/aDZ": "SAPT0_adz_3_IE_TT_OPT_d4_diff",
    #         "0/jDZ": "SAPT0_jdz_3_IE_diff",
    #         "0/aDZ": "SAPT0_adz_3_IE_diff",
    #         "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
    #         "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
    #         "SAPT(DFT)-D4/aTZ": "SAPT_DFT_atz_3_IE_d4_diff",
    #         "SAPT(DFT)/aTZ": "SAPT_DFT_atz_3_IE_diff",
    #     },
    #     None,
    #     f"{selected}_ATM_DFT",
    #     bottom=0.45,
    #     ylim=[-18, 22],
    #     # figure_size=(6, 6),
    #     dpi=1200,
    #     pdf=False,
    # )
    if True:
        print(df[["SAPT_DFT_atz_3_IE_diff", "SAPT_DFT_adz_3_IE_diff"]])
        # plot_violin_d3_d4_ALL_zoomed(
        plot_violin_d3_d4_ALL_zoomed_min_max(
            df,
            {
                "0/jDZ": "SAPT0_jdz_3_IE_diff",
                "0/aDZ": "SAPT0_adz_3_IE_diff",
                "0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
                # "0-D3MBJ(ATM)/jDZ": "jdz_diff_d3mbj_atm",
                "0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
                # "0-D3MBJ(ATM)/aDZ": "adz_diff_d3mbj_atm",
                "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
                # "0-D4(ATM)/aDZ": "SAPT0_dz_3_IE_ATM_SHARED_d4_diff",
                "0-D4(ATM)/aDZ": "SAPT0_adz_ATM_opt_all_diff",
                "0-D4(ATMu)/aDZ": "adz_diff_d4_ATM",  # (2B ATM) renamed to (ATMu)
                # "0-D4(2B@G ATM)/aDZ": "adz_diff_d4_2B@ATM_G",
                # "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
                # "0-D4(ATM TT ALL)/aDZ": "SAPT0_adz_3_IE_TT_ALL_d4_diff",
                "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
                # "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
                "SAPT(DFT)/aTZ": "SAPT_DFT_atz_3_IE_diff",
            },
            "",  # f"All Dimers (8299)",
            # f"8299 Dimer Dataset",
            f"{selected}_ATM2",
            bottom=0.45,
            ylim=[-5, 5],
            legend_loc="upper right",
            transparent=True,
            # figure_size=(6, 6),
        )
        plot_violin_d3_d4_ALL(
            df,
            {
                "0/jDZ": "SAPT0_jdz_3_IE_diff",
                "0/aDZ": "SAPT0_adz_3_IE_diff",
                "0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
                # "0-D3MBJ(ATM)/jDZ": "jdz_diff_d3mbj_atm",
                "0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
                # "0-D3MBJ(ATM)/aDZ": "adz_diff_d3mbj_atm",
                "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
                # "0-D4(ATM)/aDZ": "SAPT0_dz_3_IE_ATM_SHARED_d4_diff",
                "0-D4(ATM)/aDZ": "SAPT0_adz_ATM_opt_all_diff",
                "0-D4(ATMu)/aDZ": "adz_diff_d4_ATM",  # (2B ATM) renamed to (ATMu)
                # "0-D4(2B@G ATM)/aDZ": "adz_diff_d4_2B@ATM_G",
                # "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
                # "0-D4(ATM TT ALL)/aDZ": "SAPT0_adz_3_IE_TT_ALL_d4_diff",
                "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
                # "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
                "SAPT(DFT)/aTZ": "SAPT_DFT_atz_3_IE_diff",
            },
            "",  # f"All Dimers (8299)",
            f"{selected}_ATM",
            bottom=0.45,
            ylim=[-18, 26],
            legend_loc="upper right",
            # figure_size=(6, 6),
        )
        plot_violin_d3_d4_ALL(
            df,
            {
                "0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
                "0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
                "0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
                "0/jDZ": "SAPT0_jdz_3_IE_diff",
                "0/aDZ": "SAPT0_adz_3_IE_diff",
                "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
                "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
                "SAPT(DFT)/aTZ": "SAPT_DFT_atz_3_IE_diff",
                "SAPT(DFT)-D4/aTZ": "SAPT_DFT_atz_3_IE_d4_diff",
            },
            # f"All Dimers (8299)",
            "",
            f"{selected}_saptdft_d4_dhf",
            bottom=0.45,
            ylim=[-18, 26],
            # figure_size=(6, 6),
        )
        plot_dbs_d3_d4(
            df,
            "adz_diff_d4",
            "adz_diff_d4_ATM_G",
            "-D4",
            "-D4(ATM)",
            bottom=0.35,
            title_name=None,
            # title_name=f"DB Breakdown SAPT0-D4/aug-cc-pVDZ ({selected})",
            # title_name=f"-D4 Two-Body versus Three-Body (ATM)",
            pfn=f"{selected}_db_breakdown_2B_ATM",
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
            # "All Dimers (8299)",
            None,
            f"{selected}_basis_set",
        )
        # charged
        assert df["SAPT_DFT_adz_3_IE_diff"].isnull().sum() == 0
        assert df["SAPT_DFT_atz_3_IE_diff"].isnull().sum() == 0
        df_charged = get_charged_df(df)
        # ensure that SAPT_DFT_atz_3_IE_diff does not have NaN values
        pd.set_option("display.max_rows", None)
        df_charged["SAPT_DFT_adz_t"] = df_charged["SAPT_DFT_adz"].apply(lambda x: x[0])
        assert df_charged["SAPT_DFT_adz_3_IE_diff"].isnull().sum() == 0
        assert df_charged["SAPT_DFT_atz_3_IE_diff"].isnull().sum() == 0
        plot_violin_d3_d4_ALL(
            df_charged,
            {
                "0/jDZ": "SAPT0_jdz_3_IE_diff",
                "0/aDZ": "SAPT0_adz_3_IE_diff",
                "0-D3/jDZ": "jdz_diff_d3",
                "0-D3/aDZ": "adz_diff_d3",
                # "0-D3MBJ(ATM)/jDZ": "jdz_diff_d3mbj_atm",
                # "0-D3MBJ(ATM)/aDZ": "adz_diff_d3mbj_atm",
                "0-D4/jDZ": "jdz_diff_d4",
                "0-D4/aDZ": "adz_diff_d4",
                # "0-D4(2B@G ATM@G)/jDZ": "jdz_diff_d4_ATM_G",
                # "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
                "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
                "SAPT(DFT)/aTZ": "SAPT_DFT_atz_3_IE_diff",
            },
            # f"Charged Dimers ({len(df_charged)})",
            None,
            f"{selected}_charged",
            bottom=0.42,
            ylim=[-10, 15],
        )
    return df


def select_element(vals, ind):
    if vals is not None:
        return vals[ind]
    else:
        return np.nan


def plotting_setup_dft(
    df, build_df=False, df_out: str = "plots/basis_study.pkl", compute_d3=True
):
    df, selected = df
    selected = selected.split("/")[-1].split(".")[0]
    df_out = f"plots/{selected}.pkl"
    if build_df:
        for basis in ["adz", "atz"]:
            df[f"SAPT_DFT_{basis}_3_IE"] = df.apply(
                lambda x: x[f"SAPT_DFT_{basis}"][1]
                + x[f"SAPT_DFT_{basis}"][2]
                + x[f"SAPT_DFT_{basis}"][3],
                axis=1,
            )
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT_DFT_adz_IE",
                    "SAPT_DFT_adz_3_IE_ATM",
                    "SAPT_DFT_atz_3_IE_ATM_LINKED",
                    "SAPT_DFT_adz_3_IE",
                ],
                [
                    "SAPT_DFT_adz_IE",
                    "SAPT_DFT_adz_3_IE",
                    "SAPT_DFT_OPT_END3",
                    "SAPT_DFT_adz_3_IE",
                ],
                [
                    "SAPT_DFT_atz_IE",
                    "SAPT_DFT_atz_3_IE",
                    "SAPT_DFT_atz_3_IE",
                    "SAPT_DFT_atz_3_IE",
                ],
                [
                    "SAPT_DFT_atz_IE",
                    "SAPT_DFT_atz_3_IE_ATM",
                    "SAPT_DFT_atz_3_IE_ATM_LINKED",
                    "SAPT_DFT_atz_3_IE",
                ],
            ],
        )
        print(df.columns.values)
        # Need to get components
        for basis in ["adz", "atz", "tz", "jdz"]:
            df[f"SAPT0_adz_d4"] = df.apply(
                lambda x: x[f"SAPT0_adz_3_IE"] + x[f"-D4 (SAPT0_adz_3_IE)"], axis=1
            )
            df[f"SAPT0_{basis}_total"] = df.apply(
                lambda x: x[f"SAPT0_{basis}"][0], axis=1
            )
            df[f"SAPT0_{basis}_elst"] = df.apply(
                lambda x: x[f"SAPT0_{basis}"][1], axis=1
            )
            df[f"SAPT0_{basis}_exch"] = df.apply(
                lambda x: x[f"SAPT0_{basis}"][2], axis=1
            )
            df[f"SAPT0_{basis}_indu"] = df.apply(
                lambda x: x[f"SAPT0_{basis}"][3], axis=1
            )
            df[f"SAPT0_{basis}_disp"] = df.apply(
                lambda x: x[f"SAPT0_{basis}"][4], axis=1
            )
            df[f"SAPT0_{basis}_3_IE"] = df.apply(
                lambda x: x[f"SAPT0_{basis}_elst"]
                + x[f"SAPT0_{basis}_exch"]
                + x[f"SAPT0_{basis}_indu"],
                axis=1,
            )
        for basis in ["adz", "atz"]:
            df[f"SAPT_DFT_{basis}_total"] = df.apply(
                lambda x: x[f"SAPT_DFT_{basis}"][0], axis=1
            )
            df[f"SAPT_DFT_{basis}_elst"] = df.apply(
                lambda x: select_element(x[f"SAPT_DFT_{basis}"], 1), axis=1
            )
            df[f"SAPT_DFT_{basis}_exch"] = df.apply(
                lambda x: select_element(x[f"SAPT_DFT_{basis}"], 2), axis=1
            )
            df[f"SAPT_DFT_{basis}_indu"] = df.apply(
                lambda x: select_element(x[f"SAPT_DFT_{basis}"], 3), axis=1
            )
            df[f"SAPT_DFT_{basis}_disp"] = df.apply(
                lambda x: select_element(x[f"SAPT_DFT_{basis}"], 4), axis=1
            )
            df[f"SAPT_DFT_{basis}_3_IE"] = df.apply(
                lambda x: x[f"SAPT_DFT_{basis}_elst"]
                + x[f"SAPT_DFT_{basis}_exch"]
                + x[f"SAPT_DFT_{basis}_indu"],
                axis=1,
            )
            df[f"SAPT_DFT_{basis}_3_IE_d4"] = df.apply(
                lambda x: x[f"SAPT_DFT_{basis}_3_IE"]
                + x[f"-D4 (SAPT_DFT_{basis}_3_IE)"],
                axis=1,
            )
            df[f"SAPT_DFT_{basis}_3_IE_d4_ATM"] = df.apply(
                lambda x: x[f"SAPT_DFT_{basis}_3_IE"]
                + x[f"-D4 (SAPT_DFT_{basis}_3_IE_ATM)"],
                axis=1,
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
        ylim=[-16, 30],
        transparent=False,
    )
    return df


def plotting_setup_dft_ddft(
    selected,
    build_df=False,
    df_out: str = "plots/ddft_study.pkl",
    split_components=False,
):
    reference = "SAPT2+3(CCD)DMP2"
    ref_basis = "aTZ"
    if build_df:
        df = pd.read_pickle(selected)
        len_before = len(df)
        df.dropna(subset=['SAPT_DFT_pbe0_adz', 'SAPT_DFT_pbe0_atz', "C6s"], inplace=True)
        print(f"Removed {len_before - len(df)} NaN values")

        df = df[df["SAPT_DFT_pbe0_adz"].notna()].copy()

        def compute_saptdft_ddft_ie(r, b):
            return r[f"SAPT_DFT_pbe0_{b}"][1] + r[f"SAPT_DFT_pbe0_{b}"][2] + r[f"SAPT_DFT_pbe0_{b}"][3] + r[f"SAPT_DFT_pbe0_{b}_D4_IE"] + r[f"SAPT_DFT_pbe0_{b}_dDFT"] - r[f"SAPT_DFT_pbe0_{b}_dHF"]

        # SAPT(DFT) - aDZ
        # df["SAPT_DFT_D4_pbe0_adz_total"] = df.apply(
        #     lambda x: x["SAPT_DFT_pbe0_adz"][1]
        #     + x["SAPT_DFT_pbe0_adz"][2]
        #     + x["SAPT_DFT_pbe0_adz"][3]
        #     + x["SAPT_DFT_pbe0_adz_D4_IE"]
        #     + x["SAPT_DFT_pbe0_adz_dDFT"]
        #     - x["SAPT_DFT_pbe0_adz_dHF"],
        #     axis=1,
        # )
        df["SAPT_DFT_D4_pbe0_adz_total"] = df.apply(
            lambda x: compute_saptdft_ddft_ie(x, "adz"), 
            axis=1,
        )
        df["SAPT_DFT_pbe0_adz_total"] = df.apply(
            lambda x: x["SAPT_DFT_pbe0_adz"][0],
            axis=1,
        )

        df["DFT-D4/aDZ"] = df.apply(
            lambda x: x["SAPT_DFT_pbe0_adz_DFT_IE"] + x["SAPT_DFT_pbe0_adz_D4_IE"],
            axis=1,
        )
        # print(df[["SAPT_DFT_D4_pbe0_adz_total", "DFT-D4/aDZ"]])
        for n, i in df.iterrows():
            if not np.allclose(
                i["SAPT_DFT_D4_pbe0_adz_total"], i["DFT-D4/aDZ"], atol=1e-6
            ):
                print(
                    n,
                    i["benchmark ref energy"],
                    i["SAPT_DFT_D4_pbe0_adz_total"],
                    i["DFT-D4/aDZ"],
                )
        df.dropna(subset=["SAPT_DFT_D4_pbe0_adz_total"], inplace=True)

        assert np.allclose(
            df["SAPT_DFT_D4_pbe0_adz_total"], df["DFT-D4/aDZ"], atol=1e-6
        )
        df["SAPT_DFT_pbe0_adz_elst"] = df["SAPT_DFT_pbe0_adz"].apply(lambda x: x[1])
        df["SAPT_DFT_pbe0_adz_exch"] = df["SAPT_DFT_pbe0_adz"].apply(lambda x: x[2])
        df["SAPT_DFT_pbe0_adz_indu"] = df["SAPT_DFT_pbe0_adz"].apply(lambda x: x[3])
        df["SAPT_DFT_pbe0_adz_disp"] = df["SAPT_DFT_pbe0_adz"].apply(lambda x: x[4])
        df["SAPT_DFT_pbe0_adz_3_IE"] = (
            df["SAPT_DFT_pbe0_adz_elst"]
            + df["SAPT_DFT_pbe0_adz_exch"]
            + df["SAPT_DFT_pbe0_adz_indu"]
        )
        df["SAPT_DFT_pbe0_adz_d4_disp"] = df.apply(
            lambda x:
            x["SAPT_DFT_pbe0_adz_dDFT"]
            - x["SAPT_DFT_pbe0_adz_dHF"]
            + x["SAPT_DFT_pbe0_adz_D4_IE"],
            axis=1,
        )
        df["SAPT0_adz"] = df.apply(
            lambda x: np.array(
                [
                    x["SAPT0 ELST ENERGY adz"]
                    + x["SAPT0 EXCH ENERGY adz"]
                    + x["SAPT0 IND ENERGY adz"]
                    + x["SAPT0 DISP ENERGY adz"],
                    x["SAPT0 ELST ENERGY adz"],
                    x["SAPT0 EXCH ENERGY adz"],
                    x["SAPT0 IND ENERGY adz"],
                    x["SAPT0 DISP ENERGY adz"],
                ]
            ) * h2kcalmol,
            axis=1,
        )
        df["SAPT0_adz_3_IE"] = df.apply(
            lambda x: (x["SAPT0 ELST ENERGY adz"]
            + x["SAPT0 EXCH ENERGY adz"]
            + x["SAPT0 IND ENERGY adz"]) * h2kcalmol,
            axis=1,
        )
        df["SAPT0_adz_total"] = df["SAPT0_adz"].apply(lambda x: x[0])
        df["SAPT0_adz_elst"] = df["SAPT0_adz"].apply(lambda x: x[1])
        df["SAPT0_adz_exch"] = df["SAPT0_adz"].apply(lambda x: x[2])
        df["SAPT0_adz_indu"] = df["SAPT0_adz"].apply(lambda x: x[3])
        df["SAPT0_adz_disp"] = df["SAPT0_adz"].apply(lambda x: x[4])
        print(df['SAPT_DFT_pbe0_atz'])

        # SAPT(DFT) - aTZ
        df["SAPT_DFT_D4_pbe0_atz_total"] = df.apply(
            lambda x: compute_saptdft_ddft_ie(x, "atz"),
            axis=1,
        )
        df["SAPT_DFT_pbe0_atz_total"] = df.apply(
            lambda x: x["SAPT_DFT_pbe0_atz"][0],
            axis=1,
        )

        df["DFT-D4/aTZ"] = df.apply(
            lambda x: x["SAPT_DFT_pbe0_atz_DFT_IE"] + x["SAPT_DFT_pbe0_atz_D4_IE"],
            axis=1,
        )
        # print(df[["SAPT_DFT_D4_pbe0_atz_total", "DFT-D4/atz"]])
        for n, i in df.iterrows():
            if not np.allclose(
                i["SAPT_DFT_D4_pbe0_atz_total"], i["DFT-D4/aTZ"], atol=1e-6
            ):
                print(
                    n,
                    i["benchmark ref energy"],
                    i["SAPT_DFT_D4_pbe0_atz_total"],
                    i["DFT-D4/aTZ"],
                )
        df.dropna(subset=["SAPT_DFT_D4_pbe0_atz_total"], inplace=True)

        assert np.allclose(
            df["SAPT_DFT_D4_pbe0_atz_total"], df["DFT-D4/aTZ"], atol=1e-6
        )
        df["SAPT_DFT_pbe0_atz_elst"] = df["SAPT_DFT_pbe0_atz"].apply(lambda x: x[1])
        df["SAPT_DFT_pbe0_atz_exch"] = df["SAPT_DFT_pbe0_atz"].apply(lambda x: x[2])
        df["SAPT_DFT_pbe0_atz_indu"] = df["SAPT_DFT_pbe0_atz"].apply(lambda x: x[3])
        df["SAPT_DFT_pbe0_atz_disp"] = df["SAPT_DFT_pbe0_atz"].apply(lambda x: x[4])
        df["SAPT_DFT_pbe0_atz_3_IE"] = (
            df["SAPT_DFT_pbe0_atz_elst"]
            + df["SAPT_DFT_pbe0_atz_exch"]
            + df["SAPT_DFT_pbe0_atz_indu"]
        )
        df["SAPT_DFT_pbe0_atz_d4_disp"] = df.apply(
            lambda x:
            x["SAPT_DFT_pbe0_atz_dDFT"]
            - x["SAPT_DFT_pbe0_atz_dHF"]
            + x["SAPT_DFT_pbe0_atz_D4_IE"],
            axis=1,
        )
        df[f"{reference} TOTAL ENERGY adz"] = df[f"{reference} TOTAL ENERGY adz"] * h2kcalmol
        df["SAPT0_atz"] = df.apply(
            lambda x: np.array(
                [
                    x["SAPT0 ELST ENERGY atz"]
                    + x["SAPT0 EXCH ENERGY atz"]
                    + x["SAPT0 IND ENERGY atz"]
                    + x["SAPT0 DISP ENERGY atz"],
                    x["SAPT0 ELST ENERGY atz"],
                    x["SAPT0 EXCH ENERGY atz"],
                    x["SAPT0 IND ENERGY atz"],
                    x["SAPT0 DISP ENERGY atz"],
                ]
            ) * h2kcalmol,
            axis=1,
        )
        df["SAPT0_atz_3_IE"] = df.apply(
            lambda x: (x["SAPT0 ELST ENERGY atz"]
            + x["SAPT0 EXCH ENERGY atz"]
            + x["SAPT0 IND ENERGY atz"]) * h2kcalmol,
            axis=1,
        )
        df["SAPT0_atz_total"] = df["SAPT0_atz"].apply(lambda x: x[0])
        df["SAPT0_atz_elst"] = df["SAPT0_atz"].apply(lambda x: x[1])
        df["SAPT0_atz_exch"] = df["SAPT0_atz"].apply(lambda x: x[2])
        df["SAPT0_atz_indu"] = df["SAPT0_atz"].apply(lambda x: x[3])
        df["SAPT0_atz_disp"] = df["SAPT0_atz"].apply(lambda x: x[4])

        df[f"{reference} ELST ENERGY"] = (
            df[f"{reference} ELST ENERGY {ref_basis.lower()}"] * h2kcalmol
        )
        # print(df[f"{reference} ELST ENERGY"])
        df[f"{reference} EXCH ENERGY"] = (
            df[f"{reference} EXCH ENERGY {ref_basis.lower()}"] * h2kcalmol
        )
        df[f"{reference} IND ENERGY"] = (
            df[f"{reference} IND ENERGY {ref_basis.lower()}"] * h2kcalmol
        )
        df[f"{reference} DISP ENERGY"] = (
            df[f"{reference} DISP ENERGY {ref_basis.lower()}"] * h2kcalmol
        )
        df[f"{reference} TOTAL ENERGY"] = (
            df[f"{reference} TOTAL ENERGY {ref_basis.lower()}"] * h2kcalmol
        )
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT0_adz_total",
                    "SAPT0_adz_3_IE",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_adz_3_IE",
                ],
                [
                    "SAPT0_atz_total",
                    "SAPT0_atz_3_IE",
                    "SAPT0_atz_3_IE_2B",
                    "SAPT0_atz_3_IE",
                ],
            ],
            benchmark_label="benchmark ref energy",
        )
        df[f"SAPT0_adz_d4"] = df.apply(
            lambda x: x[f"SAPT0_adz_3_IE"] + x[f"-D4 (SAPT0_adz_3_IE)"], axis=1
        )
        df[f"SAPT0_atz_d4"] = df.apply(
            lambda x: x[f"SAPT0_atz_3_IE"] + x[f"-D4 (SAPT0_atz_3_IE)"], axis=1
        )
        df.dropna(subset=["-D4 (SAPT0_adz_3_IE)"], inplace=True)
        df.dropna(subset=["-D4 (SAPT0_atz_3_IE)"], inplace=True)
        df["SAPT0-D4/aDZ"] = df.apply(
            lambda row: row["SAPT0_adz_3_IE"] + row["-D4 (SAPT0_adz_3_IE)"], axis=1
        )
    else:
        df = pd.read_pickle(df_out)
    basename_selected = selected.split("/")[-1].split(".")[0]
    assert df["SAPT_DFT_pbe0_adz_elst"].isnull().sum() == 0
    print(f"Length of df prior plotting: {len(df)}")
    dimer_dataset_size = len(df)
    # plot_violin_SAPT0_DFT_components(
    plot_violin_SAPT0_DFT_components(
        df,
        pfn=f"{basename_selected}_saptdft_components2",
        elst_vals={
            "name": "Electrostatics",
            "reference": [f"{reference}/{ref_basis} Ref.", f"{reference} ELST ENERGY"],
            "vals": {
                "SAPT0/aDZ": "SAPT0_adz_elst",
                "SAPT0/aTZ": "SAPT0_atz_elst",
                "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_elst",
            },
        },
        exch_vals={
            "name": "Exchange",
            "reference": [f"{reference}/{ref_basis} Ref.", f"{reference} EXCH ENERGY"],
            "vals": {
                "SAPT0/aDZ": "SAPT0_adz_exch",
                "SAPT0/aTZ": "SAPT0_atz_exch",
                "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_exch",
            },
        },
        indu_vals={
            "name": "Induction",
            "reference": [f"{reference}/{ref_basis} Ref.", f"{reference} IND ENERGY"],
            "vals": {
                "SAPT0/aDZ": "SAPT0_adz_indu",
                "SAPT0/aTZ": "SAPT0_atz_indu",
                "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_indu",
            },
        },
        disp_vals={
            "name": "Dispersion",
            "reference": [f"{reference}/{ref_basis} Ref.", f"{reference} DISP ENERGY"],
            "vals": {
                # "SAPT0/aDZ": "SAPT0_adz_indu",
                # "SAPT0/aTZ": "SAPT0_atz_indu",
                # "SAPT0-D4/aDZ": "-D4 (SAPT0_adz_3_IE)",
                "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_disp",
                "SAPT(DFT)-D4/aDZ": "SAPT_DFT_pbe0_adz_d4_disp",
            },
        },
        three_total_vals={
            "name": "(Elst. + Exch. + Indu.)",
            "reference": ["CCSD(T)/CBS IE Ref.", "benchmark ref energy"],
            "vals": {
                "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_3_IE",
                "SAPT0/aDZ": "SAPT0_adz_3_IE",
                "SAPT0/aTZ": "SAPT0_atz_3_IE",
            },
        },
        total_vals={
            "name": f"{dimer_dataset_size} Dimer Dataset",
            "reference": ["CCSD(T)/CBS IE Ref.", "benchmark ref energy"],
            "vals": {
                "SAPT0/aDZ": "SAPT0_adz_total",
                "SAPT0-D4/aDZ": "SAPT0-D4/aDZ",
                # "SAPT0/aTZ": "SAPT0_atz_total",
                "SAPT0-D4/aDZ": "SAPT0_adz_d4",
                "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_total",
                # "SAPT(DFT)D4/aDZ": "SAPT_DFT_D4_pbe0_adz_total",
                "PBE0-D4/aDZ IE": "SAPT_DFT_D4_pbe0_adz_total",
                "SAPT(DFT)/aTZ": "SAPT_DFT_pbe0_atz_total",
                "PBE0-D4/aTZ IE": "SAPT_DFT_D4_pbe0_atz_total",
                f"SAPT2+3(CCD)DMP2/aDZ": f"SAPT2+3(CCD)DMP2 TOTAL ENERGY adz",
                f"SAPT2+3(CCD)DMP2/{ref_basis}": f"{reference} TOTAL ENERGY",
                # "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4",
                # "SAPT(DFT)-D4(ATM)/aDZ": "SAPT_DFT_adz_3_IE_d4_ATM",
                # "SAPT(DFT)/aTZ": "SAPT_DFT_atz_total",
                # "SAPT(DFT)-D4/aTZ": "SAPT_DFT_atz_3_IE_d4",
                # "SAPT(DFT)-D4(ATM)/aTZ": "SAPT_DFT_atz_3_IE_d4_ATM",
            },
        },
        split_components=split_components,
        sub_fontsize=24,
        sub_rotation=35,
    )
    return df


def plotting_setup_G(
    df,
    build_df=False,
    df_out: str = "plots/basis_study.pkl",
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
        # title_name=f"-D4 Two-Body version Three-Body",
        title_name=None,
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
        # f"All Dimers with SAPT0 ({selected})",
        None,
        f"{selected}_qz_d3_d4_total_sapt0",
    )
    return


def plot_violin_d3_d4_ALL_zoomed(
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
    dpi=600,
    pdf=False,
    legend_loc="upper left",
) -> None:
    print(f"Plotting {pfn}")
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []

    annotations = []  # [(x, y, text), ...]
    cnt = 1
    for k, v in vals.items():
        df[v] = pd.to_numeric(df[v])
        df_sub = df[df[v].notna()].copy()
        vData.append(df_sub[v].to_list())
        k_label = "\\textbf{" + k + "}"
        # k_label = k
        vLabels.append(k_label)
        m = df_sub[v].max()
        rmse = df_sub[v].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[v].apply(lambda x: abs(x)).mean()
        max_error = df_sub[v].apply(lambda x: abs(x)).max()
        text = r"\textit{%.2f}" % mae
        text += "\n"
        text += r"\textbf{%.2f}" % rmse
        text += "\n"
        text += r"\textrm{%.2f}" % max_error
        annotations.append((cnt, m, text))
        cnt += 1

    pd.set_option("display.max_columns", None)
    # print(df[vals.values()].describe(include="all"))
    # transparent figure
    fig = plt.figure(dpi=dpi)
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[0.15, 1]
    )  # Adjust height ratios to change the size of subplots

    # Create the main violin plot axis
    ax = plt.subplot(gs[1])  # This will create the subplot for the main violin plot.
    # ax = plt.subplot(111)
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        quantiles=[[0.05, 0.95] for i in range(len(vData))],
        widths=widths,
    )
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cmeans"]):
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
    # color_gt_olympic_teal = (0 /255, 140/255,  149/255)  # Olympic teal
    # color_gt_bold_blue = (58/255, 93/255, 174/255)
    # colors = [color_gt_bold_blue if i % 2 == 0 else color_gt_olympic_teal for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        pc.set_alpha(0.6)
        # pc.set_alpha(1)

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
    ax.set_xticks(xs)
    # minor_yticks = np.arange(ylim[0], ylim[1], 2)
    # ax.set_yticks(minor_yticks, minor=True)

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="10")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    # lg.get_frame().set_alpha(None)
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    # ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k", fontsize="14")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")

    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)
    # Annotations of RMSE

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="10")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")
    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    ax_error = plt.subplot(gs[0], sharex=ax)
    # ax_error.spines['top'].set_visible(False)
    ax_error.spines["right"].set_visible(False)
    ax_error.spines["left"].set_visible(False)
    ax_error.spines["bottom"].set_visible(False)
    ax_error.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Synchronize the x-limits with the main subplot
    ax_error.set_xlim((0, len(vLabels)))
    ax_error.set_ylim(0, 1)  # Assuming the upper subplot should have no y range
    print(f"ax_error xlim: {ax_error.get_xlim()}")
    # Populate ax_error with error statistics through annotations
    # text = r"\textit{%.2f}" % mae
    # text += r"\textbf{%.2f}" % rmse
    # text += r"\textrm{%.2f}" % max_error
    error_labels = r"\textit{MAE}"
    error_labels += "\n"
    error_labels += r"\textbf{RMSE}"
    error_labels += "\n"
    error_labels += r"\textrm{MaxAE}"
    ax_error.annotate(
        error_labels,
        xy=(0, 1),  # Position at the vertical center of the narrow subplot
        xytext=(0, 0),
        color="black",
        fontsize="8",
        ha="center",
        va="center",
    )
    for idx, (x, y, text) in enumerate(annotations):
        print(f"Annotation: {x}, {y}, {text}")
        ax_error.annotate(
            text,
            xy=(x, 1),  # Position at the vertical center of the narrow subplot
            # xytext=(0, 0),
            xytext=(x, 0),
            color="black",
            fontsize="8",
            ha="center",
            va="center",
        )

    if title_name is not None:
        plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)

    if pdf:
        fn_pdf = f"plots/{pfn}_dbs_violin.pdf"
        fn_png = f"plots/{pfn}_dbs_violin.png"
        plt.savefig(
            fn_pdf,
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
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
            f"plots/{pfn}_dbs_violin.png",
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
        )
    plt.clf()
    return


def plot_component_violin_zoomed(
    vData,
    vLabels,
    annotations,
    title_name,
    widths=0.85,
    fontsize=8,
    sub_rotation=45,
    ylim=None,
    figure_size=None,
    dpi=600,
    transparent=True,
    legend_loc="upper right",
    pfn="los_SAPTDFT_zoomed",
    set_xlable=False,
    pdf=False,
    jpeg=True,
):
    image_ext = "png"
    if jpeg:
        image_ext = "jpeg"
    color1 = (220 / 255, 198 / 255, 135 / 255)  # Original color (yellow)
    print(f"Plotting {pfn}")
    # fig = plt.figure(facecolor=color1)
    fig = plt.figure(dpi=dpi)
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[0.15, 1]
    )  # Adjust height ratios to change the size of subplots

    # Create the main violin plot axis
    ax = plt.subplot(gs[1])  # This will create the subplot for the main violin plot.
    # ax = plt.subplot(111)
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        quantiles=[[0.05, 0.95] for i in range(len(vData))],
        widths=widths,
    )
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cmeans"]):
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

    color1 = (220 / 255, 198 / 255, 135 / 255)  # Original color (yellow)
    color2 = (35 / 255, 57 / 255, 120 / 255)  # Complementary color (deep blue)
    color3 = (220 / 255, 135 / 255, 135 / 255)  # Analogous color (red)
    color4 = (220 / 255, 220 / 255, 135 / 255)  # Analogous color (green)
    color5 = (135 / 255, 220 / 255, 198 / 255)  # Triadic color (teal)
    color6 = (198 / 255, 135 / 255, 220 / 255)  # Triadic color (purple)
    color_gt_blue = (0 / 255, 0 / 255, 128 / 255)  # Dark blue
    # color_gt_olympic_teal = (100 /255, 204/255,  201/255)  # Olympic teal
    darker_purple = (4 / 255, 36 / 255, 51 / 255)
    color_gt_olympic_teal = (0 / 255, 140 / 255, 149 / 255)  # Olympic teal
    color_gt_bold_blue = (58 / 255, 93 / 255, 174 / 255)
    colors = [
        color_gt_bold_blue if i % 2 == 0 else color_gt_olympic_teal
        for i in range(len(vLabels))
    ]
    colors = ["blue" if i % 2 == 0 else "green" for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        pc.set_alpha(0.6)
        # pc.set_alpha(1)

    # vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    print(xs, vLabels, len(vData))
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
    ax.set_xticks(xs)
    # minor_yticks = np.arange(ylim[0], ylim[1], 2)
    # ax.set_yticks(minor_yticks, minor=True)

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="12")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    # lg.get_frame().set_alpha(None)
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    # ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k", fontsize="14")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="16")

    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)
    # Annotations of RMSE

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="12")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    lg.get_frame().set_facecolor("none")  # Make the legend background transparent
    # legend.get_frame().set_alpha(0.5)  # Adjust the transparency level (0.0 to 1.0)
    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="16")
    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    ax_error = plt.subplot(gs[0], sharex=ax)
    # ax_error.spines['top'].set_visible(False)
    ax_error.spines["right"].set_visible(False)
    ax_error.spines["left"].set_visible(False)
    ax_error.spines["bottom"].set_visible(False)
    ax_error.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Synchronize the x-limits with the main subplot
    ax_error.set_xlim((0, len(vLabels)))
    ax_error.set_ylim(0, 1)  # Assuming the upper subplot should have no y range
    # Populate ax_error with error statistics through annotations
    # text = r"\textit{%.2f}" % mae
    # text += r"\textbf{%.2f}" % rmse
    # text += r"\textrm{%.2f}" % max_error
    error_labels = r"\textit{MAE}"
    error_labels += "\n"
    error_labels += r"\textbf{RMSE}"
    error_labels += "\n"
    error_labels += r"\textrm{MaxE}"
    error_labels += "\n"
    error_labels += r"\textrm{MinE}"
    ax_error.annotate(
        error_labels,
        xy=(0, 1),  # Position at the vertical center of the narrow subplot
        xytext=(0, 0.15),
        color="black",
        fontsize="10",
        ha="center",
        va="center",
    )
    for idx, (x, y, text) in enumerate(annotations):
        ax_error.annotate(
            text,
            xy=(x, 1),  # Position at the vertical center of the narrow subplot
            # xytext=(0, 0),
            xytext=(x, 0.15),
            color="black",
            fontsize="10",
            ha="center",
            va="center",
        )

    if title_name is not None:
        plt.title(f"{title_name}", fontsize="16")
    # fig.subplots_adjust(bottom=bottom)

    if pdf:
        fn_pdf = f"plots/{pfn}_components_TOTAL.pdf"
        fn_png = f"plots/{pfn}_components_TOTAL.png"
        plt.savefig(
            fn_pdf,
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
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
            f"plots/{pfn}_dbs_violin.{image_ext}",
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
        )
    plt.clf()
    return


def plot_violin_d3_d4_ALL_zoomed_min_max(
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
    dpi=800,
    pdf=False,
    jpeg=True,
    legend_loc="upper left",
) -> None:
    print(f"Plotting {pfn}")
    image_ext = "png"
    if jpeg:
        image_ext = "jpeg"

    vLabels, vData = [], []
    annotations = []  # [(x, y, text), ...]
    cnt = 1
    for k, v in vals.items():
        df[v] = pd.to_numeric(df[v])
        df_sub = df[df[v].notna()].copy()
        vData.append(df_sub[v].to_list())
        k_label = "\\textbf{" + k + "}"
        # k_label = k
        vLabels.append(k_label)
        m = df_sub[v].max()
        rmse = df_sub[v].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[v].apply(lambda x: abs(x)).mean()
        max_pos_error = df_sub[v].apply(lambda x: x).max()
        max_neg_error = df_sub[v].apply(lambda x: x).min()
        text = r"\textit{%.2f}" % mae
        text += "\n"
        text += r"\textbf{%.2f}" % rmse
        text += "\n"
        text += r"\textrm{%.2f}" % max_pos_error
        text += "\n"
        text += r"\textrm{%.2f}" % max_neg_error
        annotations.append((cnt, m, text))
        cnt += 1

    pd.set_option("display.max_columns", None)
    # print(df[vals.values()].describe(include="all"))
    # transparent figure
    fig = plt.figure(dpi=dpi)
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[0.22, 1]
    )  # Adjust height ratios to change the size of subplots

    # Create the main violin plot axis
    ax = plt.subplot(gs[1])  # This will create the subplot for the main violin plot.
    # ax = plt.subplot(111)
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        quantiles=[[0.05, 0.95] for i in range(len(vData))],
        widths=widths,
    )
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cmeans"]):
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
    # color_gt_olympic_teal = (0 /255, 140/255,  149/255)  # Olympic teal
    # color_gt_bold_blue = (58/255, 93/255, 174/255)
    # colors = [color_gt_bold_blue if i % 2 == 0 else color_gt_olympic_teal for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        pc.set_alpha(0.6)
        # pc.set_alpha(1)

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
    ax.set_xticks(xs)
    # minor_yticks = np.arange(ylim[0], ylim[1], 2)
    # ax.set_yticks(minor_yticks, minor=True)

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="10")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    # lg.get_frame().set_alpha(None)
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    # ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k", fontsize="14")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")

    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)
    # Annotations of RMSE

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="10")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")
    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    ax_error = plt.subplot(gs[0], sharex=ax)
    # ax_error.spines['top'].set_visible(False)
    ax_error.spines["right"].set_visible(False)
    ax_error.spines["left"].set_visible(False)
    ax_error.spines["bottom"].set_visible(False)
    ax_error.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Synchronize the x-limits with the main subplot
    ax_error.set_xlim((0, len(vLabels)))
    ax_error.set_ylim(0, 1)  # Assuming the upper subplot should have no y range
    # Populate ax_error with error statistics through annotations
    # text = r"\textit{%.2f}" % mae
    # text += r"\textbf{%.2f}" % rmse
    # text += r"\textrm{%.2f}" % max_error
    error_labels = r"\textit{MAE}"
    error_labels += "\n"
    error_labels += r"\textbf{RMSE}"
    error_labels += "\n"
    error_labels += r"\textrm{MaxE}"
    error_labels += "\n"
    error_labels += r"\textrm{MinE}"
    ax_error.annotate(
        error_labels,
        xy=(0, 1),  # Position at the vertical center of the narrow subplot
        xytext=(0, 0.2),
        color="black",
        fontsize="8",
        ha="center",
        va="center",
    )
    for idx, (x, y, text) in enumerate(annotations):
        ax_error.annotate(
            text,
            xy=(x, 1),  # Position at the vertical center of the narrow subplot
            # xytext=(0, 0),
            xytext=(x, 0.2),
            color="black",
            fontsize="8",
            ha="center",
            va="center",
        )

    if title_name is not None:
        plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)

    if pdf:
        fn_pdf = f"plots/{pfn}_dbs_violin.pdf"
        fn_png = f"plots/{pfn}_dbs_violin.png"
        plt.savefig(
            fn_pdf,
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
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
            f"plots/{pfn}_dbs_violin.{image_ext}",
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
        )
    plt.clf()
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
    dpi=600,
    pdf=False,
    jpeg=True,
    legend_loc="upper left",
) -> None:
    """ """
    print(f"Plotting {pfn}")
    image_ext = "png"
    if jpeg:
        image_ext = "jpeg"
    vLabels, vData = [], []

    annotations = []  # [(x, y, text), ...]
    cnt = 1
    for k, v in vals.items():
        df[v] = pd.to_numeric(df[v])
        df_sub = df[df[v].notna()].copy()
        vData.append(df_sub[v].to_list())
        k_label = "\\textbf{" + k + "}"
        # k_label = k
        vLabels.append(k_label)
        m = df_sub[v].max()
        rmse = df_sub[v].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[v].apply(lambda x: abs(x)).mean()
        max_error = df_sub[v].apply(lambda x: abs(x)).max()
        text = r"\textit{%.2f}" % mae
        text += "\n"
        text += r"\textbf{%.2f}" % rmse
        text += "\n"
        text += r"\textrm{%.2f}" % max_error
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
    # color_gt_olympic_teal = (0 /255, 140/255,  149/255)  # Olympic teal
    # color_gt_bold_blue = (58/255, 93/255, 174/255)
    # colors = [color_gt_bold_blue if i % 2 == 0 else color_gt_olympic_teal for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        # pc.set_alpha(0.6)
        pc.set_alpha(1)

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
    ax.set_xticks(xs)
    # minor_yticks = np.arange(ylim[0], ylim[1], 2)
    # ax.set_yticks(minor_yticks, minor=True)

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="10")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    # lg.get_frame().set_alpha(None)
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    # ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k", fontsize="14")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")

    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)
    # Annotations of RMSE
    for x, y, text in annotations:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y + 0.1),
            color="black",
            fontsize="10.0",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    if title_name is not None:
        plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)

    if pdf:
        fn_pdf = f"plots/{pfn}_dbs_violin.pdf"
        fn_png = f"plots/{pfn}_dbs_violin.png"
        plt.savefig(
            fn_pdf,
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
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
            f"plots/{pfn}_dbs_violin.{image_ext}",
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
        )
    plt.clf()
    return


def collect_component_data(df, vals, extended_errors=False, verbose=False):
    vLabels, vData = [], []
    annotations = []  # [(x, y, text), ...]
    cnt = 1
    for k, v in vals["vals"].items():
        print(k)
        try:
            df[v] = pd.to_numeric(df[v])
        except Exception as e:
            print(e)
            print(df[v])
            SystemExit()
        df[f"{v}_diff"] = df[vals["reference"][1]] - df[v]
        df_sub = df[df[f"{v}_diff"].notna()].copy()
        vData.append(df_sub[f"{v}_diff"].to_list())
        vLabels.append(k)
        m = df_sub[f"{v}_diff"].max()
        rmse = df_sub[f"{v}_diff"].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[f"{v}_diff"].apply(lambda x: abs(x)).mean()
        if extended_errors:
            max_pos_error = df_sub[f"{v}_diff"].apply(lambda x: x).max()
            max_neg_error = df_sub[f"{v}_diff"].apply(lambda x: x).min()
            text = r"\textit{%.2f}" % mae
            text += "\n"
            text += r"\textbf{%.2f}" % rmse
            text += "\n"
            text += r"\textrm{%.2f}" % max_pos_error
            text += "\n"
            text += r"\textrm{%.2f}" % max_neg_error
        else:
            max_error = df_sub[f"{v}_diff"].apply(lambda x: abs(x)).max()
            text = r"\textit{%.2f}" % mae
            text += "\n"
            text += r"\textbf{%.2f}" % rmse
            text += "\n"
            text += r"\textrm{%.2f}" % max_error
        annotations.append((cnt, m, text))
        cnt += 1
    if verbose:
        tmp_df = pd.DataFrame(vData, index=vLabels).T
        tmp_df[vals["reference"][0]] = df[vals["reference"][1]]
        print(tmp_df)
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
    ax,
    vData,
    vLabels,
    annotations,
    title_name,
    ylabel,
    widths=0.85,
    fontsize=8,
    sub_rotation=45,
    ylim=None,
):
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        quantiles=[[0.05, 0.95] for _ in range(len(vData))],
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
    ylim_empty = False
    if ylim is None:
        ylim = ax.get_ylim()
        ylim_empty = True
    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)
    diff = abs(ylim[1] - ylim[0])
    print(diff)
    if diff > 20 and ylim_empty:
        ax.set_ylim((ylim[0], int(ylim[1] + diff * 0.40)))
    elif ylim_empty:
        ax.set_ylim((ylim[0], int(ylim[1] + diff * 0.40)))
    else:
        ax.set_ylim(ylim)
    # set ytick fontsize
    ax.tick_params(axis="y", labelsize=fontsize - 2)
    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for _ in range(len(xs_error))],
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
    ax.set_xticks(xs)
    plt.setp(
        ax.set_xticklabels(vLabels), rotation=sub_rotation, fontsize=f"{fontsize-2}"
    )
    ax.set_xlim((0, len(vLabels)))
    # lg = ax.legend(loc="upper left", edgecolor="black", fontsize="8")
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    # set minor ticks to be between major ticks

    ax.grid(color="grey", which="major", linewidth=0.5, alpha=0.3)
    ax.grid(color="grey", which="minor", linewidth=0.5, alpha=0.3)
    # Set subplot title
    if ylabel is not None and len(ylabel) > 0:
        ylabel = f"{ylabel} Error\n" + r" ($\mathrm{kcal\cdot mol^{-1}}$)"
        ax.set_ylabel(ylabel, color="k", fontsize=f"{fontsize - 1}")
    title_color = "k"
    if title_name == "Electrostatics":
        title_color = "red"
    elif title_name == "Exchange":
        title_color = "blue"
    elif title_name == "Induction":
        title_color = "green"
    elif title_name == "Dispersion":
        title_color = "orange"
    ax.set_title(title_name, color=title_color, fontsize=f"{fontsize + 1}")

    # Annotations of RMSE
    for x, y, text in annotations:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y + 0.1),
            color="black",
            fontsize=f"{fontsize - 2}",
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
        "reference": ["SAPT0/aDZ Ref.", "SAPT0_adz_elst"],
        "vals": {
            "SAPT(DFT)/aDZ": "SAPT_DFT_adz_elst",
            "SAPT(DFT)/aTZ": "SAPT_DFT_atz_elst",
            "SAPT0/jDZ": "SAPT0_jdz_elst",
            "SAPT0/aTZ": "SAPT0_atz_elst",
        },
    },
    exch_vals={
        "name": "Exchange",
        "reference": ["SAPT0/aDZ Ref.", "SAPT0_adz_exch"],
        "vals": {
            "SAPT(DFT)/aDZ": "SAPT_DFT_adz_exch",
            "SAPT(DFT)/aTZ": "SAPT_DFT_atz_exch",
            "SAPT0/jDZ": "SAPT0_jdz_exch",
            "SAPT0/aTZ": "SAPT0_atz_exch",
        },
    },
    indu_vals={
        "name": "Induction",
        "reference": ["SAPT0/aDZ Ref.", "SAPT0_adz_indu"],
        "vals": {
            "SAPT(DFT)/aDZ": "SAPT_DFT_adz_indu",
            "SAPT(DFT)/aTZ": "SAPT_DFT_atz_indu",
            "SAPT0/jDZ": "SAPT0_jdz_indu",
            "SAPT0/aTZ": "SAPT0_atz_indu",
        },
    },
    disp_vals={
        "name": "Dispersion",
        "reference": ["SAPT0/aDZ Ref.", "SAPT0_adz_disp"],
        "vals": {
            "SAPT(DFT)/aDZ": "SAPT_DFT_adz_disp",
            "SAPT(DFT)/aTZ": "SAPT_DFT_atz_disp",
            "SAPT0/jDZ": "SAPT0_jdz_disp",
            "SAPT0/aTZ": "SAPT0_atz_disp",
            "-D4/aDZ (SAPT0_2B)": "-D4 (SAPT0_adz_3_IE)",
            "-D4/aDZ (SAPT_DFT_2B)": "-D4 (SAPT_DFT_adz_3_IE)",
            "-D4/aDZ (SAPT_DFT_ATM)": "-D4 (SAPT_DFT_adz_3_IE_ATM)",
        },
    },
    three_total_vals={
        "name": "(Elst. + Exch. + Indu.)",
        "reference": ["CCSD(T)/CBS Ref.", "Benchmark"],
        "vals": {
            "SAPT0/aDZ": "SAPT0_adz_3_IE",
            "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE",
            "SAPT(DFT)/aTZ": "SAPT_DFT_atz_3_IE",
        },
    },
    total_vals={
        "name": "(Elst. + Exch. + Indu. + Disp.)",
        "reference": ["CCSD(T)/CBS Ref.", "Benchmark"],
        "vals": {
            "SAPT0/aDZ": "SAPT0_adz_total",
            "SAPT0-D4/aDZ": "SAPT0_adz_d4",
            "SAPT(DFT)/aDZ": "SAPT_DFT_adz_total",
            "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4",
            "SAPT(DFT)-D4(ATM)/aDZ": "SAPT_DFT_adz_3_IE_d4_ATM",
            "SAPT(DFT)/aTZ": "SAPT_DFT_atz_total",
            "SAPT(DFT)-D4/aTZ": "SAPT_DFT_atz_3_IE_d4",
            "SAPT(DFT)-D4(ATM)/aTZ": "SAPT_DFT_atz_3_IE_d4_ATM",
        },
    },
    pfn: str = "sapt0_dft_components",
    transparent=False,
    widths=0.95,
    split_components=False,
    sub_fontsize=8,
    sub_rotation=45,
) -> None:
    print(f"Plotting {pfn}")
    if split_components:
        fig, axs = plt.subplots(
            nrows=1, ncols=4, figsize=(20, 5), dpi=1000, constrained_layout=True
        )
        three_total_ax = None
        total_ax = None
        exch_vals["reference"][0] = None
        indu_vals["reference"][0] = None
        disp_vals["reference"][0] = None
        elst_ax = axs[0]
        exch_ax = axs[1]
        indu_ax = axs[2]
        disp_ax = axs[3]
    else:
        fig, axs = plt.subplots(3, 2, figsize=(8, 6), dpi=1000)
        three_total_ax = axs[2, 0]
        total_ax = axs[2, 1]
        elst_ax = axs[0, 0]
        exch_ax = axs[0, 1]
        indu_ax = axs[1, 0]
        disp_ax = axs[1, 1]
    # add extra space for subplot titles
    fig.subplots_adjust(hspace=0.6, wspace=0.3)

    # Component Data
    print("\nELST")
    elst_data, elst_labels, elst_annotations = collect_component_data(df, elst_vals)
    print("\nEXCH")
    exch_data, exch_labels, exch_annotations = collect_component_data(df, exch_vals)
    print("\nINDU")
    indu_data, indu_labels, indu_annotations = collect_component_data(df, indu_vals)
    print("\nDISP")
    disp_data, disp_labels, disp_annotations = collect_component_data(df, disp_vals)
    print("\n3-TOTAL")
    three_total_data, three_total_labels, three_total_annotations = (
        collect_component_data(df, three_total_vals)
    )
    print("\nTOTAL")
    total_data, total_labels, total_annotations = collect_component_data(
        df, total_vals, extended_errors=True
    )

    # Plot violins
    plot_component_violin(
        elst_ax,
        elst_data,
        elst_labels,
        elst_annotations,
        elst_vals["name"],
        elst_vals["reference"][0],
        widths,
        fontsize=sub_fontsize,
        sub_rotation=sub_rotation,
        ylim=[-4, 6],
    )
    plot_component_violin(
        exch_ax,
        exch_data,
        exch_labels,
        exch_annotations,
        exch_vals["name"],
        exch_vals["reference"][0],
        widths,
        fontsize=sub_fontsize,
        sub_rotation=sub_rotation,
        ylim=[-4, 25],
    )
    plot_component_violin(
        indu_ax,
        indu_data,
        indu_labels,
        indu_annotations,
        indu_vals["name"],
        indu_vals["reference"][0],
        widths,
        fontsize=sub_fontsize,
        sub_rotation=sub_rotation,
        ylim=[-5, 15],
    )
    plot_component_violin(
        disp_ax,
        disp_data,
        disp_labels,
        disp_annotations,
        disp_vals["name"],
        disp_vals["reference"][0],
        widths,
        fontsize=sub_fontsize,
        sub_rotation=sub_rotation,
        ylim=[-12, 15],
    )

    if not split_components:
        plot_component_violin(
            three_total_ax,
            three_total_data,
            three_total_labels,
            three_total_annotations,
            three_total_vals["name"],
            three_total_vals["reference"][0],
            widths,
        )
        plot_component_violin(
            total_ax,
            total_data,
            total_labels,
            total_annotations,
            total_vals["name"],
            total_vals["reference"][0],
            widths,
            # ylim=[-25, 45],
        )
        # plt add space at bottom of figure
        plt.savefig(f"plots/{pfn}.png", transparent=transparent, bbox_inches="tight")
        plt.clf()
    else:
        plt.savefig(
            f"plots/{pfn}_ONLY.png", transparent=transparent, bbox_inches="tight"
        )
        plt.clf()
        fig, axs = plt.subplots(1, 1, figsize=(12, 4), dpi=600, squeeze=True)
        plot_component_violin(
            axs,
            total_data,
            total_labels,
            total_annotations,
            total_vals["name"],
            total_vals["reference"][0],
            widths,
            sub_rotation=20,
            fontsize=22,
            # ylim=[-25, 45],
        )
        plt.savefig(
            f"plots/{pfn}_TOTAL.png", transparent=transparent, bbox_inches="tight"
        )
        # plot_component_violin_zoomed(
        #     df,
        #     total_vals['vals'],
        #     title_name="Levels of SAPT (4569)",# f"All Dimers (8299)",
        #     f"los_SAPTDFT_components_TOTAL",
        #     ylim=[-5, 5],
        #     legend_loc="upper right",
        #     transparent=True,
        #     # figure_size=(6, 6),
        # )

        plot_component_violin_zoomed(
            total_data,
            total_labels,
            total_annotations,
            total_vals["name"],
            # total_vals["reference"][0],
            widths=widths,
            sub_rotation=20,
            fontsize=28,
            figure_size=(5, 4),
            ylim=[-4, 6],
        )
        # plt.savefig(
        #     f"plots/{pfn}_TOTAL.png", transparent=transparent, bbox_inches="tight"
        # )
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
    outlier_cutoff=5,
    bottom=0.3,
    transparent=True,
    dpi=800,
    pdf=False,
    jpeg=True,
    verbose=False,
    ylim=None,
) -> None:
    print(f"Plotting {pfn}")
    image_ext = "png"
    if jpeg:
        image_ext = "jpeg"
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData, vDataErrors = [], [], []
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
            if verbose:
                for n, r in df3.iterrows():
                    print(
                        f"\n{r['DB']}, {r['System']}\nid: {r['id']}, error: {c2}={r[c2]:.2f}, {c1}={r[c1]:.2f}"
                    )
                    tools.print_cartesians(r["Geometry"])
        else:
            vDataErrors.append([])
        d_str = d.replace(" ", "")
        vLabels.append(rf"\textbf{{{d_str}-{l1}}}")
        vLabels.append(rf"\textbf{{{d_str}-{l2}}}")

    fig = plt.figure(dpi=dpi)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
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

    xs, ys = [], []
    for n, y in enumerate(vDataErrors):
        if len(y) > 0:
            xs.extend([n + 1 for _ in range(len(y))])
            ys.extend(y)
    ax.scatter(
        xs,
        ys,
        color="orange",
        s=8.0,
        label=r"Errors Beyond $\pm$5 $\mathrm{kcal\cdot mol^{-1}}$ ",
    )

    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for _ in range(len(xs_error))],
        "k--",
        label=r"$\pm$1 $\mathrm{kcal\cdot mol^{-1}}$",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [0 for _ in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        # label=r"0 $kcal\cdot mol^{-1}$",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for _ in range(len(xs_error))],
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

    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="9")
    ax.set_xlim((0, len(vLabels)))
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="lower left", fontsize="9")
    ax.set_xlabel("Database", fontsize="12")
    # ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
    # ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k", fontsize="14")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    # ax.grid(color="gray", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):
        if n % 2 != 0:
            xtick.set_color("blue")
        else:
            xtick.set_color("red")

    # plt.minorticks_on()
    # plt.minorticks_on()
    ax.tick_params(axis="y", which="minor")
    if title_name is not None:
        plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)
    if pdf:
        fn_pdf = f"plots/{pfn}_dbs_violin.pdf"
        fn_png = f"plots/{pfn}_dbs_violin.png"
        plt.savefig(
            fn_pdf,
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
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
            f"plots/{pfn}_dbs_violin.{image_ext}",
            transparent=transparent,
            bbox_inches="tight",
            dpi=dpi,
        )
    plt.clf()
    return
