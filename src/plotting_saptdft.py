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
from . import dftd3
import qcelemental as qcel
from pprint import pprint as pp

h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# COLORS
BLUE = "#1F77B4"  # BLUE
TEAL = "#17BECF"  # TEAL
LIGHT_PURPLE = "#7F7FFF"  # LIGHT BLUE
GREEN = "#2CA02C"  # GREEN
LIME_GREEN = "#8CA83C"  # LIME GREEN
Aquamarine = "#7FFFD4"  # Aquamarine
Slate_Blue = "#6A5ACD"  # Slate Blue
Medium_Sea_Green = "#3CB371"  # Medium Sea Green
INDIGO = "#7057FF"  # INDIGO
PURPLE = "#643B9F"
GREY = "#808080"

colors_components_saptdftd4 = []
colors_total_saptdftd4 = [
    [
        # TEAL,
        # LIGHT_BLUE,
        # Medium_Sea_Green,
        # INDIGO,
        TEAL,
        LIGHT_PURPLE,
        # Medium_Sea_Green,
        # INDIGO,
        # TEAL,
        # LIGHT_PURPLE,
        # Medium_Sea_Green,
        # INDIGO,
        TEAL,
        LIGHT_PURPLE,
        # Medium_Sea_Green,
        # INDIGO,
        TEAL,
        LIGHT_PURPLE,
        TEAL,
        TEAL,
        LIGHT_PURPLE,
        TEAL,
        TEAL,
        LIGHT_PURPLE,
        PURPLE,
        PURPLE,
        BLUE,
        GREEN,
        BLUE,
    ]
    for i in range(5)
]

colors_comps_saptdftd4 = [
    [
        TEAL,
        LIGHT_PURPLE,
        # Medium_Sea_Green,
        PURPLE,
        BLUE,
    ],
    [
        TEAL,
        LIGHT_PURPLE,
        # Medium_Sea_Green,
        PURPLE,
        GREEN,
    ],
    [
        TEAL,
        LIGHT_PURPLE,
        # Medium_Sea_Green,
        PURPLE,
        GREEN,
        BLUE,
    ],
]

colors_disp_saptdftd4 = [
    [
        TEAL,
        LIGHT_PURPLE,
        # Medium_Sea_Green,
        TEAL,
        LIGHT_PURPLE,
        TEAL,
        LIGHT_PURPLE,
        TEAL,
        TEAL,
        LIGHT_PURPLE,
        TEAL,
        LIGHT_PURPLE,
        Medium_Sea_Green,
        # BLUE,
        PURPLE,
        PURPLE,
        GREEN,
        BLUE,
        GREY,
    ]
]

# plt.rcParams["text.usetex"] = True
import matplotlib.font_manager as fm

font_path = "/usr/share/texlive/texmf-dist/fonts/tfm/adobe/helvetic/phvr7t.tfm"
my_font = fm.FontProperties(fname=font_path)
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
    disp_compute=locald4.compute_disp_2B_BJ_ATM_CHG_dimer,
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
        print(f"{i[2]} {params_2B = }")
        df[f"-D4 ({i[1]})"] = df.apply(
            lambda row: disp_compute(
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


def plot_basis_sets_d4_TT(df, build_df=False, df_out: str = "basis_study", df_name=""):
    selected = df_out
    df_out = f"plots/{df_out}.pkl"
    if build_df:
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT0_dz_IE",
                    "SAPT0_dz_3_IE_TT",
                    "SAPT0_dz_3_IE_2B_TT",
                    "SAPT0_dz_3_IE",
                ],
                [
                    "SAPT0_jdz_IE",
                    "SAPT0_jdz_3_IE_TT",
                    "SAPT0_jdz_3_IE_2B_TT",
                    "SAPT0_jdz_3_IE",
                ],
                [
                    "SAPT0_adz_IE",
                    "SAPT0_adz_3_IE_TT",
                    "SAPT0_adz_3_IE_2B_TT",
                    "SAPT0_adz_3_IE",
                ],
                [
                    "SAPT0_tz_IE",
                    "SAPT0_tz_3_IE_TT",
                    "SAPT0_tz_3_IE_2B_TT",
                    "SAPT0_tz_3_IE",
                ],
                [
                    "SAPT0_mtz_IE",
                    "SAPT0_mtz_3_IE_TT",
                    "SAPT0_mtz_3_IE_2B_TT",
                    "SAPT0_mtz_3_IE",
                ],
                [
                    "SAPT0_jtz_IE",
                    "SAPT0_jtz_3_IE_TT",
                    "SAPT0_jtz_3_IE_2B_TT",
                    "SAPT0_jtz_3_IE",
                ],
                [
                    "SAPT0_atz_IE",
                    "SAPT0_atz_3_IE_TT",
                    "SAPT0_atz_3_IE_2B_TT",
                    "SAPT0_atz_3_IE",
                ],
            ],
            disp_compute=locald4.compute_disp_2B_TT_ATM_TT_dimer,
        )
        df.to_pickle(df_out)
    else:
        df = pd.read_pickle(df_out)
    plot_violin_d3_d4_ALL_zoomed_min_max(
        df,
        {
            "0-D4(BJ)/DZ": "SAPT0_dz_3_IE_d4_diff",
            "0-D4(TT)/DZ": "SAPT0_dz_3_IE_TT_d4_diff",
            "0-D4(BJ)/jDZ": "SAPT0_jdz_3_IE_d4_diff",
            "0-D4(TT)/jDZ": "SAPT0_jdz_3_IE_TT_d4_diff",
            "0-D4(BJ)/aDZ": "SAPT0_adz_3_IE_d4_diff",
            "0-D4(TT)/aDZ": "SAPT0_adz_3_IE_TT_d4_diff",
            "0-D4(BJ)/TZ": "SAPT0_tz_3_IE_d4_diff",
            "0-D4(TT)/TZ": "SAPT0_tz_3_IE_TT_d4_diff",
            "0-D4(BJ)/mTZ": "SAPT0_mtz_3_IE_d4_diff",
            "0-D4(TT)/mTZ": "SAPT0_mtz_3_IE_TT_d4_diff",
            "0-D4(BJ)/jTZ": "SAPT0_jtz_3_IE_d4_diff",
            "0-D4(TT)/jTZ": "SAPT0_jtz_3_IE_TT_d4_diff",
            "0-D4(BJ)/aTZ": "SAPT0_atz_3_IE_d4_diff",
            "0-D4(TT)/aTZ": "SAPT0_atz_3_IE_TT_d4_diff",
        },
        "",  # f"All Dimers (8299)",
        f"{selected}_d4_zoomed_TT",
        bottom=0.45,
        ylim=[-5, 5],
        legend_loc="upper right",
        transparent=True,
        # figure_size=(6, 6),
    )
    return


def plot_basis_sets_d4_Inter_vs_Super(
    df, build_df=False, df_out: str = "basis_study", df_name=""
):
    selected = df_out
    df_out = f"plots/{df_out}.pkl"
    if build_df:
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT0_dz_IE",
                    "SAPT0_dz_3_IE_BJ_inter",
                    "SAPT0_dz_3_IE_2B_BJ_inter",
                    "SAPT0_dz_3_IE",
                ],
                [
                    "SAPT0_jdz_IE",
                    "SAPT0_jdz_3_IE_BJ_inter",
                    "SAPT0_jdz_3_IE_2B_BJ_inter",
                    "SAPT0_jdz_3_IE",
                ],
                [
                    "SAPT0_adz_IE",
                    "SAPT0_adz_3_IE_BJ_inter",
                    "SAPT0_adz_3_IE_2B_BJ_inter",
                    "SAPT0_adz_3_IE",
                ],
                [
                    "SAPT0_tz_IE",
                    "SAPT0_tz_3_IE_BJ_inter",
                    "SAPT0_tz_3_IE_2B_BJ_inter",
                    "SAPT0_tz_3_IE",
                ],
                [
                    "SAPT0_mtz_IE",
                    "SAPT0_mtz_3_IE_BJ_inter",
                    "SAPT0_mtz_3_IE_2B_BJ_inter",
                    "SAPT0_mtz_3_IE",
                ],
                [
                    "SAPT0_jtz_IE",
                    "SAPT0_jtz_3_IE_BJ_inter",
                    "SAPT0_jtz_3_IE_2B_BJ_inter",
                    "SAPT0_jtz_3_IE",
                ],
                [
                    "SAPT0_atz_IE",
                    "SAPT0_atz_3_IE_BJ_inter",
                    "SAPT0_atz_3_IE_2B_BJ_inter",
                    "SAPT0_atz_3_IE",
                ],
            ],
            disp_compute=locald4.compute_disp_2B_BJ_dimer_inter,
        )
        df.to_pickle(df_out)
    else:
        df = pd.read_pickle(df_out)
    plot_violin_d3_d4_ALL_zoomed_min_max(
        df,
        {
            "0-D4(BJ Super)/DZ": "SAPT0_dz_3_IE_d4_diff",
            "0-D4(BJ Intermol)/DZ": "SAPT0_dz_3_IE_BJ_inter_d4_diff",
            "0-D4(BJ Super)/jDZ": "SAPT0_jdz_3_IE_d4_diff",
            "0-D4(BJ Intermol)/jDZ": "SAPT0_jdz_3_IE_BJ_inter_d4_diff",
            "0-D4(BJ Super)/aDZ": "SAPT0_adz_3_IE_d4_diff",
            "0-D4(BJ Intermol)/aDZ": "SAPT0_adz_3_IE_BJ_inter_d4_diff",
            "0-D4(BJ Super)/TZ": "SAPT0_tz_3_IE_d4_diff",
            "0-D4(BJ Intermol)/TZ": "SAPT0_tz_3_IE_BJ_inter_d4_diff",
            "0-D4(BJ Super)/mTZ": "SAPT0_mtz_3_IE_d4_diff",
            "0-D4(BJ Intermol)/mTZ": "SAPT0_mtz_3_IE_BJ_inter_d4_diff",
            "0-D4(BJ Super)/jTZ": "SAPT0_jtz_3_IE_d4_diff",
            "0-D4(BJ Intermol)/jTZ": "SAPT0_jtz_3_IE_BJ_inter_d4_diff",
            "0-D4(BJ Super)/aTZ": "SAPT0_atz_3_IE_d4_diff",
            "0-D4(BJ Intermol)/aTZ": "SAPT0_atz_3_IE_BJ_inter_d4_diff",
        },
        "",  # f"All Dimers (8299)",
        f"{selected}_d4_zoomed_Inter_vs_Super",
        bottom=0.45,
        ylim=[-3, 3],
        legend_loc="upper right",
        transparent=True,
        # figure_size=(6, 6),
    )
    return


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
        params_d3 = params_dict[i[2]]
        if len(params_d3) == 2:
            params_d3 = params_dict[i[2]][0][1:4]
        else:
            params_d3, _ = paramsTable.generate_2B_ATM_param_subsets(params_d3)
        # Need to write function for computing D3Data for LoS dataset...
        print(params_d3)
        df[f"-D3 ({i[1]})"] = df.apply(
            lambda row: (
                jeff.compute_BJ_CPP(
                    params_d3,
                    row["D3Data"],
                )
                if row["D3Data"] is not None
                else np.nan
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
        plot_violin_d3_d4_ALL_zoomed_min_max_TOC(
            df,
            {
                # "SAPT0/jDZ": "SAPT0_jdz_3_IE_diff",
                "SAPT0/aDZ": "SAPT0_adz_3_IE_diff",
                # "SAPT0-D3/jDZ": "SAPT0_jdz_3_IE_d3_diff",
                # SAPT "0-D3MBJ(ATM)/jDZ": "jdz_diff_d3mbj_atm",
                "SAPT0-D3/aDZ": "SAPT0_adz_3_IE_d3_diff",
                # SAPT "0-D3MBJ(ATM)/aDZ": "adz_diff_d3mbj_atm",
                "SAPT0-D4/aDZ": "SAPT0_adz_3_IE_d4_diff",
                # "0-D4(ATM)/aDZ": "SAPT0_dz_3_IE_ATM_SHARED_d4_diff",
                # "0-D4(ATM)/aDZ": "SAPT0_adz_ATM_opt_all_diff",
                # "0-D4(ATMu)/aDZ": "adz_diff_d4_ATM",  # (2B ATM) renamed to (ATMu)
                # "0-D4(2B@G ATM)/aDZ": "adz_diff_d4_2B@ATM_G",
                # "0-D4(2B@G ATM@G)/aDZ": "adz_diff_d4_ATM_G",
                # "0-D4(ATM TT ALL)/aDZ": "SAPT0_adz_3_IE_TT_ALL_d4_diff",
                "SAPT(DFT)/aDZ": "SAPT_DFT_adz_3_IE_diff",
                # "SAPT(DFT)-D4/aDZ": "SAPT_DFT_adz_3_IE_d4_diff",
                "SAPT(DFT)/aTZ": "SAPT_DFT_atz_3_IE_diff",
            },
            "",  # f"All Dimers (8299)",
            # f"8299 Dimer Dataset",
            f"{selected}_ATM_TOC",
            bottom=0.45,
            ylim=[-3, 3],
            legend_loc="upper right",
            transparent=True,
            figure_size=(6, 2.0),
            jpeg=False,
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
                lambda x: (
                    x[f"SAPT_DFT_{basis}"][1]
                    + x[f"SAPT_DFT_{basis}"][2]
                    + x[f"SAPT_DFT_{basis}"][3]
                ),
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
                lambda x: (
                    x[f"SAPT0_{basis}_elst"]
                    + x[f"SAPT0_{basis}_exch"]
                    + x[f"SAPT0_{basis}_indu"]
                ),
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
                lambda x: (
                    x[f"SAPT_DFT_{basis}_elst"]
                    + x[f"SAPT_DFT_{basis}_exch"]
                    + x[f"SAPT_DFT_{basis}_indu"]
                ),
                axis=1,
            )
            df[f"SAPT_DFT_{basis}_3_IE_d4"] = df.apply(
                lambda x: (
                    x[f"SAPT_DFT_{basis}_3_IE"] + x[f"-D4 (SAPT_DFT_{basis}_3_IE)"]
                ),
                axis=1,
            )
            df[f"SAPT_DFT_{basis}_3_IE_d4_ATM"] = df.apply(
                lambda x: (
                    x[f"SAPT_DFT_{basis}_3_IE"] + x[f"-D4 (SAPT_DFT_{basis}_3_IE_ATM)"]
                ),
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


def compute_saptdft_ddft_ie(r, b, functional="pbe0"):
    if r[f"SAPT_DFT_{functional}_{b}"]:
        return (
            r[f"SAPT_DFT_{functional}_{b}"][1]
            + r[f"SAPT_DFT_{functional}_{b}"][2]
            + r[f"SAPT_DFT_{functional}_{b}"][3]
            + r[f"SAPT_DFT_{functional}_{b}_D4_IE"]
            + r[f"SAPT_DFT_{functional}_{b}_dDFT"]
            - r[f"SAPT_DFT_{functional}_{b}_dHF"]
        )
    else:
        return np.nan


def prepare_saptdft_columns(df, functional, basis_set):
    if f"SAPT_DFT_{functional}_{basis_set}" not in df.columns:
        print(f"Zeroing SAPT_DFT_{functional}_{basis_set} because not found...")
        df[f"SAPT_DFT_{functional}_{basis_set}"] = df.apply(
            lambda x: [0, 0, 0, 0, 0], axis=1
        )
        df[f"SAPT_DFT_{functional}_{basis_set}_dDFT"] = df.apply(lambda x: 0, axis=1)
        df[f"SAPT_DFT_{functional}_{basis_set}_dHF"] = df.apply(lambda x: 0, axis=1)
        df[f"SAPT_DFT_{functional}_{basis_set}_D4_IE"] = df.apply(lambda x: 0, axis=1)
        df[f"SAPT_DFT_{functional}_{basis_set}_DFT_IE"] = df.apply(lambda x: 0, axis=1)
    # else:
    #     print(df[f"SAPT_DFT_{functional}_{basis_set}"])

    df[f"SAPT_DFT_D4_{functional}_{basis_set}_total"] = df.apply(
        lambda x: compute_saptdft_ddft_ie(x, f"{basis_set}", functional=functional),
        axis=1,
    )
    df[f"DFT-D4/{basis_set}"] = df.apply(
        lambda x: (
            x[f"SAPT_DFT_{functional}_{basis_set}_DFT_IE"]
            + x[f"SAPT_DFT_{functional}_{basis_set}_D4_IE"]
            if x[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            else np.nan
        ),
        axis=1,
    )
    df[f"SAPT_DFT_{functional}_{basis_set}_total"] = df[
        f"SAPT_DFT_{functional}_{basis_set}"
    ].apply(
        lambda x: x[0] if isinstance(x, np.ndarray) or isinstance(x, list) else np.nan
    )
    df[f"SAPT_DFT_{functional}_{basis_set}_elst"] = df[
        f"SAPT_DFT_{functional}_{basis_set}"
    ].apply(
        lambda x: x[1] if isinstance(x, np.ndarray) or isinstance(x, list) else np.nan
    )
    df[f"SAPT_DFT_{functional}_{basis_set}_exch"] = df[
        f"SAPT_DFT_{functional}_{basis_set}"
    ].apply(
        lambda x: x[2] if isinstance(x, np.ndarray) or isinstance(x, list) else np.nan
    )
    df[f"SAPT_DFT_{functional}_{basis_set}_indu"] = df[
        f"SAPT_DFT_{functional}_{basis_set}"
    ].apply(
        lambda x: x[3] if isinstance(x, np.ndarray) or isinstance(x, list) else np.nan
    )
    df[f"SAPT_DFT_{functional}_{basis_set}_disp"] = df[
        f"SAPT_DFT_{functional}_{basis_set}"
    ].apply(
        lambda x: x[4] if isinstance(x, np.ndarray) or isinstance(x, list) else np.nan
    )

    df[f"SAPT_DFT_{functional}_{basis_set}_3_IE"] = (
        df[f"SAPT_DFT_{functional}_{basis_set}_elst"]
        + df[f"SAPT_DFT_{functional}_{basis_set}_exch"]
        + df[f"SAPT_DFT_{functional}_{basis_set}_indu"]
    )
    df[f"SAPT_DFT_{functional}_{basis_set}_3_IE_pre_d4"] = (
        df[f"SAPT_DFT_{functional}_{basis_set}_elst"]
        + df[f"SAPT_DFT_{functional}_{basis_set}_exch"]
        + df[f"SAPT_DFT_{functional}_{basis_set}_indu"]
        + df[f"SAPT_DFT_{functional}_{basis_set}_dDFT"]
        - df[f"SAPT_DFT_{functional}_{basis_set}_dHF"]
    )
    df[f"SAPT_DFT_{functional}_{basis_set}_d4_disp"] = df.apply(
        lambda x: (
            x[f"SAPT_DFT_{functional}_{basis_set}_dDFT"]
            - x[f"SAPT_DFT_{functional}_{basis_set}_dHF"]
            + x[f"SAPT_DFT_{functional}_{basis_set}_D4_IE"]
        ),
        axis=1,
    )
    df[f"{functional.upper()} IE {basis_set}"] = df.apply(
        lambda r: (
            r[f"SAPT_DFT_{functional}_{basis_set}_dDFT"]
            + r[f"SAPT_DFT_{functional}_{basis_set}_elst"]
            + r[f"SAPT_DFT_{functional}_{basis_set}_exch"]
            + r[f"SAPT_DFT_{functional}_{basis_set}_indu"]
            - r[f"SAPT_DFT_{functional}_{basis_set}_dHF"]
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            else np.nan
        ),
        axis=1,
    )
    df[f"{functional.upper()}-D4 IE {basis_set}"] = df.apply(
        lambda r: (
            r[f"{functional.upper()} IE {basis_set}"]
            + r[f"SAPT_DFT_{functional}_{basis_set}_D4_IE"]
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            else np.nan
        ),
        axis=1,
    )
    return df


def plotting_setup_dft_ddft(
    selected,
    build_df=False,
    df_out: str = "plots/ddft_study.pkl",
    split_components=False,
    original_plot=False,
):
    reference = "SAPT2+3(CCD)DMP2"
    ref_basis = "aTZ"
    if build_df:
        df = pd.read_pickle(selected)
        len_before = len(df)
        df_c6s_dispml = pd.read_pickle("./plots/ddft_study.pkl")
        df = df.merge(df_c6s_dispml, on="system_id", how="left", suffixes=("", "_drop"))
        df.drop([c for c in df.columns if "drop" in c], axis=1, inplace=True)
        df.dropna(
            subset=["SAPT_DFT_pbe0_adz", "SAPT_DFT_pbe0_atz", "C6s"], inplace=True
        )
        print(f"Removed {len_before - len(df)} NaN values")

        df = df[df["SAPT_DFT_pbe0_adz"].notna()].copy()

        # df_disp = pd.read_pickle("./dfs/dispml.pkl")
        # df_disp = df_disp[["D3-ML", "system_id"]].copy()
        # df = df.merge(df_disp, on="system_id")

        # df_d3data = pd.read_pickle("./dfs/los_d3data.pkl")
        # df_d3data = df_d3data[["D3Data", "system_id"]].copy()
        # df = df.merge(df_d3data, on="system_id")
        basis_set = "adz"
        functional = "pbe0"
        df = prepare_saptdft_columns(df, "pbe0", "adz")

        df.dropna(subset=[f"SAPT_DFT_D4_{functional}_{basis_set}_total"], inplace=True)

        df = prepare_saptdft_columns(df, "pbe0", "atz")
        df = prepare_saptdft_columns(df, "pbe0", "aqz")

        df = prepare_saptdft_columns(df, "b3lyp", "adz")
        df = prepare_saptdft_columns(df, "b3lyp", "atz")
        df = prepare_saptdft_columns(df, "b3lyp", "aqz")

        df = prepare_saptdft_columns(df, "b2plyp", "adz")
        df = prepare_saptdft_columns(df, "b2plyp", "atz")
        df = prepare_saptdft_columns(df, "b2plyp", "aqz")

        df = prepare_saptdft_columns(df, "wb97x", "adz")
        df = prepare_saptdft_columns(df, "wb97x", "atz")
        df = prepare_saptdft_columns(df, "wb97x", "aqz")

        df[f"SAPT0_{basis_set}"] = df.apply(
            lambda x: (
                np.array(
                    [
                        x[f"SAPT0 ELST ENERGY {basis_set}"]
                        + x[f"SAPT0 EXCH ENERGY {basis_set}"]
                        + x[f"SAPT0 IND ENERGY {basis_set}"]
                        + x[f"SAPT0 DISP ENERGY {basis_set}"],
                        x[f"SAPT0 ELST ENERGY {basis_set}"],
                        x[f"SAPT0 EXCH ENERGY {basis_set}"],
                        x[f"SAPT0 IND ENERGY {basis_set}"],
                        x[f"SAPT0 DISP ENERGY {basis_set}"],
                    ]
                )
                * h2kcalmol
            ),
            axis=1,
        )
        df[f"SAPT0_{basis_set}_total"] = df[f"SAPT0_{basis_set}"].apply(lambda x: x[0])
        df[f"SAPT0_{basis_set}_elst"] = df[f"SAPT0_{basis_set}"].apply(lambda x: x[1])
        df[f"SAPT0_{basis_set}_exch"] = df[f"SAPT0_{basis_set}"].apply(lambda x: x[2])
        df[f"SAPT0_{basis_set}_indu"] = df[f"SAPT0_{basis_set}"].apply(lambda x: x[3])
        df[f"SAPT0_{basis_set}_disp"] = df[f"SAPT0_{basis_set}"].apply(lambda x: x[4])
        df[f"SAPT0_{basis_set}_3_IE"] = df.apply(
            lambda x: (
                (
                    x[f"SAPT0 ELST ENERGY {basis_set}"]
                    + x[f"SAPT0 EXCH ENERGY {basis_set}"]
                    + x[f"SAPT0 IND ENERGY {basis_set}"]
                )
                * h2kcalmol
            ),
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
        # df[f"{reference} TOTAL ENERGY adz"] = df[f"{reference} TOTAL ENERGY adz"] * h2kcalmol
        df["SAPT0_atz"] = df.apply(
            lambda x: (
                np.array(
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
                )
                * h2kcalmol
            ),
            axis=1,
        )
        df["SAPT0_atz_3_IE"] = df.apply(
            lambda x: (
                (
                    x["SAPT0 ELST ENERGY atz"]
                    + x["SAPT0 EXCH ENERGY atz"]
                    + x["SAPT0 IND ENERGY atz"]
                )
                * h2kcalmol
            ),
            axis=1,
        )
        df["SAPT0_atz_total"] = df["SAPT0_atz"].apply(lambda x: x[0])
        df["SAPT0_atz_elst"] = df["SAPT0_atz"].apply(lambda x: x[1])
        df["SAPT0_atz_exch"] = df["SAPT0_atz"].apply(lambda x: x[2])
        df["SAPT0_atz_indu"] = df["SAPT0_atz"].apply(lambda x: x[3])
        df["SAPT0_atz_disp"] = df["SAPT0_atz"].apply(lambda x: x[4])

        # SAPT(DFT) - aQZ
        # df["SAPT_DFT_D4_pbe0_aqz_total"] = df.apply(
        #     lambda x: compute_saptdft_ddft_ie(x, "aqz"),
        #     axis=1,
        # )
        #
        # df["DFT-D4/aqz"] = df.apply(
        #     lambda x: x["SAPT_DFT_pbe0_aqz_DFT_IE"] + x["SAPT_DFT_pbe0_aqz_D4_IE"],
        #     axis=1,
        # )
        df["MP2 IE aqz"] = df.apply(
            lambda r: (
                r["SAPT MP2(2) ENERGY aqz"] + r["SAPT2 TOTAL ENERGY aqz"]
                if r["SAPT MP2(2) ENERGY aqz"]
                else np.nan
            ),
            axis=1,
        )

        # print(df[["SAPT_DFT_D4_pbe0_aqz_total", "DFT-D4/aqz"]])
        # df["SAPT_DFT_pbe0_aqz_total"] = df["SAPT_DFT_pbe0_aqz"].apply(
        #     lambda x: x[0] if x else np.nan
        # )
        # df["SAPT_DFT_pbe0_aqz_elst"] = df["SAPT_DFT_pbe0_aqz"].apply(
        #     lambda x: x[1] if x else np.nan
        # )
        # df["SAPT_DFT_pbe0_aqz_exch"] = df["SAPT_DFT_pbe0_aqz"].apply(
        #     lambda x: x[2] if x else np.nan
        # )
        # df["SAPT_DFT_pbe0_aqz_indu"] = df["SAPT_DFT_pbe0_aqz"].apply(
        #     lambda x: x[3] if x else np.nan
        # )
        # df["SAPT_DFT_pbe0_aqz_disp"] = df["SAPT_DFT_pbe0_aqz"].apply(
        #     lambda x: x[4] if x else np.nan
        # )
        # df["SAPT_DFT_pbe0_aqz_3_IE"] = (
        #     df["SAPT_DFT_pbe0_aqz_elst"]
        #     + df["SAPT_DFT_pbe0_aqz_exch"]
        #     + df["SAPT_DFT_pbe0_aqz_indu"]
        # )
        # df["SAPT_DFT_pbe0_aqz_3_IE_pre_d4"] = (
        #     df["SAPT_DFT_pbe0_aqz_elst"]
        #     + df["SAPT_DFT_pbe0_aqz_exch"]
        #     + df["SAPT_DFT_pbe0_aqz_indu"]
        #     + df["SAPT_DFT_pbe0_aqz_dDFT"]
        #     - df["SAPT_DFT_pbe0_aqz_dHF"]
        # )
        # df["SAPT_DFT_pbe0_aqz_d4_disp"] = df.apply(
        #     lambda x: x["SAPT_DFT_pbe0_aqz_dDFT"]
        #     - x["SAPT_DFT_pbe0_aqz_dHF"]
        #     + x["SAPT_DFT_pbe0_aqz_D4_IE"],
        #     axis=1,
        # )
        # df[f"{reference} TOTAL ENERGY adz"] = df[f"{reference} TOTAL ENERGY adz"] * h2kcalmol
        df["SAPT0_aqz"] = df.apply(
            lambda x: (
                np.array(
                    [
                        x["SAPT0 ELST ENERGY aqz"]
                        + x["SAPT0 EXCH ENERGY aqz"]
                        + x["SAPT0 IND ENERGY aqz"]
                        + x["SAPT0 DISP ENERGY aqz"],
                        x["SAPT0 ELST ENERGY aqz"],
                        x["SAPT0 EXCH ENERGY aqz"],
                        x["SAPT0 IND ENERGY aqz"],
                        x["SAPT0 DISP ENERGY aqz"],
                    ]
                )
                * h2kcalmol
            ),
            axis=1,
        )
        df["SAPT0_aqz_3_IE"] = df.apply(
            lambda x: (
                (
                    x["SAPT0 ELST ENERGY aqz"]
                    + x["SAPT0 EXCH ENERGY aqz"]
                    + x["SAPT0 IND ENERGY aqz"]
                )
                * h2kcalmol
            ),
            axis=1,
        )
        df["SAPT0_aqz_total"] = df["SAPT0_aqz"].apply(lambda x: x[0])
        df["SAPT0_aqz_elst"] = df["SAPT0_aqz"].apply(lambda x: x[1])
        df["SAPT0_aqz_exch"] = df["SAPT0_aqz"].apply(lambda x: x[2])
        df["SAPT0_aqz_indu"] = df["SAPT0_aqz"].apply(lambda x: x[3])
        df["SAPT0_aqz_disp"] = df["SAPT0_aqz"].apply(lambda x: x[4])

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
                [
                    "SAPT0_aqz_total",
                    "SAPT0_aqz_3_IE",
                    "SAPT0_adz_3_IE_2B",
                    "SAPT0_aqz_3_IE",
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
        df[f"SAPT0_aqz_d4"] = df.apply(
            lambda x: x[f"SAPT0_aqz_3_IE"] + x[f"-D4 (SAPT0_aqz_3_IE)"], axis=1
        )
        df.dropna(subset=["-D4 (SAPT0_adz_3_IE)"], inplace=True)
        df.dropna(subset=["-D4 (SAPT0_atz_3_IE)"], inplace=True)
        df["SAPT0-D4/aDZ"] = df.apply(
            lambda row: row["SAPT0_adz_3_IE"] + row["-D4 (SAPT0_adz_3_IE)"], axis=1
        )

        # Dispersion Term fittings...
        df["SAPT_DFT_pbe0_adz_DIFF_SAPT2+3(CCD)DMP2"] = df.apply(
            lambda r: (
                -(
                    r["SAPT_DFT_pbe0_adz"][0]
                    - r[f"SAPT2+3(CCD)DMP2 TOTAL ENERGY atz"] * h2kcalmol
                )
            ),
            axis=1,
        )

        df["SAPT_DFT_pbe0_atz_DIFF_SAPT2+3(CCD)DMP2"] = df.apply(
            lambda r: (
                -(
                    r["SAPT_DFT_pbe0_atz"][0]
                    - r[f"SAPT2+3(CCD)DMP2 TOTAL ENERGY atz"] * h2kcalmol
                )
            ),
            axis=1,
        )

        # DISP-ML Section

        df["SAPT(DFT)D3-ML TOTAL ENERGY adz"] = df.apply(
            lambda r: (
                sum(r["SAPT_DFT_pbe0_adz"][1:4]) / h2kcalmol + r["D3-ML"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)D3-ML TOTAL ENERGY atz"] = df.apply(
            lambda r: (
                sum(r["SAPT_DFT_pbe0_atz"][1:4]) / h2kcalmol + r["D3-ML"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)D3-ML TOTAL ENERGY aqz"] = df.apply(
            lambda r: (
                sum(r["SAPT_DFT_pbe0_aqz"][1:4]) / h2kcalmol + r["D3-ML"] / h2kcalmol
                if r["SAPT_DFT_pbe0_aqz"]
                else np.nan
            ),
            axis=1,
        )
        df["D3-ML DISP ENERGY adz"] = df.apply(lambda r: r["D3-ML"] / h2kcalmol, axis=1)
        df["D3-ML DISP ENERGY atz"] = df.apply(lambda r: r["D3-ML"] / h2kcalmol, axis=1)
        df["D3-ML DISP ENERGY aqz"] = df.apply(lambda r: r["D3-ML"] / h2kcalmol, axis=1)

        # SAPT(DFT)+D4 - correcting SAPT(DFT) dispersion error
        df = compute_d4_from_opt_params(
            df,
            bases=[
                # [
                #     "SAPT_DFT_b3lyp_adz_total",
                #     "SAPT_DFT_b3lyp_adz_3_IE",
                #     "b3lyp",
                #     "SAPT_DFT_b3lyp_adz_3_IE",
                # ],
                # [
                #     "SAPT_DFT_b2plyp_adz_total",
                #     "SAPT_DFT_b2plyp_adz_3_IE",
                #     "b2plyp",
                #     "SAPT_DFT_b2plyp_adz_3_IE",
                # ],
                [
                    "SAPT_DFT_pbe0_adz_total",
                    "SAPT_DFT_pbe0_adz_3_IE",
                    "SAPT_DFT_pbe0_adz_3_IE",
                    # "pbe0",
                    "SAPT_DFT_pbe0_adz_3_IE",
                ],
                [
                    "SAPT_DFT_pbe0_atz_total",
                    "SAPT_DFT_pbe0_atz_3_IE",
                    "SAPT_DFT_pbe0_atz_3_IE",
                    # "pbe0",
                    "SAPT_DFT_pbe0_atz_3_IE",
                ],
                [
                    "SAPT_DFT_pbe0_aqz_total",
                    "SAPT_DFT_pbe0_aqz_3_IE",
                    "SAPT_DFT_pbe0_atz_3_IE",
                    # "pbe0",
                    "SAPT_DFT_pbe0_aqz_3_IE",
                ],
                [
                    "SAPT_DFT_pbe0_adz_total",
                    "SAPT_DFT_adz_plus_D4",
                    "SAPT_DFT_pbe0_adz_disp_targeting_SAPT2+3(CCD)dMP2",
                    "SAPT_DFT_pbe0_adz_total",
                ],
                [
                    "SAPT_DFT_pbe0_atz_total",
                    "SAPT_DFT_atz_plus_D4",
                    "SAPT_DFT_pbe0_atz_disp_targeting_SAPT2+3(CCD)dMP2",
                    "SAPT_DFT_pbe0_atz_total",
                ],
                [
                    "SAPT_DFT_pbe0_aqz_total",
                    "SAPT_DFT_aqz_plus_D4",
                    "SAPT_DFT_pbe0_atz_disp_targeting_SAPT2+3(CCD)dMP2",
                    "SAPT_DFT_pbe0_aqz_total",
                ],
            ],
            benchmark_label="benchmark ref energy",
        )
        df["SAPT(DFT)-D4 TOTAL ENERGY adz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_adz_3_IE"] / h2kcalmol
                + r["-D4 (SAPT_DFT_pbe0_adz_3_IE)"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)-D4 DISP ENERGY adz"] = df.apply(
            lambda r: r["-D4 (SAPT_DFT_pbe0_adz_3_IE)"] / h2kcalmol, axis=1
        )
        df["SAPT(DFT)-D4 TOTAL ENERGY atz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_atz_3_IE"] / h2kcalmol
                + r["-D4 (SAPT_DFT_pbe0_atz_3_IE)"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)-D4 DISP ENERGY atz"] = df.apply(
            lambda r: r["-D4 (SAPT_DFT_pbe0_atz_3_IE)"] / h2kcalmol, axis=1
        )
        df["SAPT(DFT)-D4 TOTAL ENERGY aqz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_aqz_3_IE"] / h2kcalmol
                + r["-D4 (SAPT_DFT_pbe0_aqz_3_IE)"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)-D4 DISP ENERGY aqz"] = df.apply(
            lambda r: r["-D4 (SAPT_DFT_pbe0_aqz_3_IE)"] / h2kcalmol, axis=1
        )

        df["SAPT(DFT)+D4 TOTAL ENERGY adz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_adz"][0] / h2kcalmol
                + r["-D4 (SAPT_DFT_adz_plus_D4)"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)+D4 DISP ENERGY adz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_adz"][-1] / h2kcalmol
                + r["-D4 (SAPT_DFT_adz_plus_D4)"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)+D4 TOTAL ENERGY atz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_atz"][0] / h2kcalmol
                + r["-D4 (SAPT_DFT_atz_plus_D4)"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)+D4 DISP ENERGY atz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_atz"][-1] / h2kcalmol
                + r["-D4 (SAPT_DFT_atz_plus_D4)"] / h2kcalmol
            ),
            axis=1,
        )
        df["SAPT(DFT)+D4 TOTAL ENERGY aqz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_aqz"][0] / h2kcalmol
                + r["-D4 (SAPT_DFT_aqz_plus_D4)"] / h2kcalmol
                if r["SAPT_DFT_pbe0_aqz"]
                else np.nan
            ),
            axis=1,
        )
        df["SAPT(DFT)+D4 DISP ENERGY aqz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_aqz"][-1] / h2kcalmol
                + r["-D4 (SAPT_DFT_aqz_plus_D4)"] / h2kcalmol
                if r["SAPT_DFT_pbe0_aqz"]
                else np.nan
            ),
            axis=1,
        )
        # DFT IEs
        df["PBE0 IE adz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_adz_dDFT"]
                + r["SAPT_DFT_pbe0_adz_elst"]
                + r["SAPT_DFT_pbe0_adz_exch"]
                + r["SAPT_DFT_pbe0_adz_indu"]
                - r["SAPT_DFT_pbe0_adz_dHF"]
                if r["SAPT_DFT_D4_pbe0_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["B3LYP IE adz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_b3lyp_adz_dDFT"]
                + r["SAPT_DFT_b3lyp_adz_elst"]
                + r["SAPT_DFT_b3lyp_adz_exch"]
                + r["SAPT_DFT_b3lyp_adz_indu"]
                - r["SAPT_DFT_b3lyp_adz_dHF"]
                if r["SAPT_DFT_D4_b3lyp_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["B2PLYP IE adz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_b2plyp_adz_dDFT"]
                + r["SAPT_DFT_b2plyp_adz_elst"]
                + r["SAPT_DFT_b2plyp_adz_exch"]
                + r["SAPT_DFT_b2plyp_adz_indu"]
                - r["SAPT_DFT_b2plyp_adz_dHF"]
                if r["SAPT_DFT_D4_b2plyp_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["WB97X IE adz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_wb97x_adz_dDFT"]
                + r["SAPT_DFT_wb97x_adz_elst"]
                + r["SAPT_DFT_wb97x_adz_exch"]
                + r["SAPT_DFT_wb97x_adz_indu"]
                - r["SAPT_DFT_wb97x_adz_dHF"]
                if r["SAPT_DFT_D4_wb97x_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["PBE0-D4 IE adz"] = df.apply(
            lambda r: (
                r["PBE0 IE adz"] + r["SAPT_DFT_pbe0_adz_D4_IE"]
                if r["SAPT_DFT_D4_pbe0_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["PBE0-D4 IE adz"] = df.apply(
            lambda r: (
                r["PBE0 IE adz"] + r["SAPT_DFT_pbe0_adz_D4_IE"]
                if r["SAPT_DFT_D4_pbe0_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["B2PLYP-D4 IE adz"] = df.apply(
            lambda r: (
                r["B2PLYP IE adz"] + r["SAPT_DFT_b2plyp_adz_D4_IE"]
                if r["SAPT_DFT_D4_b2plyp_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["B3LYP-D4 IE adz"] = df.apply(
            lambda r: (
                r["B3LYP IE adz"] + r["SAPT_DFT_b3lyp_adz_D4_IE"]
                if r["SAPT_DFT_D4_b3lyp_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        df["WB97X-D4 IE adz"] = df.apply(
            lambda r: (
                r["WB97X IE adz"] + r["SAPT_DFT_wb97x_adz_D4_IE"]
                if r["SAPT_DFT_D4_wb97x_adz_total"]
                else np.nan
            ),
            axis=1,
        )

        for n, r in df.iterrows():
            if np.abs(r["SAPT_DFT_D4_pbe0_adz_total"] - r["PBE0-D4 IE adz"]) > 1e-12:
                print(
                    n,
                    r["benchmark ref energy"],
                    r["SAPT_DFT_D4_pbe0_adz_total"],
                    r["PBE0-D4 IE adz"],
                )
        df_test = df.dropna(subset=["SAPT_DFT_D4_pbe0_adz_total"])
        df_test = df_test.dropna(subset=["PBE0-D4 IE adz"])
        print(df_test[["PBE0-D4 IE adz", "SAPT_DFT_D4_pbe0_adz_total"]])
        assert np.allclose(
            df_test["PBE0-D4 IE adz"], df_test["SAPT_DFT_D4_pbe0_adz_total"], atol=1e-16
        )
        df["PBE0 IE atz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_atz_dDFT"]
                + r["SAPT_DFT_pbe0_atz_elst"]
                + r["SAPT_DFT_pbe0_atz_exch"]
                + r["SAPT_DFT_pbe0_atz_indu"]
                - r["SAPT_DFT_pbe0_atz_dHF"]
                if r["SAPT_DFT_pbe0_atz_total"]
                else np.nan
            ),
            axis=1,
        )
        df["PBE0 IE aqz"] = df.apply(
            lambda r: (
                r["SAPT_DFT_pbe0_aqz_dDFT"]
                + r["SAPT_DFT_pbe0_aqz_elst"]
                + r["SAPT_DFT_pbe0_aqz_exch"]
                + r["SAPT_DFT_pbe0_aqz_indu"]
                - r["SAPT_DFT_pbe0_aqz_dHF"]
                if r["SAPT_DFT_pbe0_aqz_total"]
                else np.nan
            ),
            axis=1,
        )
        df.to_pickle(df_out)
    else:
        print(f"Loading {df_out}")
        df = pd.read_pickle(df_out)
        print(df)
    basename_selected = selected.split("/")[-1].split(".")[0]
    print(len(df), df["SAPT_DFT_pbe0_adz_elst"].isnull().sum())
    assert df["SAPT_DFT_pbe0_adz_elst"].isnull().sum() == 0
    print(f"Length of df prior plotting: {len(df)}")
    dimer_dataset_size = len(df)

    if True:
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT0_adz_total",
                    "SAPT0_adz_3_IE_2B_BJ_inter",
                    "SAPT0_adz_3_IE_2B_BJ_inter",
                    "SAPT0_adz_3_IE",
                ],
            ],
            benchmark_label="benchmark ref energy",
            disp_compute=locald4.compute_disp_2B_BJ_dimer_inter,
        )
        # -D3(I) and -D3(S) for SAPT(PBE0) and SAPT(B3LYP)
    if False:
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT_DFT_pbe0_adz_total",
                    "SAPT_DFT_pbe0_adz_3_IE_NO_DAMPING",
                    "SAPT_DFT_pbe0_adz_3_IE_inter_NO_DAMPING",
                    # "pbe0",
                    "SAPT_DFT_pbe0_adz_3_IE",
                ],
            ],
            benchmark_label="benchmark ref energy",
            disp_compute=locald4.compute_disp_2B_NO_DAMPING,
        )
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT_DFT_pbe0_adz_total",
                    "SAPT_DFT_pbe0_adz_3_IE_inter",
                    "SAPT_DFT_pbe0_adz_3_IE_inter",
                    "SAPT_DFT_pbe0_adz_3_IE",
                ],
                [
                    "SAPT_DFT_b3lyp_adz_total",
                    "SAPT_DFT_b3lyp_adz_3_IE_inter",
                    "SAPT_DFT_b3lyp_adz_3_IE_inter",
                    "SAPT_DFT_b3lyp_adz_3_IE",
                ],
                [
                    "SAPT_DFT_pbe0_adz_total",
                    "SAPT_DFT_pbe0_adz_3_IE_inter_NO_DAMPING",
                    "SAPT_DFT_pbe0_adz_3_IE_inter_NO_DAMPING",
                    # "pbe0",
                    "SAPT_DFT_pbe0_adz_3_IE",
                ],
            ],
            benchmark_label="benchmark ref energy",
            disp_compute=locald4.compute_disp_2B_BJ_dimer_inter,
        )
        df = compute_d4_from_opt_params(
            df,
            bases=[
                [
                    "SAPT0_adz",
                    "HF",
                    "HF",
                    "SAPT0_adz",
                ],
                [
                    "SAPT0_adz",
                    "HF_ATM",
                    "HF_ATM",
                    "SAPT0_adz",
                ],
            ],
            benchmark_label="benchmark ref energy",
            disp_compute=locald4.compute_disp_2B_BJ_ATM_CHG_dimer,
        )
        df.to_pickle(df_out)
        # df = compute_d4_from_opt_params(
        #     df,
        #     bases=[
        #         [
        #             "SAPT_DFT_pbe0_adz_total",
        #             "SAPT_DFT_adz_TT_inter",
        #             "SAPT_DFT_adz_TT_inter",
        #             # "pbe0",
        #             "SAPT_DFT_pbe0_adz_3_IE",
        #         ],
        #     ],
        #     benchmark_label="benchmark ref energy",
        #     disp_compute=locald4.compute_disp_2B_TT_dimer_inter,
        # )
        # print(df[["-D4 (SAPT_DFT_adz_TT_inter)"]])
        # df.to_pickle(df_out)
    # plot_violin_SAPT0_DFT_components(
    if original_plot:
        plot_violin_SAPT0_DFT_components(
            df,
            pfn=f"{basename_selected}_saptdft_components2",
            elst_vals={
                "name": "Electrostatics",
                "reference": [
                    f"{reference}/{ref_basis} Ref.",
                    f"{reference} ELST ENERGY",
                ],
                "vals": {
                    "SAPT0/aDZ": "SAPT0_adz_elst",
                    "SAPT0/aTZ": "SAPT0_atz_elst",
                    "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_elst",
                    "SAPT(DFT)/aTZ": "SAPT_DFT_pbe0_atz_elst",
                },
            },
            exch_vals={
                "name": "Exchange",
                "reference": [
                    f"{reference}/{ref_basis} Ref.",
                    f"{reference} EXCH ENERGY",
                ],
                "vals": {
                    "SAPT0/aDZ": "SAPT0_adz_exch",
                    "SAPT0/aTZ": "SAPT0_atz_exch",
                    "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_exch",
                    "SAPT(DFT)/aTZ": "SAPT_DFT_pbe0_atz_exch",
                },
            },
            indu_vals={
                "name": "Induction",
                "reference": [
                    f"{reference}/{ref_basis} Ref.",
                    f"{reference} IND ENERGY",
                ],
                "vals": {
                    "SAPT0/aDZ": "SAPT0_adz_indu",
                    "SAPT0/aTZ": "SAPT0_atz_indu",
                    "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_indu",
                    "SAPT(DFT)/aTZ": "SAPT_DFT_pbe0_atz_indu",
                },
            },
            disp_vals={
                "name": "Dispersion",
                "reference": [
                    f"{reference}/{ref_basis} Ref.",
                    f"{reference} DISP ENERGY",
                ],
                "vals": {
                    # "SAPT0/aDZ": "SAPT0_adz_indu",
                    # "SAPT0/aTZ": "SAPT0_atz_indu",
                    # "SAPT0-D4/aDZ": "-D4 (SAPT0_adz_3_IE)",
                    "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_disp",
                    "SAPT(DFT)/aTZ": "SAPT_DFT_pbe0_atz_disp",
                    "SAPT(DFT)-D4/aDZ": "SAPT_DFT_pbe0_adz_d4_disp",
                    "SAPT(DFT)-D4/aTZ": "SAPT_DFT_pbe0_atz_d4_disp",
                },
            },
            three_total_vals={
                "name": "(Elst. + Exch. + Indu.)",
                "reference": ["CCSD(T)/CBS IE Ref.", "benchmark ref energy"],
                "vals": {
                    "SAPT(DFT)/aDZ": "SAPT_DFT_pbe0_adz_3_IE",
                    "SAPT(DFT)/aTZ": "SAPT_DFT_pbe0_atz_3_IE",
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


def plot_violin_d3_d4_ALL_zoomed_min_max_TOC(
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
        1, 1, height_ratios=[1]
    )  # Adjust height ratios to change the size of subplots

    # Create the main violin plot axis
    ax = plt.subplot(gs[0])  # This will create the subplot for the main violin plot.
    # ax = plt.subplot(111)
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        # quantiles=[[0.05, 0.95] for i in range(len(vData))],
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
    # for n, partname in enumerate(["cquantiles"]):
    #     vp = vplot[partname]
    #     vp.set_edgecolor(quantile_color)
    #     vp.set_linewidth(quantile_linewidth)
    #     vp.set_linestyle(quantile_style)
    #     vp.set_alpha(1)

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

    # plt.setp(ax.set_xticklabels(vLabels), rotation=-45, fontsize="16", ha="left")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)

    # lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    # lg.get_frame().set_alpha(None)
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    # ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k", fontsize="14")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")

    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)
    # Annotations of RMSE

    plt.setp(ax.set_xticklabels(vLabels), rotation=35, fontsize="16", ha="right")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)

    minor_yticks = create_minor_y_ticks(ylim)
    ax.set_yticks(minor_yticks, minor=True)
    # incread tick sizes
    ax.tick_params(axis="both", which="major", labelsize=16)
    # lg = ax.legend(loc=legend_loc, edgecolor="black", fontsize="9")
    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k", fontsize="12")
    ax.set_ylabel(r"Error (kcal$\cdot$mol$^{-1}$)", color="k", fontsize="14")
    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    # ax_error = plt.subplot(gs[0], sharex=ax)
    # # ax_error.spines['top'].set_visible(False)
    # ax_error.spines["right"].set_visible(False)
    # ax_error.spines["left"].set_visible(False)
    # ax_error.spines["bottom"].set_visible(False)
    # ax_error.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    #
    # # Synchronize the x-limits with the main subplot
    # ax_error.set_xlim((0, len(vLabels)))
    # ax_error.set_ylim(0, 1)  # Assuming the upper subplot should have no y range
    # # Populate ax_error with error statistics through annotations
    # # text = r"\textit{%.2f}" % mae
    # # text += r"\textbf{%.2f}" % rmse
    # # text += r"\textrm{%.2f}" % max_error
    # error_labels = r"\textit{MAE}"
    # error_labels += "\n"
    # error_labels += r"\textbf{RMSE}"
    # error_labels += "\n"
    # error_labels += r"\textrm{MaxE}"
    # error_labels += "\n"
    # error_labels += r"\textrm{MinE}"
    # ax_error.annotate(
    #     error_labels,
    #     xy=(0, 1),  # Position at the vertical center of the narrow subplot
    #     xytext=(0, 0.2),
    #     color="black",
    #     fontsize="8",
    #     ha="center",
    #     va="center",
    # )
    # for idx, (x, y, text) in enumerate(annotations):
    #     ax_error.annotate(
    #         text,
    #         xy=(x, 1),  # Position at the vertical center of the narrow subplot
    #         # xytext=(0, 0),
    #         xytext=(x, 0.2),
    #         color="black",
    #         fontsize="9",
    #         ha="center",
    #         va="center",
    #     )

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
        ax.set_xticklabels(vLabels), rotation=sub_rotation, fontsize=f"{fontsize - 2}"
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
        title_color = "green"
    elif title_name == "Induction":
        title_color = "blue"
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
        ylim=[-4, 10],
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
        ylim=[-4, 50],
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


def compute_CRE(
    row,
    energy_col="SAPT0 TOTAL ENERGY",
    r_eq_col="R",
    benchmark_col="benchmark ref energy",
    benchmark_col_system="E_R_eq",
    gamma=0.2,
    convert_energy_col=True,
    benchmark_col_weight=None,
    benchmark_col_system_weight=None,
    debug=False,
):
    """
    Compute CRE for a given row to compute MCURE
    If benchmark_col_weight=None and benchmark_col_system_weight=None, then
    use energy for CRE for E_weight as well.
    """
    if benchmark_col_system_weight is None or benchmark_col_weight is None:
        benchmark_col_system_weight = benchmark_col_system
        benchmark_col_weight = benchmark_col
    if convert_energy_col:
        row_E = row[energy_col] * h2kcalmol
    if debug:
        sys_id = row["system_id"]
        db = row["DB"]
        ref = row[benchmark_col]
        print(f"{sys_id = }, {db = }, {ref = }")
    if row["DB"].lower() == "ion43" or row["DB"].lower() == "ssi":
        E_weight = max(abs(row[benchmark_col_weight]), 0.5)
    else:
        E_weight = max(
            abs(row[benchmark_col_weight]),
            gamma
            * abs(row[benchmark_col_weight] - row[benchmark_col_system_weight])
            / row[r_eq_col] ** 3,
        )
    CRE = (row_E - row[benchmark_col]) / E_weight
    if debug:
        # print only 2 decimal places
        ref = row[benchmark_col]
        sys_id = row["system_id"]
        print(
            f"{sys_id = } {row_E = :.2f}, {ref = :.2f} {E_weight = :.2f}, {CRE = :.2f}"
        )
    return CRE


def _total_sapt_methods():
    return [
        "MP2 IE",
        "PBE0 IE",
        "B3LYP IE",
        "B2PLYP IE",
        "WB97X IE",
        "SAPT0 TOTAL ENERGY",
        "SSAPT0 TOTAL ENERGY",
        "SAPT2 TOTAL ENERGY",
        "SAPT2+ TOTAL ENERGY",
        "SAPT2+(3) TOTAL ENERGY",
        "SAPT2+3 TOTAL ENERGY",
        "SAPT2+(CCD) TOTAL ENERGY",
        "SAPT2+(3)(CCD) TOTAL ENERGY",
        "SAPT2+3(CCD) TOTAL ENERGY",
        "SAPT2+DMP2 TOTAL ENERGY",
        "SAPT2+(3)DMP2 TOTAL ENERGY",
        "SAPT2+3DMP2 TOTAL ENERGY",
        "SAPT2+(CCD)DMP2 TOTAL ENERGY",
        "SAPT2+(3)(CCD)DMP2 TOTAL ENERGY",
        "SAPT2+3(CCD)DMP2 TOTAL ENERGY",
        "SAPT0-D4 TOTAL ENERGY",
        "SAPT TOTAL ENERGY",
        "SAPT(DFT)D3-ML TOTAL ENERGY",
        "SAPT(B3LYP)D3-ML TOTAL ENERGY",
        "SAPT(DFT)-D4 TOTAL ENERGY",
        "SAPT(DFT)+D4 TOTAL ENERGY",
        "SAPT(PBE0)-D4 INTER TOTAL ENERGY",
        "SAPT(B3LYP)-D4 INTER TOTAL ENERGY",
        "SAPT(PBE0)-D3 INTER TOTAL ENERGY",
        "SAPT(B3LYP)-D3 INTER TOTAL ENERGY",
        "PBE0-D3 TOTAL ENERGY",
        "B3LYP-D3 TOTAL ENERGY",
    ]


def _total_local_method_map(basis):
    return {
        f"SAPT_DFT_D4_pbe0_{basis}_total": "PBE0-D4 TOTAL ENERGY",
        f"SAPT_DFT_pbe0_{basis}_total": "SAPT(DFT) [PBE0] TOTAL ENERGY",
        f"SAPT_DFT_D4_b3lyp_{basis}_total": "B3LYP-D4 TOTAL ENERGY",
        f"SAPT_DFT_b3lyp_{basis}_total": "SAPT(DFT) [B3LYP] TOTAL ENERGY",
        f"SAPT_DFT_D4_b2plyp_{basis}_total": "B2PLYP-D4 TOTAL ENERGY",
        f"SAPT_DFT_b2plyp_{basis}_total": "SAPT(DFT) [B2PLYP] TOTAL ENERGY",
        f"SAPT_DFT_D4_wb97x_{basis}_total": "WB97X-D4 TOTAL ENERGY",
        f"SAPT_DFT_wb97x_{basis}_total": "SAPT(DFT) [WB97X] TOTAL ENERGY",
    }


def _total_plot_labels():
    return {
        "SAPT(PBE0)": "SAPT(DFT) [PBE0] TOTAL ENERGY Error",
        "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] TOTAL ENERGY Error",
        "PBE0-D4": "PBE0-D4 TOTAL ENERGY Error",
        "B3LYP-D4": "B3LYP-D4 TOTAL ENERGY Error",
        "PBE0-D3": "PBE0-D3 TOTAL ENERGY Error",
        "B3LYP-D3": "B3LYP-D3 TOTAL ENERGY Error",
        "SAPT(PBE0)-D4(S)": "SAPT(DFT)-D4 TOTAL ENERGY Error",
        "SAPT(PBE0)-D4(I)": "SAPT(PBE0)-D4 INTER TOTAL ENERGY Error",
        "SAPT(B3LYP)-D4(I)": "SAPT(B3LYP)-D4 INTER TOTAL ENERGY Error",
        "SAPT(PBE0)-D3(I)": "SAPT(PBE0)-D3 INTER TOTAL ENERGY Error",
        "SAPT(PBE0)D3-ML": "SAPT(DFT)D3-ML TOTAL ENERGY Error",
        "SAPT(B3LYP)D3-ML": "SAPT(B3LYP)D3-ML TOTAL ENERGY Error",
        "SAPT0-D4": "SAPT0-D4 TOTAL ENERGY Error",
        "SAPT0": "SAPT0 TOTAL ENERGY Error",
        "SAPT2+3": "SAPT2+3 TOTAL ENERGY Error",
        "SAPT2+3(CCD)": "SAPT2+3(CCD) TOTAL ENERGY Error",
        "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD)DMP2 TOTAL ENERGY Error",
    }


def _filter_plot_labels_by_available_columns(df_labels_and_columns, dfs):
    if not dfs:
        return {}
    common = set(dfs[0]["df"].columns)
    for d in dfs[1:]:
        common &= set(d["df"].columns)
    return {k: v for k, v in df_labels_and_columns.items() if v in common}


def _prepare_total_violin_dfs(
    df,
    bases,
    limit_to_column_not_nan=None,
    subset_only=False,
):
    if subset_only:
        df = df[df["subset"]].copy()
        print(f"Subset: {len(df)}")
    if limit_to_column_not_nan is not None:
        size_prior = len(df)
        df = df[df[limit_to_column_not_nan].notna()].copy()
        print(
            f"Limiting to {limit_to_column_not_nan} not NaN: {size_prior} -> {len(df)}"
        )

    sapt_methods = _total_sapt_methods()
    base_cols = ["DB", "system_id", "benchmark ref energy", "E_R_eq", "R"]
    basis_labels = {
        "adz": "aug-cc-pVDZ",
        "atz": "aug-cc-pVTZ",
        "aqz": "aug-cc-pVQZ",
    }
    dfs = []

    for basis in bases:
        local_rename = _total_local_method_map(basis)
        copy_cols = [c for c in base_cols if c in df.columns]
        copy_cols.extend([c for c in local_rename if c in df.columns])
        copy_cols.extend(
            [f"{m} {basis}" for m in sapt_methods if f"{m} {basis}" in df.columns]
        )

        df_basis = df[copy_cols].copy()
        df_basis.columns = [c.replace(f" {basis}", "") for c in df_basis.columns]
        rename_map = {k: v for k, v in local_rename.items() if k in df_basis.columns}
        df_basis.rename(columns=rename_map, inplace=True)

        methods_for_error = [m for m in sapt_methods if m in df_basis.columns]
        methods_for_error.extend(
            [v for v in local_rename.values() if v in df_basis.columns]
        )
        methods_for_error = list(dict.fromkeys(methods_for_error))

        reference = "benchmark ref energy"
        df_basis[reference] = pd.to_numeric(df_basis[reference], errors="coerce")
        for method in methods_for_error:
            df_basis[method] = pd.to_numeric(df_basis[method], errors="coerce")
            df_basis[f"{method} Error"] = (
                df_basis[method] * h2kcalmol - df_basis[reference]
            )

        dfs.append(
            {
                "df": df_basis,
                "label": basis_labels[basis],
                "basis": basis_labels[basis],
                "ylim": [[-3, 3] for _ in range(len(bases))],
                "name": basis,
            }
        )
    return dfs


def violin_plots_multi(df, limit_to_column_not_nan=None, slide=True, dfs=None):
    if dfs is None:
        dfs = _prepare_total_violin_dfs(
            df,
            bases=("adz", "atz"),
            limit_to_column_not_nan=limit_to_column_not_nan,
        )
    df_labels_and_columns = _filter_plot_labels_by_available_columns(
        _total_plot_labels(), dfs
    )

    import cdsg_plot

    if slide:
        fig_size = (9, 6)
        grid_heights = [0.9, 2, 0.7, 2]
        table_fontsize = 12
        x_label_fontsize = 12
    else:
        fig_size = (9, 8)
        grid_heights = [0.6, 2, 0.4, 2]
        table_fontsize = 8
        x_label_fontsize = 8

    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_total=df_labels_and_columns,
        output_filename="./plots/LoS_all_adz_atz_saptdft.jpg",
        wspace=0.8,
        usetex=True,
        colors=colors_total_saptdftd4,
        violin_alphas=0.9,
        legend_loc="lower right",
        table_fontsize=table_fontsize,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=11,
        figure_size=fig_size,
        grid_heights=grid_heights,
        grid_widths=[1],
    )
    return


def violin_plots_multi_individual(df, limit_to_column_not_nan=None, dfs=None):
    if dfs is None:
        dfs = _prepare_total_violin_dfs(
            df,
            bases=("adz", "atz"),
            limit_to_column_not_nan=limit_to_column_not_nan,
        )
    df_labels_and_columns = _filter_plot_labels_by_available_columns(
        {
            "SAPT0": "SAPT0 TOTAL ENERGY Error",
            "PBE0-D4": "PBE0-D4 TOTAL ENERGY Error",
            "SAPT(PBE0)-D4(I)": "SAPT(PBE0)-D4 INTER TOTAL ENERGY Error",
        },
        dfs,
    )

    import cdsg_plot

    for d in dfs:
        cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
            [d],
            df_labels_and_columns_total=df_labels_and_columns,
            output_filename=f"./plots/individuals/LoS_all_{d['name']}_saptdft.jpg",
            usetex=True,
            figure_size=(4, 3),
            colors=[[PURPLE, TEAL, BLUE] for _ in range(5)],
            grid_widths=[1.0],
            grid_heights=[0.35, 2],
            violin_alphas=0.9,
            legend_loc="lower right",
            table_fontsize=12,
            x_label_fontsize=11,
            y_label_fontsize=11,
            MaxE=None,
            MinE=None,
            add_title=False,
            x_label_rotation=15,
        )
    return


def violin_plots_multi_subset(df, limit_to_column_not_nan=None, dfs=None):
    if dfs is None:
        dfs = _prepare_total_violin_dfs(
            df,
            bases=("adz", "atz", "aqz"),
            limit_to_column_not_nan=limit_to_column_not_nan,
            subset_only=True,
        )
    df_labels_and_columns = _filter_plot_labels_by_available_columns(
        _total_plot_labels(), dfs
    )

    import cdsg_plot

    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_total=df_labels_and_columns,
        output_filename="./plots/LoS_all_adz_atz_aqz_saptdft_subset.jpg",
        colors=colors_total_saptdftd4,
        wspace=0.8,
        usetex=True,
        violin_alphas=0.9,
        legend_loc="lower right",
        y_label_fontsize=11,
        table_fontsize=12,
        x_label_fontsize=12,
        figure_size=(9, 9),
        grid_heights=[0.9, 2, 0.7, 2, 0.7, 2],
        grid_widths=[1],
    )
    return


def violin_plots_multi_subset_individual(df, limit_to_column_not_nan=None, dfs=None):
    if dfs is None:
        dfs = _prepare_total_violin_dfs(
            df,
            bases=("adz", "atz", "aqz"),
            limit_to_column_not_nan=limit_to_column_not_nan,
            subset_only=True,
        )
    df_labels_and_columns = _filter_plot_labels_by_available_columns(
        _total_plot_labels(), dfs
    )

    import cdsg_plot

    for d in dfs:
        cdsg_plot.error_statistics.violin_plot_table_multi(
            [d],
            df_labels_and_columns,
            f"./plots/individuals/LoS_all_{d['name']}_saptdft_subset.jpg",
            table_fontsize=8,
            usetex=True,
            legend_loc="lower right",
            figure_size=(10, 3),
            colors=[
                TEAL,
                LIGHT_PURPLE,
                Medium_Sea_Green,
                INDIGO,
                TEAL,
                LIGHT_PURPLE,
                Medium_Sea_Green,
                INDIGO,
            ],
            violin_alpha=0.9,
            x_label_fontsize=12,
            error_labels_position=(-0.3, 0.25),
            grid_widths=[1.0],
            grid_heights=[0.45, 2],
        )
    return


def sapt_error_comp(
    df,
    df_ref,
    sapt_reference,
    sapt_methods,
    reference="benchmark ref energy",
    extra_label="",
    conv=h2kcalmol,
):
    for i in sapt_methods:
        print(i)
        if "ELST" in i:
            ref = sapt_reference["ELST"]
        elif "EXCH" in i:
            ref = sapt_reference["EXCH"]
        elif "IND" in i:
            ref = sapt_reference["IND"]
        elif "DISP" in i:
            ref = sapt_reference["DISP"]
        # elif "TOTAL" in i:
        else:
            ref = reference
            df[f"{i} Error{extra_label}"] = df[i] * conv - df_ref[ref]
            continue
        # else:
        #     raise ValueError(f"Error: {i = }")
        print(i, ref)
        df[f"{i} Error{extra_label}"] = (df[i] - df_ref[ref]) * conv
    return df


def prep_saptdft_components(df, functional, basis_set):
    df[f"SAPT_DFT_{functional}_{basis_set}"] = df.apply(
        lambda r: (
            [i / h2kcalmol for i in r[f"SAPT_DFT_{functional}_{basis_set}"]]
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else r[f"SAPT_DFT_{functional}_{basis_set}"]
        ),
        axis=1,
    )
    print(df[[f"SAPT_DFT_{functional}_{basis_set}"]].isna().sum())
    df[f"SAPT(DFT) [{functional.upper()}] ELST ENERGY {basis_set}"] = df.apply(
        lambda r: (
            r[f"SAPT_DFT_{functional}_{basis_set}"][1]
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    df[f"SAPT(DFT) [{functional.upper()}] EXCH ENERGY {basis_set}"] = df.apply(
        lambda r: (
            r[f"SAPT_DFT_{functional}_{basis_set}"][2]
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    # if functional == 'b2plyp':
    #     # df[f'SAPT MP2(2) ENERGY {basis_set}'] = df[f'SAPT MP2(2) ENERGY {basis_set}'] * h2kcalmol
    #     # print("B2PLYP diff")
    #     print(df[[f'MP2 IE {basis_set}', f"SAPT_DFT_{functional}_{basis_set}", f"SAPT DISP20 ENERGY {basis_set}"]])
    #     print(
    #             df[f"MP2 IE {basis_set}"],
    #             df[f"SAPT_DFT_{functional}_{basis_set}"],
    #             df[f'SAPT DISP20 ENERGY {basis_set}'],
    #             df[f'SAPT EXCH-DISP20 ENERGY {basis_set}'],
    #             df[f'SAPT HF(2) ENERGY {basis_set}'],
    #     )
    df[f"SAPT(DFT) [{functional.upper()}] dMP2 ENERGY {basis_set}"] = df.apply(
        lambda r: (
            (
                # need to define dMP2 as dHF-like but with DFT non-disp
                # components and SAPT0 disp components
                r[f"MP2 IE {basis_set}"]
                - (
                    r[f"SAPT_DFT_{functional}_{basis_set}"][1]
                    + r[f"SAPT_DFT_{functional}_{basis_set}"][2]
                    + r[f"SAPT_DFT_{functional}_{basis_set}"][3]
                    + r[f"SAPT DISP20 ENERGY {basis_set}"]
                    + r[f"SAPT EXCH-DISP20 ENERGY {basis_set}"]
                )
            )
            * h2kcalmol
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    # if functional == 'b2plyp':
    df[f"SAPT(DFT) [{functional.upper()}] dMP2 IND ENERGY {basis_set}"] = df.apply(
        lambda r: (
            r[f"SAPT_DFT_{functional}_{basis_set}"][3]
            # + r[f'SAPT MP2(2) ENERGY {basis_set}']
            + r[f"SAPT(DFT) [{functional.upper()}] dMP2 ENERGY {basis_set}"] / h2kcalmol
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    # else:
    df[f"SAPT(DFT) [{functional.upper()}] dMP2 EXCH ENERGY {basis_set}"] = df.apply(
        lambda r: (
            r[f"SAPT_DFT_{functional}_{basis_set}"][2]
            + r[f"SAPT(DFT) [{functional.upper()}] dMP2 ENERGY {basis_set}"] / h2kcalmol
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    # else:
    df[f"SAPT(DFT) [{functional.upper()}] IND ENERGY {basis_set}"] = df.apply(
        lambda r: (
            r[f"SAPT_DFT_{functional}_{basis_set}"][3]
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    df[f"SAPT(DFT) [{functional.upper()}] DISP ENERGY {basis_set}"] = df.apply(
        lambda r: (
            r[f"SAPT_DFT_{functional}_{basis_set}"][4]
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    # if functional == 'b2plyp':
    # print("B2PLYP diff")
    df[f"{functional.upper()}-D4 dMP2 DISP ENERGY {basis_set}"] = df.apply(
        lambda r: (
            (
                r[f"SAPT_DFT_{functional}_{basis_set}_D4_IE"]
                + r[f"SAPT_DFT_{functional}_{basis_set}_dDFT"]
                - r[f"SAPT_DFT_{functional}_{basis_set}_dHF"]
                # - r[f'SAPT MP2(2) ENERGY {basis_set}'] * h2kcalmol
                - r[f"SAPT(DFT) [{functional.upper()}] dMP2 ENERGY {basis_set}"]
            )
            / h2kcalmol
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    # else:
    df[f"{functional.upper()}-D4 DISP ENERGY {basis_set}"] = df.apply(
        lambda r: (
            (
                r[f"SAPT_DFT_{functional}_{basis_set}_D4_IE"]
                + r[f"SAPT_DFT_{functional}_{basis_set}_dDFT"]
                - r[f"SAPT_DFT_{functional}_{basis_set}_dHF"]
            )
            / h2kcalmol
            if r[f"SAPT_DFT_D4_{functional}_{basis_set}_total"]
            and r[f"SAPT_DFT_{functional}_{basis_set}"]
            else np.nan
        ),
        axis=1,
    )
    return df


def violin_plots_multi_components_sapt0d4(df):
    df = prep_saptdft_components(df, "pbe0", "adz")
    df = prep_saptdft_components(df, "pbe0", "atz")

    df["SAPT0-D4 (Intermol.) DISP ENERGY adz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 (Intermol.) DISP ENERGY atz"] = df["-D4 (SAPT0_atz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 (Intermol.) DISP ENERGY aqz"] = df["-D4 (SAPT0_atz_3_IE)"] / h2kcalmol

    df["SAPT0-D4 (Super.) DISP ENERGY adz"] = (
        df["-D4 (SAPT0_adz_3_IE_BJ_inter)"] / h2kcalmol
    )
    df["SAPT0-D4 (Super.) DISP ENERGY atz"] = (
        df["-D4 (SAPT0_atz_3_IE_BJ_inter)"] / h2kcalmol
    )
    df["SAPT0-D4 (Super.) DISP ENERGY aqz"] = (
        df["-D4 (SAPT0_atz_3_IE_BJ_inter)"] / h2kcalmol
    )

    df["SAPT0-D3 (Super.) DISP ENERGY adz"] = df["-D3 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D3 (Super.) DISP ENERGY atz"] = df["-D3 (SAPT0_atz_3_IE)"] / h2kcalmol
    df["SAPT0-D3 (Super.) DISP ENERGY aqz"] = df["-D3 (SAPT0_atz_3_IE)"] / h2kcalmol

    sapt_methods = [
        "SAPT0 ELST ENERGY",
        "SAPT2 ELST ENERGY",
        "SAPT2+(3) ELST ENERGY",
        "SAPT(DFT) [PBE0] ELST ENERGY",
        "SAPT0 EXCH ENERGY",
        "SAPT2 EXCH ENERGY",
        "SAPT(DFT) [PBE0] EXCH ENERGY",
        "SAPT0 IND ENERGY",
        "SSAPT0 IND ENERGY",
        "SAPT2 IND ENERGY",
        "SAPT2+DMP2 IND ENERGY",
        "SAPT2+3DMP2 IND ENERGY",
        "SAPT(DFT) [PBE0] IND ENERGY",
        "SAPT0 DISP ENERGY",
        "SSAPT0 DISP ENERGY",
        "SAPT2+ DISP ENERGY",
        "SAPT2+(3) DISP ENERGY",
        "SAPT2+3 DISP ENERGY",
        "SAPT2+(CCD) DISP ENERGY",
        "SAPT2+(3)(CCD) DISP ENERGY",
        "SAPT2+3(CCD) DISP ENERGY",
        # local disp
        "SAPT0-D3 (Super.) DISP ENERGY",
        "SAPT0-D4 (Intermol.) DISP ENERGY",
        "SAPT0-D4 (Super.) DISP ENERGY",
        "SAPT(DFT) [PBE0] DISP ENERGY",
    ]
    sapt_reference = {
        "ELST": "SAPT2+3(CCD)DMP2 ELST ENERGY",
        "EXCH": "SAPT2+3(CCD)DMP2 EXCH ENERGY",
        "IND": "SAPT2+3(CCD)DMP2 IND ENERGY",
        "DISP": "SAPT2+3(CCD)DMP2 DISP ENERGY",
    }

    reference = "benchmark ref energy"
    copy_cols_start = [
        "DB",
        "system_id",
        "benchmark ref energy",
        "E_R_eq",
        "R",
        "Ref_elst_atz",
        "E_R_eq_elst_atz",
        "Ref_exch_atz",
        "E_R_eq_exch_atz",
        "Ref_ind_atz",
        "E_R_eq_ind_atz",
        "Ref_disp_atz",
        "E_R_eq_disp_atz",
    ]
    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} adz" for c in sapt_methods])
    copy_cols.extend([f"{c} adz" for c in sapt_reference.values()])
    df_adz = df[copy_cols].copy()
    df_adz.columns = [c.replace(" adz", "") for c in df_adz.columns]

    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} atz" for c in sapt_methods])
    copy_cols.extend([f"{c} atz" for c in sapt_reference.values()])
    df_atz = df[copy_cols].copy()
    df_atz.columns = [c.replace(" atz", "") for c in df_atz.columns]

    df_adz = sapt_error_comp(df_adz, df_atz, sapt_reference, sapt_methods)
    df_atz = sapt_error_comp(df_atz, df_atz, sapt_reference, sapt_methods)

    adz_ylims = [
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
    ]
    atz_ylims = [
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
    ]

    dfs = [
        {
            "df": df_adz,
            "basis": "aug-cc-pVDZ",
            "label": "aug-cc-pVDZ",
            "ylim": adz_ylims,
        },
        {
            "df": df_atz,
            "basis": "aug-cc-pVTZ",
            "label": "aug-cc-pVTZ",
            "ylim": atz_ylims,
        },
    ]
    mcures_labels_start = [
        {
            "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
        },
        {
            "SAPT2+": "SAPT2 ELST ENERGY Error",
        },
        {
            "SAPT2+3(CCD)DMP2": "SAPT2+(3) ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
        },
        {
            "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
        },
        {
            "SAPT2+,\\\\SAPT2+3(CCD)DMP2": "SAPT2 EXCH ENERGY Error",
        },
        {
            "SAPT0": "SAPT0 IND ENERGY Error",
        },
        {
            "sSAPT0": "SSAPT0 IND ENERGY Error",
        },
        {
            "SAPT2+": "SAPT2 IND ENERGY Error",
        },
        {
            "SAPT2+3(CCD)DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
        },
        {
            "SAPT0": "SAPT0 DISP ENERGY Error",
        },
        {
            "sSAPT0": "SSAPT0 DISP ENERGY Error",
        },
        {
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
        },
        {
            "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
        },
        {
            "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
        },
        {
            "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
        },
        {
            "SAPT0-D4 (Intermol.)": "SAPT0-D4 (Intermol.) DISP ENERGY Error",
        },
        {
            "SAPT0-D4 (Super.)": "SAPT0-D4 (Super.) DISP ENERGY Error",
        },
        {
            "SAPT0-D3 (Super.)": "SAPT0-D3 (Super.) DISP ENERGY Error",
        },
    ]
    mcure_labels = {
        "ELST": {},
        "EXCH": {},
        "IND": {},
        "DISP": {},
    }
    # TODO DEBUG
    df_atz["SAPT2+3(CCD)DMP2 EXCH ENERGY"] *= h2kcalmol
    debug = False
    for i in mcures_labels_start:
        k, v = list(i.items())[0]
        for d in dfs:
            if "ELST" in v:
                if k not in mcure_labels["ELST"]:
                    mcure_labels["ELST"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_ELST"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_elst_atz",
                        benchmark_col_system=f"E_R_eq_elst_atz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_ELST"].abs().mean() * 100
                mcure_labels["ELST"][k].append(mcure)
            elif "EXCH" in v:
                if k not in mcure_labels["EXCH"]:
                    mcure_labels["EXCH"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_EXCH"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_exch_atz",
                        benchmark_col_system=f"E_R_eq_exch_atz",
                        debug=debug,
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_EXCH"].abs().mean() * 100
                mcure_labels["EXCH"][k].append(mcure)
            elif "IND" in v:
                if k not in mcure_labels["IND"]:
                    mcure_labels["IND"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_IND"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_ind_atz",
                        benchmark_col_system=f"E_R_eq_ind_atz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_IND"].abs().mean() * 100
                mcure_labels["IND"][k].append(mcure)
            elif "DISP" in v:
                if k not in mcure_labels["DISP"]:
                    mcure_labels["DISP"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_DISP"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_disp_atz",
                        benchmark_col_system=f"E_R_eq_disp_atz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_DISP"].abs().mean() * 100
                mcure_labels["DISP"][k].append(mcure)
            else:
                raise ValueError(f"Error: {v = }")
    pp(mcure_labels)

    import cdsg_plot

    # cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
    #     dfs,
    #     df_labels_and_columns_elst={
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
    #         "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
    #         "SAPT2+": "SAPT2 ELST ENERGY Error",
    #         "SAPT2+3(CCD)DMP2": "SAPT2+(3) ELST ENERGY Error",
    #     },
    #     df_labels_and_columns_exch={
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
    #         "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
    #         "SAPT2+,\\\\SAPT2+3(CCD)DMP2": "SAPT2 EXCH ENERGY Error",
    #     },
    #     df_labels_and_columns_indu={
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
    #         "SAPT0": "SAPT0 IND ENERGY Error",
    #         "sSAPT0": "SSAPT0 IND ENERGY Error",
    #         "SAPT2+": "SAPT2 IND ENERGY Error",
    #         "SAPT2+3(CCD)DMP2": "SAPT2+3DMP2 IND ENERGY Error",
    #     },
    #     df_labels_and_columns_disp={
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
    #         "SAPT0": "SAPT0 DISP ENERGY Error",
    #         "SAPT0-D3 (Super.)": "SAPT0-D3 (Super.) DISP ENERGY Error",
    #         "SAPT0-D4 (Intermol.)": "SAPT0-D4 (Intermol.) DISP ENERGY Error",
    #         "SAPT0-D4 (Super.)": "SAPT0-D4 (Super.) DISP ENERGY Error",
    #         "sSAPT0": "SSAPT0 DISP ENERGY Error",
    #         "SAPT2+": "SAPT2+ DISP ENERGY Error",
    #         # "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
    #         # "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
    #         # "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
    #         # "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
    #         "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD) DISP ENERGY Error",
    #     },
    #     output_filename=f"./plots/basis_set_components_LoS.jpg",
    #     table_fontsize=8,
    #     usetex=True,
    #     legend_loc="lower right",
    #     figure_size=(12, 8),
    #     grid_heights = [
    #         0.45,
    #         2,
    #         0.20,
    #         2,
    #     ],
    #     grid_widths = [0.75, 0.65, 1.1, 2.0],
    #     mcure=mcure_labels,
    # )
    print(df_adz.columns.tolist())
    print(df_adz[["SAPT0 IND ENERGY Error"]])
    # Presentation
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_elst={
            # "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
            "SAPT0": "SAPT0 ELST ENERGY Error",
            # "SAPT2+": "SAPT2 ELST ENERGY Error",
            "SAPT2+3\\\\(CCD)DMP2": "SAPT2+(3) ELST ENERGY Error",
        },
        df_labels_and_columns_exch={
            # "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
            "SAPT0": "SAPT0 EXCH ENERGY Error",
            "SAPT2+3\\\\(CCD)DMP2": "SAPT2 EXCH ENERGY Error",
        },
        df_labels_and_columns_indu={
            # "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
            "SAPT0": "SAPT0 IND ENERGY Error",
            # "sSAPT0": "SSAPT0 IND ENERGY Error",
            # "SAPT2+": "SAPT2 IND ENERGY Error",
            "SAPT2+3\\\\(CCD)DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        df_labels_and_columns_disp={
            "SAPT0": "SAPT0 DISP ENERGY Error",
            # "SAPT0-D3 (Super.)": "SAPT0-D3 (Super.) DISP ENERGY Error",
            # "SAPT0-D4 (Intermol.)": "SAPT0-D4 (Intermol.) DISP ENERGY Error",
            "SAPT0-D4": "SAPT0-D4 (Super.) DISP ENERGY Error",
            # "sSAPT0": "SSAPT0 DISP ENERGY Error",
            # "SAPT2+": "SAPT2+ DISP ENERGY Error",
            "SAPT2+3\\\\(CCD)DMP2": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        output_filename=f"./plots/basis_set_components_LoS_pres.jpg",
        usetex=True,
        legend_loc=None,
        figure_size=(10, 5),
        grid_widths=[2, 2, 2, 3],
        grid_heights=[
            0.08,
            1,
            0.08,
            1,
        ],
        colors=[
            [PURPLE, GREY],
            [PURPLE, GREY],
            [PURPLE, GREY],
            [PURPLE, PURPLE, GREY],
        ],
        table_fontsize=14,
        x_label_fontsize=13,
        y_label_fontsize=13,
        title_fontsize=16,
        mcure=None,
        MAE="textbf",
        RMSE=False,
        MinE=False,
        MaxE=False,
        annotations_texty=0.0,
        share_y_axis=True,
        wspace=0.05,
        table_delimiter=",",
        gridlines_linewidths=1.5,
        violin_alphas=1.0,
        quantile_linewidth=1.2,
        pm_alpha=0.5,
        zero_alpha=1.0,
    )
    return


def violin_plots_multi_components_subset_sapt0d4(df):
    df = df[df["subset"]].copy()
    print(f"Subset: {len(df)}")
    df = prep_saptdft_components(df, "pbe0", "adz")
    df = prep_saptdft_components(df, "pbe0", "atz")
    df = prep_saptdft_components(df, "pbe0", "aqz")
    print(
        df[
            [
                "system_id",
                "SAPT(DFT) [PBE0] DISP ENERGY adz",
                "SAPT(DFT) [PBE0] DISP ENERGY aqz",
            ]
        ]
    )

    df["SAPT0-D4 (Intermol.) DISP ENERGY adz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 (Intermol.) DISP ENERGY atz"] = df["-D4 (SAPT0_atz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 (Intermol.) DISP ENERGY aqz"] = df["-D4 (SAPT0_atz_3_IE)"] / h2kcalmol

    df["SAPT0-D4 (Super.) DISP ENERGY adz"] = (
        df["-D4 (SAPT0_adz_3_IE_BJ_inter)"] / h2kcalmol
    )
    df["SAPT0-D4 (Super.) DISP ENERGY atz"] = (
        df["-D4 (SAPT0_atz_3_IE_BJ_inter)"] / h2kcalmol
    )
    df["SAPT0-D4 (Super.) DISP ENERGY aqz"] = (
        df["-D4 (SAPT0_atz_3_IE_BJ_inter)"] / h2kcalmol
    )

    df["SAPT0-D3 (Super.) DISP ENERGY adz"] = df["-D3 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D3 (Super.) DISP ENERGY atz"] = df["-D3 (SAPT0_atz_3_IE)"] / h2kcalmol
    df["SAPT0-D3 (Super.) DISP ENERGY aqz"] = df["-D3 (SAPT0_atz_3_IE)"] / h2kcalmol

    sapt_methods = [
        "SAPT0 ELST ENERGY",
        "SAPT2 ELST ENERGY",
        "SAPT2+(3) ELST ENERGY",
        "SAPT(DFT) [PBE0] ELST ENERGY",
        "SAPT0 EXCH ENERGY",
        "SAPT2 EXCH ENERGY",
        "SAPT(DFT) [PBE0] EXCH ENERGY",
        "SAPT0 IND ENERGY",
        "SSAPT0 IND ENERGY",
        "SAPT2 IND ENERGY",
        "SAPT2+DMP2 IND ENERGY",
        "SAPT2+3DMP2 IND ENERGY",
        "SAPT(DFT) [PBE0] IND ENERGY",
        "SAPT0 DISP ENERGY",
        "SSAPT0 DISP ENERGY",
        "SAPT2+ DISP ENERGY",
        "SAPT2+(3) DISP ENERGY",
        "SAPT2+3 DISP ENERGY",
        "SAPT2+(CCD) DISP ENERGY",
        "SAPT2+(3)(CCD) DISP ENERGY",
        "SAPT2+3(CCD) DISP ENERGY",
        # local disp
        "SAPT0-D3 (Super.) DISP ENERGY",
        "SAPT0-D4 (Intermol.) DISP ENERGY",
        "SAPT0-D4 (Super.) DISP ENERGY",
        "SAPT(DFT) [PBE0] DISP ENERGY",
    ]
    sapt_reference = {
        "ELST": "SAPT2+3(CCD)DMP2 ELST ENERGY",
        "EXCH": "SAPT2+3(CCD)DMP2 EXCH ENERGY",
        "IND": "SAPT2+3(CCD)DMP2 IND ENERGY",
        "DISP": "SAPT2+3(CCD)DMP2 DISP ENERGY",
    }

    reference = "benchmark ref energy"
    copy_cols_start = [
        "DB",
        "system_id",
        "benchmark ref energy",
        "E_R_eq",
        "R",
        "Ref_elst_aqz",
        "E_R_eq_elst_aqz",
        "Ref_exch_aqz",
        "E_R_eq_exch_aqz",
        "Ref_ind_aqz",
        "E_R_eq_ind_aqz",
        "Ref_disp_aqz",
        "E_R_eq_disp_aqz",
    ]
    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} adz" for c in sapt_methods])
    copy_cols.extend([f"{c} adz" for c in sapt_reference.values()])
    df_adz = df[copy_cols].copy()
    df_adz.columns = [c.replace(" adz", "") for c in df_adz.columns]

    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} atz" for c in sapt_methods])
    copy_cols.extend([f"{c} atz" for c in sapt_reference.values()])
    df_atz = df[copy_cols].copy()
    df_atz.columns = [c.replace(" atz", "") for c in df_atz.columns]

    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} aqz" for c in sapt_methods])
    copy_cols.extend([f"{c} aqz" for c in sapt_reference.values()])
    df_aqz = df[copy_cols].copy()
    df_aqz.columns = [c.replace(" aqz", "") for c in df_aqz.columns]

    df_adz = sapt_error_comp(df_adz, df_aqz, sapt_reference, sapt_methods)
    df_atz = sapt_error_comp(df_atz, df_aqz, sapt_reference, sapt_methods)
    df_aqz = sapt_error_comp(df_aqz, df_aqz, sapt_reference, sapt_methods)

    adz_ylims = [
        [-3, 3],
        [-3, 3],
        [-3, 3],
        [-3, 3],
    ]

    dfs = [
        {
            "df": df_adz,
            "basis": "aug-cc-pVDZ",
            "label": "aug-cc-pVDZ",
            "ylim": adz_ylims,
        },
        {
            "df": df_atz,
            "basis": "aug-cc-pVTZ",
            "label": "aug-cc-pVTZ",
            "ylim": adz_ylims,
        },
        {
            "df": df_aqz,
            "basis": "aug-cc-pVQZ",
            "label": "aug-cc-pVQZ",
            "ylim": adz_ylims,
        },
    ]
    mcures_labels_start = [
        {
            "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
        },
        {
            "SAPT2+": "SAPT2 ELST ENERGY Error",
        },
        {
            "SAPT2+3(CCD)DMP2": "SAPT2+(3) ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
        },
        {
            "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
        },
        {
            "SAPT2+,\\\\SAPT2+3(CCD)DMP2": "SAPT2 EXCH ENERGY Error",
        },
        {
            "SAPT0": "SAPT0 IND ENERGY Error",
        },
        {
            "sSAPT0": "SSAPT0 IND ENERGY Error",
        },
        {
            "SAPT2+": "SAPT2 IND ENERGY Error",
        },
        {
            "SAPT2+3(CCD)DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
        },
        {
            "SAPT0": "SAPT0 DISP ENERGY Error",
        },
        {
            "sSAPT0": "SSAPT0 DISP ENERGY Error",
        },
        {
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
        },
        {
            "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
        },
        {
            "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
        },
        {
            "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
        },
        {
            "SAPT0-D4 (Intermol.)": "SAPT0-D4 (Intermol.) DISP ENERGY Error",
        },
        {
            "SAPT0-D4 (Super.)": "SAPT0-D4 (Super.) DISP ENERGY Error",
        },
        {
            "SAPT0-D3 (Super.)": "SAPT0-D3 (Super.) DISP ENERGY Error",
        },
    ]
    mcure_labels = {
        "ELST": {},
        "EXCH": {},
        "IND": {},
        "DISP": {},
    }
    # TODO DEBUG
    df_atz["SAPT2+3(CCD)DMP2 EXCH ENERGY"] *= h2kcalmol
    debug = False
    for i in mcures_labels_start:
        k, v = list(i.items())[0]
        for d in dfs:
            if "ELST" in v:
                if k not in mcure_labels["ELST"]:
                    mcure_labels["ELST"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_ELST"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_elst_aqz",
                        benchmark_col_system=f"E_R_eq_elst_aqz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_ELST"].abs().mean() * 100
                mcure_labels["ELST"][k].append(mcure)
            elif "EXCH" in v:
                if k not in mcure_labels["EXCH"]:
                    mcure_labels["EXCH"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_EXCH"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_exch_aqz",
                        benchmark_col_system=f"E_R_eq_exch_aqz",
                        debug=debug,
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_EXCH"].abs().mean() * 100
                mcure_labels["EXCH"][k].append(mcure)
            elif "IND" in v:
                if k not in mcure_labels["IND"]:
                    mcure_labels["IND"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_IND"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_ind_aqz",
                        benchmark_col_system=f"E_R_eq_ind_aqz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_IND"].abs().mean() * 100
                mcure_labels["IND"][k].append(mcure)
            elif "DISP" in v:
                if k not in mcure_labels["DISP"]:
                    mcure_labels["DISP"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_DISP"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_disp_aqz",
                        benchmark_col_system=f"E_R_eq_disp_aqz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_DISP"].abs().mean() * 100
                mcure_labels["DISP"][k].append(mcure)
            else:
                raise ValueError(f"Error: {v = }")
    pp(mcure_labels)

    import cdsg_plot

    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_elst={
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
            "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
            "SAPT2+": "SAPT2 ELST ENERGY Error",
            "SAPT2+3(CCD)DMP2": "SAPT2+(3) ELST ENERGY Error",
        },
        df_labels_and_columns_exch={
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
            "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
            "SAPT2+,\\\\SAPT2+3(CCD)DMP2": "SAPT2 EXCH ENERGY Error",
        },
        df_labels_and_columns_indu={
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
            "SAPT0": "SAPT0 IND ENERGY Error",
            "sSAPT0": "SSAPT0 IND ENERGY Error",
            "SAPT2+": "SAPT2 IND ENERGY Error",
            "SAPT2+3(CCD)DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        df_labels_and_columns_disp={
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
            "SAPT0": "SAPT0 DISP ENERGY Error",
            "SAPT0-D3 (Super.)": "SAPT0-D3 (Super.) DISP ENERGY Error",
            "SAPT0-D4 (Intermol.)": "SAPT0-D4 (Intermol.) DISP ENERGY Error",
            "SAPT0-D4 (Super.)": "SAPT0-D4 (Super.) DISP ENERGY Error",
            "sSAPT0": "SSAPT0 DISP ENERGY Error",
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
            # "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
            # "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
            # "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
            # "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
            "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        output_filename=f"./plots/basis_set_components_LoS_subset.jpg",
        table_fontsize=7.5,
        usetex=True,
        legend_loc="lower right",
        figure_size=(12, 10),
        grid_heights=[
            0.50,
            2,
            0.25,
            2,
            0.25,
            2,
        ],
        grid_widths=[0.75, 0.65, 1.1, 2.0],
        mcure=mcure_labels,
    )
    # Presentation
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_elst={
            # "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
            "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
            "SAPT2+": "SAPT2 ELST ENERGY Error",
            "SAPT2+3(CCD)DMP2": "SAPT2+(3) ELST ENERGY Error",
        },
        df_labels_and_columns_exch={
            # "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
            "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
            "SAPT2+,\\\\SAPT2+3(CCD)DMP2": "SAPT2 EXCH ENERGY Error",
        },
        df_labels_and_columns_indu={
            # "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
            "SAPT0": "SAPT0 IND ENERGY Error",
            "sSAPT0": "SSAPT0 IND ENERGY Error",
            "SAPT2+": "SAPT2 IND ENERGY Error",
            "SAPT2+3(CCD)DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        df_labels_and_columns_disp={
            "SAPT0": "SAPT0 DISP ENERGY Error",
            "SAPT0-D3 (Super.)": "SAPT0-D3 (Super.) DISP ENERGY Error",
            # "SAPT0-D4 (Intermol.)": "SAPT0-D4 (Intermol.) DISP ENERGY Error",
            "SAPT0-D4 (Super.)": "SAPT0-D4 (Super.) DISP ENERGY Error",
            "sSAPT0": "SSAPT0 DISP ENERGY Error",
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
            "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        output_filename=f"./plots/basis_set_components_LoS_subset.jpg",
        table_fontsize=7.5,
        usetex=True,
        legend_loc="lower right",
        figure_size=(12, 10),
        grid_heights=[
            0.50,
            2,
            0.25,
            2,
            0.25,
            2,
        ],
        grid_widths=[0.75, 0.65, 1.1, 2.0],
        mcure=mcure_labels,
        MAE=True,
        RMSE=False,
        MinE=False,
        MaxE=False,
    )
    return


_COMPONENT_SAPT_REFERENCE = {
    "ELST": "SAPT2+3(CCD) ELST ENERGY",
    "EXCH": "SAPT2+3(CCD) EXCH ENERGY",
    "IND": "SAPT2+3(CCD) IND ENERGY",
    "DISP": "SAPT2+3(CCD) DISP ENERGY",
}


_COMPONENT_METHODS_FULL = [
    "SAPT0 ELST ENERGY",
    "SAPT2 ELST ENERGY",
    "SAPT2+(3) ELST ENERGY",
    "SAPT(DFT) [PBE0] ELST ENERGY",
    "SAPT(DFT) [B3LYP] ELST ENERGY",
    "SAPT(DFT) [B2PLYP] ELST ENERGY",
    "SAPT(DFT) [WB97X] ELST ENERGY",
    "SAPT0 EXCH ENERGY",
    "SAPT2 EXCH ENERGY",
    "SAPT(DFT) [PBE0] EXCH ENERGY",
    "SAPT(DFT) [B3LYP] EXCH ENERGY",
    "SAPT(DFT) [B3LYP] dMP2 EXCH ENERGY",
    "SAPT(DFT) [B2PLYP] EXCH ENERGY",
    "SAPT(DFT) [WB97X] EXCH ENERGY",
    "SAPT0 IND ENERGY",
    "SAPT2 IND ENERGY",
    "SAPT2+DMP2 IND ENERGY",
    "SAPT2+3 IND ENERGY",
    "SAPT2+3DMP2 IND ENERGY",
    "SAPT(DFT) [PBE0] IND ENERGY",
    "SAPT(DFT) [B3LYP] IND ENERGY",
    "SAPT(DFT) [B2PLYP] IND ENERGY",
    "SAPT(DFT) [B2PLYP] dMP2 IND ENERGY",
    "SAPT(DFT) [WB97X] IND ENERGY",
    "SAPT0 DISP ENERGY",
    "SAPT2+ DISP ENERGY",
    "SAPT2+(3) DISP ENERGY",
    "SAPT2+3 DISP ENERGY",
    "SAPT2+(CCD) DISP ENERGY",
    "SAPT2+(3)(CCD) DISP ENERGY",
    "SAPT0-D4 DISP ENERGY",
    "PBE0-D4 DISP ENERGY",
    "PBE0-D3 DISP ENERGY",
    "B3LYP-D4 DISP ENERGY",
    "B3LYP-D3 DISP ENERGY",
    "B3LYP-D4 dMP2 DISP ENERGY",
    "B2PLYP-D4 DISP ENERGY",
    "B2PLYP-D4 dMP2 DISP ENERGY",
    "WB97X-D4 DISP ENERGY",
    "SAPT(DFT) [PBE0] DISP ENERGY",
    "SAPT(DFT) [B3LYP] DISP ENERGY",
    "SAPT(DFT) [B2PLYP] DISP ENERGY",
    "SAPT(DFT) [WB97X] DISP ENERGY",
    "D3-ML DISP ENERGY",
    "SAPT(DFT)+D4 DISP ENERGY",
    "SAPT(DFT)-D4 DISP ENERGY",
    "SAPT(PBE0)-D4 INTER DISP ENERGY",
    "SAPT(B3LYP)-D4 INTER DISP ENERGY",
    "SAPT(PBE0)-D3 INTER DISP ENERGY",
    "SAPT(B3LYP)-D3 INTER DISP ENERGY",
    "SAPT(PBE0)-D3 SUPER DISP ENERGY",
    "SAPT(B3LYP)-D3 SUPER DISP ENERGY",
]


_COMPONENT_METHODS_SUBSET = [
    *[m for m in _COMPONENT_METHODS_FULL if "SUPER" not in m],
    "SAPT2+3(CCD)DMP2 ELST ENERGY",
    "SAPT2+3(CCD)DMP2 EXCH ENERGY",
    "SAPT2+3(CCD)DMP2 IND ENERGY",
    "SAPT2+3(CCD)DMP2 DISP ENERGY",
]


def _prepare_component_violin_dfs(
    df,
    bases,
    limit_to_column_not_nan=None,
    subset_only=False,
):
    if subset_only:
        df = df[df["subset"]].copy()
        print(f"Subset: {len(df)}")
    if limit_to_column_not_nan is not None:
        size_prior = len(df)
        df = df[df[limit_to_column_not_nan].notna()].copy()
        print(
            f"Limiting to {limit_to_column_not_nan} not NaN: {size_prior} -> {len(df)}"
        )

    for functional in ("pbe0", "b3lyp", "b2plyp", "wb97x"):
        for basis in bases:
            df = prep_saptdft_components(df, functional, basis)

    for basis in bases:
        source_basis = "adz" if basis == "adz" else "atz"
        df[f"SAPT0-D4 DISP ENERGY {basis}"] = (
            df[f"-D4 (SAPT0_{source_basis}_3_IE)"] / h2kcalmol
        )

    ref_basis = "aqz" if "aqz" in bases else "atz"
    methods = (
        _COMPONENT_METHODS_SUBSET.copy()
        if "aqz" in bases
        else _COMPONENT_METHODS_FULL.copy()
    )
    methods_with_refs = methods + list(_COMPONENT_SAPT_REFERENCE.values())

    copy_cols_start = [
        "DB",
        "system_id",
        "benchmark ref energy",
        "E_R_eq",
        "R",
        f"Ref_elst_{ref_basis}",
        f"E_R_eq_elst_{ref_basis}",
        f"Ref_exch_{ref_basis}",
        f"E_R_eq_exch_{ref_basis}",
        f"Ref_ind_{ref_basis}",
        f"E_R_eq_ind_{ref_basis}",
        f"Ref_disp_{ref_basis}",
        f"E_R_eq_disp_{ref_basis}",
    ]

    df_by_basis = {}
    for basis in bases:
        copy_cols = copy_cols_start.copy()
        copy_cols.extend([f"{c} {basis}" for c in methods_with_refs])
        df_basis = df[copy_cols].copy()
        df_basis.columns = [c.replace(f" {basis}", "") for c in df_basis.columns]
        df_by_basis[basis] = df_basis

    for basis in bases:
        df_by_basis[basis] = sapt_error_comp(
            df_by_basis[basis],
            df_by_basis[ref_basis],
            _COMPONENT_SAPT_REFERENCE,
            methods_with_refs,
        )

    basis_labels = {
        "adz": "aug-cc-pVDZ",
        "atz": "aug-cc-pVTZ",
        "aqz": "aug-cc-pVQZ",
    }
    ylim = (
        [[-3, 3] for _ in range(4)] if "aqz" in bases else [[-2, 2] for _ in range(4)]
    )
    return [
        {
            "df": df_by_basis[basis],
            "basis": basis_labels[basis],
            "label": basis_labels[basis],
            "ylim": ylim,
            "name": basis,
        }
        for basis in bases
    ]


def violin_plots_multi_components(
    df,
    limit_to_column_not_nan=None,
    slide=False,
    dfs=None,
):
    if dfs is None:
        dfs = _prepare_component_violin_dfs(
            df,
            bases=("adz", "atz"),
            limit_to_column_not_nan=limit_to_column_not_nan,
            subset_only=False,
        )
    mcures_labels_start = [
        {
            "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
        },
        {
            "SAPT2,SAPT2+": "SAPT2 ELST ENERGY Error",
        },
        {
            "SAPT2+(3),SAPT2+3": "SAPT2+(3) ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] ELST ENERGY Error",
        },
        {
            "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] dMP2 EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] EXCH ENERGY Error",
        },
        {
            "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 EXCH ENERGY Error",
        },
        {
            "SAPT0": "SAPT0 IND ENERGY Error",
        },
        # {
        #     "sSAPT0": "SSAPT0 IND ENERGY Error",
        # },
        {
            "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 IND ENERGY Error",
        },
        {
            "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
        },
        {
            "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        {
            "SAPT2+3": "SAPT2+3 IND ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] IND ENERGY Error",
        },
        # {
        #     "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] dMP2 IND ENERGY Error",
        # },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] IND ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] dMP2 IND ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error",
        },
        {
            "SAPT0,SAPT2": "SAPT0 DISP ENERGY Error",
        },
        # {
        #     "sSAPT0": "SSAPT0 DISP ENERGY Error",
        # },
        {
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
        },
        {
            "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
        },
        {
            "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
        },
        {
            "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] DISP ENERGY Error",
        },
        {
            "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
        },
        {
            "PBE0-D3(SAPT)": "PBE0-D3 DISP ENERGY Error",
        },
        {
            "B3LYP-D3(SAPT)": "B3LYP-D3 DISP ENERGY Error",
        },
        {
            "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
        },
        {
            "B3LYP-D4(SAPT)": "B3LYP-D4 dMP2 DISP ENERGY Error",
        },
        {
            "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
        },
        {
            "B2PLYP-D4": "B2PLYP-D4 dMP2 DISP ENERGY Error",
        },
        {
            "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
        },
        {
            "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
        },
        {
            "D3-ML": "D3-ML DISP ENERGY Error",
        },
        {
            "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error",
        },
        {
            "SAPT(DFT)-D4": "SAPT(DFT)-D4 DISP ENERGY Error",
        },
        {
            "SAPT(PBE0)-D4(INTER)": "SAPT(PBE0)-D4 INTER DISP ENERGY Error",
        },
        {
            "SAPT(B3LYP)-D4(INTER)": "SAPT(B3LYP)-D4 INTER DISP ENERGY Error",
        },
        {
            "SAPT(PBE0)-D3(I)": "SAPT(PBE0)-D3 INTER DISP ENERGY Error",
        },
        {
            "SAPT(B3LYP)-D3(I)": "SAPT(B3LYP)-D3 INTER DISP ENERGY Error",
        },
        {
            "SAPT(PBE0)-D3(S)": "SAPT(PBE0)-D3 SUPER DISP ENERGY Error",
        },
        {
            "SAPT(B3LYP)-D3(S)": "SAPT(B3LYP)-D3 SUPER DISP ENERGY Error",
        },
    ]
    mcure_labels = {
        "ELST": {},
        "EXCH": {},
        "IND": {},
        "DISP": {},
    }
    # TODO DEBUG
    # df_atz["SAPT2+3(CCD)DMP2 EXCH ENERGY"] *= h2kcalmol
    debug = False
    for i in mcures_labels_start:
        k, v = list(i.items())[0]
        for d in dfs:
            if "ELST" in v:
                if k not in mcure_labels["ELST"]:
                    mcure_labels["ELST"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_ELST"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_elst_atz",
                        benchmark_col_system=f"E_R_eq_elst_atz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_ELST"].abs().mean() * 100
                mcure_labels["ELST"][k].append(mcure)
            elif "EXCH" in v:
                if k not in mcure_labels["EXCH"]:
                    mcure_labels["EXCH"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_EXCH"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_exch_atz",
                        benchmark_col_system=f"E_R_eq_exch_atz",
                        debug=debug,
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_EXCH"].abs().mean() * 100
                mcure_labels["EXCH"][k].append(mcure)
            elif "IND" in v:
                if k not in mcure_labels["IND"]:
                    mcure_labels["IND"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_IND"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_ind_atz",
                        benchmark_col_system=f"E_R_eq_ind_atz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_IND"].abs().mean() * 100
                mcure_labels["IND"][k].append(mcure)
            elif "DISP" in v:
                if k not in mcure_labels["DISP"]:
                    mcure_labels["DISP"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_DISP"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_disp_atz",
                        benchmark_col_system=f"E_R_eq_disp_atz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_DISP"].abs().mean() * 100
                mcure_labels["DISP"][k].append(mcure)
            else:
                raise ValueError(f"Error: {v = }")
    pp(mcure_labels)

    import cdsg_plot

    if slide:
        fig_size = (10, 4)
        grid_heights = [
            0.9,
            2,
            # 0.7,
            # 2,
        ]
        table_fontsize = 16
        x_label_fontsize = 16
        y_label_fontsize = 18
        extra_label = "_slide"
        output_filename = f"./plots/individual_components.jpg"
        dfs = dfs[:1]
    else:
        fig_size = (9, 7)
        grid_heights = [
            0.85,
            2,
            0.65,
            2,
        ]
        table_fontsize = 12
        x_label_fontsize = 12
        y_label_fontsize = 12
        extra_label = ""
        output_filename = f"./plots/LoS_components_adz_atz_nondisp{extra_label}.jpg"

    print(f"Plotting Extra label: {extra_label}")
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_elst={
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] ELST ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] ELST ENERGY Error",
            "SAPT0": "SAPT0 ELST ENERGY Error",
            "SAPT2+3": "SAPT2+(3) ELST ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] ELST ENERGY Error",
            # "SAPT(WB97X)": "SAPT(DFT) [WB97X] ELST ENERGY Error",
        },
        df_labels_and_columns_exch={
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] EXCH ENERGY Error",
            "SAPT0": "SAPT0 EXCH ENERGY Error",
            "SAPT2+3": "SAPT2 EXCH ENERGY Error",
            # "SAPT(B3LYP) dMP2": "SAPT(DFT) [B3LYP] dMP2 EXCH ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error",
            # "SAPT(WB97X)": "SAPT(DFT) [WB97X] EXCH ENERGY Error",
        },
        df_labels_and_columns_indu={
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] IND ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] IND ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] IND ENERGY Error",
            # "SAPT(WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error",
            "SAPT0": "SAPT0 IND ENERGY Error",
            # "sSAPT0": "SSAPT0 IND ENERGY Error",
            "SAPT2+3": "SAPT2+3 IND ENERGY Error",
            # should be here but moved for dense plotting...
            # "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
            # "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
            # "SAPT2+3": "SAPT2 IND ENERGY Error",
            # "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
        },
        colors=colors_comps_saptdftd4,
        violin_alphas=0.9,
        usetex=True,
        legend_loc="lower right",
        table_fontsize=table_fontsize,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        share_y_axis=True,
        figure_size=fig_size,
        grid_heights=grid_heights,
        grid_widths=[5, 5, 6],
        output_filename=output_filename,
        # mcure=mcure_labels,
    )
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_disp={
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] DISP ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
            # "SAPT(WB97X)": "SAPT(DFT) [WB97X] DISP ENERGY Error",
            # "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error",
            "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
            "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
            "PBE0-D3(SAPT)": "PBE0-D3 DISP ENERGY Error",
            "B3LYP-D3(SAPT)": "B3LYP-D3 DISP ENERGY Error",
            "SAPT(PBE0)-D4(S)": "SAPT(DFT)-D4 DISP ENERGY Error",
            "SAPT(PBE0)-D4(I)": "SAPT(PBE0)-D4 INTER DISP ENERGY Error",
            "SAPT(B3LYP)-D4(I)": "SAPT(B3LYP)-D4 INTER DISP ENERGY Error",
            "SAPT(PBE0)-D3(I)": "SAPT(PBE0)-D3 INTER DISP ENERGY Error",
            "SAPT(B3LYP)-D3(I)": "SAPT(B3LYP)-D3 INTER DISP ENERGY Error",
            # "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
            # "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
            "D3-ML": "D3-ML DISP ENERGY Error",
            "SAPT0": "SAPT0 DISP ENERGY Error",
            "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
            # "sSAPT0": "SSAPT0 DISP ENERGY Error",
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
            # "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
            "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
            # "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
            # "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
            "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        colors=colors_disp_saptdftd4,
        usetex=True,
        legend_loc="lower right",
        table_fontsize=table_fontsize,
        x_label_fontsize=x_label_fontsize,
        violin_alphas=0.9,
        y_label_fontsize=11,
        figure_size=fig_size,
        grid_heights=grid_heights,
        grid_widths=[1.0],
        output_filename=f"./plots/LoS_components_adz_atz_disp{extra_label}.jpg",
        # mcure=mcure_labels,
    )
    return


def dmp2_correlation_plots(df, limit_to_column_not_nan=None, slide=True):
    if limit_to_column_not_nan is not None:
        size_prior = len(df)
        df = df[df[limit_to_column_not_nan].notna()].copy()
        print(
            f"Limiting to {limit_to_column_not_nan} not NaN: {size_prior} -> {len(df)}"
        )
    df = prep_saptdft_components(df, "pbe0", "adz")
    df = prep_saptdft_components(df, "pbe0", "atz")
    df = prep_saptdft_components(df, "b3lyp", "adz")
    df = prep_saptdft_components(df, "b3lyp", "atz")
    df = prep_saptdft_components(df, "b2plyp", "adz")
    df = prep_saptdft_components(df, "b2plyp", "atz")
    df = prep_saptdft_components(df, "wb97x", "adz")
    df = prep_saptdft_components(df, "wb97x", "atz")
    df["SAPT0-D4 DISP ENERGY adz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 DISP ENERGY atz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 DISP ENERGY aqz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    sapt_methods = [
        "SAPT0 ELST ENERGY",
        "SAPT2 ELST ENERGY",
        "SAPT2+(3) ELST ENERGY",
        "SAPT(DFT) [PBE0] ELST ENERGY",
        "SAPT(DFT) [B3LYP] ELST ENERGY",
        "SAPT(DFT) [B2PLYP] ELST ENERGY",
        "SAPT(DFT) [WB97X] ELST ENERGY",
        "SAPT0 EXCH ENERGY",
        "SAPT2 EXCH ENERGY",
        "SAPT(DFT) [PBE0] EXCH ENERGY",
        "SAPT(DFT) [B3LYP] EXCH ENERGY",
        "SAPT(DFT) [B2PLYP] EXCH ENERGY",
        "SAPT(DFT) [WB97X] EXCH ENERGY",
        "SAPT0 IND ENERGY",
        # "SSAPT0 IND ENERGY",
        "SAPT2 IND ENERGY",
        "SAPT2+DMP2 IND ENERGY",
        "SAPT2+3DMP2 IND ENERGY",
        "SAPT(DFT) [PBE0] IND ENERGY",
        "SAPT(DFT) [B3LYP] IND ENERGY",
        "SAPT(DFT) [B2PLYP] IND ENERGY",
        "SAPT(DFT) [WB97X] IND ENERGY",
        "SAPT0 DISP ENERGY",
        # "SSAPT0 DISP ENERGY",
        "SAPT2+ DISP ENERGY",
        "SAPT2+(3) DISP ENERGY",
        "SAPT2+3 DISP ENERGY",
        "SAPT2+(CCD) DISP ENERGY",
        "SAPT2+(3)(CCD) DISP ENERGY",
        "SAPT2+3(CCD) DISP ENERGY",
        # local disp
        "SAPT0-D4 DISP ENERGY",
        "PBE0-D4 DISP ENERGY",
        "PBE0-D3 DISP ENERGY",
        "B3LYP-D4 DISP ENERGY",
        "B2PLYP-D4 DISP ENERGY",
        "WB97X-D4 DISP ENERGY",
        "SAPT(DFT) [PBE0] DISP ENERGY",
        "SAPT(DFT) [B3LYP] DISP ENERGY",
        "SAPT(DFT) [B2PLYP] DISP ENERGY",
        "SAPT(DFT) [WB97X] DISP ENERGY",
        "D3-ML DISP ENERGY",
        "SAPT(DFT)+D4 DISP ENERGY",
        "SAPT(DFT)-D4 DISP ENERGY",
    ]
    sapt_reference = {
        "ELST": "SAPT2+3(CCD)DMP2 ELST ENERGY",
        "EXCH": "SAPT2+3(CCD)DMP2 EXCH ENERGY",
        "IND": "SAPT2+3(CCD)DMP2 IND ENERGY",
        "DISP": "SAPT2+3(CCD)DMP2 DISP ENERGY",
    }
    reference = "benchmark ref energy"
    copy_cols_start = [
        "DB",
        "system_id",
        "benchmark ref energy",
        "E_R_eq",
        "R",
        "Ref_elst_atz",
        "E_R_eq_elst_atz",
        "Ref_exch_atz",
        "E_R_eq_exch_atz",
        "Ref_ind_atz",
        "E_R_eq_ind_atz",
        "Ref_disp_atz",
        "E_R_eq_disp_atz",
    ]
    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} adz" for c in sapt_methods])
    copy_cols.extend([f"{c} adz" for c in sapt_reference.values()])
    df_adz = df[copy_cols].copy()
    df_adz.columns = [c.replace(" adz", "") for c in df_adz.columns]

    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} atz" for c in sapt_methods])
    copy_cols.extend([f"{c} atz" for c in sapt_reference.values()])
    df_atz = df[copy_cols].copy()
    df_atz.columns = [c.replace(" atz", "") for c in df_atz.columns]

    df_adz = sapt_error_comp(df_adz, df_atz, sapt_reference, sapt_methods)
    df_atz = sapt_error_comp(df_atz, df_atz, sapt_reference, sapt_methods)

    adz_ylims = [
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
    ]
    atz_ylims = [
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
    ]

    dfs = [
        {
            "df": df_adz,
            "basis": "aug-cc-pVDZ",
            "label": "aug-cc-pVDZ",
            "ylim": adz_ylims,
        },
        {
            "df": df_atz,
            "basis": "aug-cc-pVTZ",
            "label": "aug-cc-pVTZ",
            "ylim": atz_ylims,
        },
    ]
    return


def violin_plots_multi_components_df_individual(df, limit_to_column_not_nan=None):
    if limit_to_column_not_nan is not None:
        size_prior = len(df)
        df = df[df[limit_to_column_not_nan].notna()].copy()
        print(
            f"Limiting to {limit_to_column_not_nan} not NaN: {size_prior} -> {len(df)}"
        )
    df = prep_saptdft_components(df, "pbe0", "adz")
    df = prep_saptdft_components(df, "pbe0", "atz")
    df = prep_saptdft_components(df, "b3lyp", "adz")
    df = prep_saptdft_components(df, "b3lyp", "atz")
    df = prep_saptdft_components(df, "b2plyp", "adz")
    df = prep_saptdft_components(df, "b2plyp", "atz")
    df = prep_saptdft_components(df, "wb97x", "adz")
    df = prep_saptdft_components(df, "wb97x", "atz")

    df["SAPT0-D4 DISP ENERGY adz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 DISP ENERGY atz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 DISP ENERGY aqz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol

    sapt_methods = [
        "SAPT0 ELST ENERGY",
        "SAPT2 ELST ENERGY",
        "SAPT2+(3) ELST ENERGY",
        "SAPT(DFT) [PBE0] ELST ENERGY",
        "SAPT(DFT) [B3LYP] ELST ENERGY",
        "SAPT(DFT) [B2PLYP] ELST ENERGY",
        "SAPT(DFT) [WB97X] ELST ENERGY",
        "SAPT0 EXCH ENERGY",
        "SAPT2 EXCH ENERGY",
        "SAPT(DFT) [PBE0] EXCH ENERGY",
        "SAPT(DFT) [B3LYP] EXCH ENERGY",
        "SAPT(DFT) [B3LYP] dMP2 EXCH ENERGY",
        "SAPT(DFT) [B2PLYP] EXCH ENERGY",
        "SAPT(DFT) [WB97X] EXCH ENERGY",
        "SAPT0 IND ENERGY",
        # "SSAPT0 IND ENERGY",
        "SAPT2 IND ENERGY",
        "SAPT2+DMP2 IND ENERGY",
        "SAPT2+3 IND ENERGY",
        "SAPT2+3DMP2 IND ENERGY",
        "SAPT(DFT) [PBE0] IND ENERGY",
        "SAPT(DFT) [B3LYP] IND ENERGY",
        # "SAPT(DFT) [B3LYP] dMP2 IND ENERGY",
        "SAPT(DFT) [B2PLYP] IND ENERGY",
        "SAPT(DFT) [B2PLYP] dMP2 IND ENERGY",
        "SAPT(DFT) [WB97X] IND ENERGY",
        "SAPT0 DISP ENERGY",
        # "SSAPT0 DISP ENERGY",
        "SAPT2+ DISP ENERGY",
        "SAPT2+(3) DISP ENERGY",
        "SAPT2+3 DISP ENERGY",
        "SAPT2+(CCD) DISP ENERGY",
        "SAPT2+(3)(CCD) DISP ENERGY",
        # "SAPT2+3(CCD) DISP ENERGY",
        # local disp
        "SAPT0-D4 DISP ENERGY",
        "PBE0-D4 DISP ENERGY",
        "PBE0-D3 DISP ENERGY",
        "B3LYP-D3 DISP ENERGY",
        "B3LYP-D4 DISP ENERGY",
        "B3LYP-D4 dMP2 DISP ENERGY",
        "B2PLYP-D4 DISP ENERGY",
        "B2PLYP-D4 dMP2 DISP ENERGY",
        "WB97X-D4 DISP ENERGY",
        "SAPT(DFT) [PBE0] DISP ENERGY",
        "SAPT(DFT) [B3LYP] DISP ENERGY",
        "SAPT(DFT) [B2PLYP] DISP ENERGY",
        "SAPT(DFT) [WB97X] DISP ENERGY",
        "D3-ML DISP ENERGY",
        "SAPT(DFT)+D4 DISP ENERGY",
        "SAPT(DFT)-D4 DISP ENERGY",
        "SAPT(PBE0)-D4 INTER DISP ENERGY",
        "SAPT(B3LYP)-D4 INTER DISP ENERGY",
        "SAPT(PBE0)-D3 INTER DISP ENERGY",
        "SAPT(B3LYP)-D3 INTER DISP ENERGY",
        "SAPT(PBE0)-D3 SUPER DISP ENERGY",
        "SAPT(B3LYP)-D3 SUPER DISP ENERGY",
        "SAPT0-D4 (I) DISP ENERGY",
        "SAPT0-D4 (I) TOTAL ENERGY",
        "SAPT0 TOTAL ENERGY",
        "SAPT2+3(CCD) TOTAL ENERGY",
        "SAPT2+DMP2 TOTAL ENERGY",
        "SAPT2+(CCD)DMP2 TOTAL ENERGY",
        "SAPT2+3(CCD)DMP2 TOTAL ENERGY",
        "SAPT0-D4 TOTAL ENERGY",
        "SAPT TOTAL ENERGY",
        "SAPT(DFT)D3-ML TOTAL ENERGY",
        "SAPT(B3LYP)D3-ML TOTAL ENERGY",
        "SAPT(DFT)-D4 TOTAL ENERGY",
        "SAPT(DFT)+D4 TOTAL ENERGY",
        "SAPT(PBE0)-D4 INTER TOTAL ENERGY",
        "SAPT(B3LYP)-D4 INTER TOTAL ENERGY",
        "SAPT(PBE0)-D3 INTER TOTAL ENERGY",
        "SAPT(B3LYP)-D3 INTER TOTAL ENERGY",
        "SAPT(PBE0)-D3 SUPER TOTAL ENERGY",
        "SAPT(B3LYP)-D3 SUPER TOTAL ENERGY",
    ]
    sapt_reference = {
        # "ELST": "SAPT2+3(CCD)DMP2 ELST ENERGY",
        # "EXCH": "SAPT2+3(CCD)DMP2 EXCH ENERGY",
        # "IND": "SAPT2+3(CCD)DMP2 IND ENERGY",
        # "DISP": "SAPT2+3(CCD)DMP2 DISP ENERGY",
        "ELST": "SAPT2+3(CCD) ELST ENERGY",
        "EXCH": "SAPT2+3(CCD) EXCH ENERGY",
        "IND": "SAPT2+3(CCD) IND ENERGY",
        "DISP": "SAPT2+3(CCD) DISP ENERGY",
    }

    copy_cols_start = [
        "DB",
        "system_id",
        "benchmark ref energy",
        "E_R_eq",
        "R",
        "Ref_elst_atz",
        "E_R_eq_elst_atz",
        "Ref_exch_atz",
        "E_R_eq_exch_atz",
        "Ref_ind_atz",
        "E_R_eq_ind_atz",
        "Ref_disp_atz",
        "E_R_eq_disp_atz",
        # local methods
        ## saptdft
        # "SAPT_DFT_pbe0_adz_elst",
        # "SAPT_DFT_pbe0_adz_exch",
        # "SAPT_DFT_pbe0_adz_ind",
        # "SAPT_DFT_pbe0_adz_disp",
        # "SAPT_DFT_pbe0d4_adz_disp",
        # "SAPT0-D4 adz",
        # "SAPT_DFT_pbe0_atz_elst",
        # "SAPT_DFT_pbe0_atz_exch",
        # "SAPT_DFT_pbe0_atz_ind",
        # "SAPT_DFT_pbe0_atz_disp",
        # "SAPT_DFT_pbe0d4_atz_disp",
        # "SAPT0-D4 atz",
        "SAPT_DFT_D4_pbe0_adz_total",
        "SAPT_DFT_D4_pbe0_atz_total",
        "SAPT_DFT_pbe0_adz_total",
        "SAPT_DFT_pbe0_atz_total",
        "SAPT_DFT_D4_b3lyp_adz_total",
        "SAPT_DFT_D4_b3lyp_atz_total",
        "SAPT_DFT_b3lyp_adz_total",
        "SAPT_DFT_b3lyp_atz_total",
        "PBE0-D3 TOTAL ENERGY adz",
        "PBE0-D3 TOTAL ENERGY atz",
        "B3LYP-D3 TOTAL ENERGY adz",
        "B3LYP-D3 TOTAL ENERGY atz",
    ]
    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} adz" for c in sapt_methods])
    copy_cols.extend([f"{c} adz" for c in sapt_reference.values()])
    print("copying cols:")
    pp(copy_cols)
    df_adz = df[copy_cols].copy()
    df_adz.columns = [c.replace(" adz", "") for c in df_adz.columns]

    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} atz" for c in sapt_methods])
    copy_cols.extend([f"{c} atz" for c in sapt_reference.values()])
    df_atz = df[copy_cols].copy()
    df_atz.columns = [c.replace(" atz", "") for c in df_atz.columns]

    sapt_methods.extend(sapt_reference.values())

    df_adz = sapt_error_comp(df_adz, df_atz, sapt_reference, sapt_methods)
    df_atz = sapt_error_comp(df_atz, df_atz, sapt_reference, sapt_methods)

    df_adz = df_adz.rename(
        columns={
            "PBE0-D3 TOTAL ENERGY adz": "PBE0-D3 TOTAL ENERGY",
            "B3LYP-D3 TOTAL ENERGY adz": "B3LYP-D3 TOTAL ENERGY",
            "SAPT_DFT_D4_pbe0_adz_total": "PBE0-D4 TOTAL ENERGY",
            "SAPT_DFT_pbe0_adz_total": "SAPT(DFT) [PBE0] TOTAL ENERGY",
            "SAPT_DFT_D4_b3lyp_adz_total": "B3LYP-D4 TOTAL ENERGY",
            "SAPT_DFT_b3lyp_adz_total": "SAPT(DFT) [B3LYP] TOTAL ENERGY",
        },
        # inplace=True,
    )
    df_atz = df_atz.rename(
        columns={
            "PBE0-D3 TOTAL ENERGY atz": "PBE0-D3 TOTAL ENERGY",
            "B3LYP-D3 TOTAL ENERGY atz": "B3LYP-D3 TOTAL ENERGY",
            "SAPT_DFT_D4_pbe0_atz_total": "PBE0-D4 TOTAL ENERGY",
            "SAPT_DFT_pbe0_atz_total": "SAPT(DFT) [PBE0] TOTAL ENERGY",
            "SAPT_DFT_D4_b3lyp_atz_total": "B3LYP-D4 TOTAL ENERGY",
            "SAPT_DFT_b3lyp_atz_total": "SAPT(DFT) [B3LYP] TOTAL ENERGY",
        },
        # inplace=True,
    )

    local_methods = [
        "PBE0-D4 TOTAL ENERGY",
        "B3LYP-D4 TOTAL ENERGY",
        "PBE0-D3 TOTAL ENERGY",
        "B3LYP-D3 TOTAL ENERGY",
        "SAPT(DFT) [PBE0] TOTAL ENERGY",
        "SAPT(DFT) [B3LYP] TOTAL ENERGY",
    ]
    reference = "benchmark ref energy"
    pp(df_adz.columns.to_list())
    for i in local_methods:
        df_adz[f"{i} Error"] = df_adz[i] * h2kcalmol - df_adz[reference]
        df_atz[f"{i} Error"] = df_atz[i] * h2kcalmol - df_atz[reference]

    adz_ylims = [
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
    ]
    atz_ylims = [
        [-2, 2],
        [-2, 2],
        [-2, 2],
        [-2, 2],
    ]

    dfs_all = [
        {
            "df": df_adz,
            "basis": "aug-cc-pVDZ",
            "label": "aug-cc-pVDZ",
            "ylim": adz_ylims,
            "name": "adz",
        },
        {
            "df": df_atz,
            "basis": "aug-cc-pVTZ",
            "label": "aug-cc-pVTZ",
            "ylim": atz_ylims,
            "name": "atz",
        },
    ]

    import cdsg_plot

    colors_disp = [
        [
            BLUE,
            GREEN,
            BLUE,
            GREEN,
            BLUE,
            TEAL,
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
            BLUE,
            BLUE,
        ]
    ]
    colors_total = [
        [
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
            BLUE,
            GREEN,
            BLUE,
            GREEN,
            BLUE,
            GREEN,
            BLUE,
            GREEN,
            BLUE,
            GREEN,
        ]
    ]
    colors_comp = [
        [
            BLUE,
            GREEN,
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
        ],
        [
            BLUE,
            GREEN,
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
        ],
        [
            BLUE,
            GREEN,
            BLUE,
            TEAL,
            # LIGHT_BLUE,
            Medium_Sea_Green,
            # INDIGO,
        ],
    ]
    # print(df[[]])

    # TOTAL
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs_all,
        df_labels_and_columns_elst={},
        df_labels_and_columns_exch={},
        df_labels_and_columns_indu={},
        df_labels_and_columns_disp={},
        df_labels_and_columns_total={
            "SAPT0": "SAPT0 TOTAL ENERGY Error",
            "SAPT0-D4(S)": "SAPT0-D4 TOTAL ENERGY Error",
            "SAPT0-D4(I)": "SAPT0-D4 (I) TOTAL ENERGY Error",
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] TOTAL ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] TOTAL ENERGY Error",
            "PBE0-D4": "PBE0-D4 TOTAL ENERGY Error",
            "B3LYP-D4": "B3LYP-D4 TOTAL ENERGY Error",
            "PBE0-D3": "PBE0-D3 TOTAL ENERGY Error",
            "B3LYP-D3": "B3LYP-D3 TOTAL ENERGY Error",
            "SAPT(PBE0)-D4(S)": "SAPT(DFT)-D4 TOTAL ENERGY Error",
            "SAPT(PBE0)-D3(S)": "SAPT(PBE0)-D3 SUPER TOTAL ENERGY Error",
            "SAPT(B3LYP)-D3(S)": "SAPT(B3LYP)-D3 SUPER TOTAL ENERGY Error",
            "SAPT(PBE0)-D4(I)": "SAPT(PBE0)-D4 INTER TOTAL ENERGY Error",
            "SAPT(PBE0)-D3(I)": "SAPT(PBE0)-D3 INTER TOTAL ENERGY Error",
            "SAPT(B3LYP)-D3(I)": "SAPT(B3LYP)-D3 INTER TOTAL ENERGY Error",
            # "SAPT(PBE0)\\\\D3-ML": "SAPT(DFT)D3-ML TOTAL ENERGY Error",
            "SAPT2+3\\\\(CCD)": "SAPT2+3(CCD) TOTAL ENERGY Error",
        },
        colors=[
            [
                # PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,'orange',GREY,
                # PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,'orange',GREY,
                PURPLE,
                PURPLE,
                PURPLE,
                TEAL,
                Medium_Sea_Green,
                TEAL,
                Medium_Sea_Green,
                TEAL,
                Medium_Sea_Green,
                TEAL,
                TEAL,  # SAPT(PBE0)-D3 (S)
                Medium_Sea_Green,
                TEAL,  # SAPT(PBE0)-D3 (I)
                TEAL,  # SAPT(PBE0)-D3 (I)
                Medium_Sea_Green,
                # "orange",
                GREY,
                Medium_Sea_Green,
                PURPLE,
                TEAL,
                Medium_Sea_Green,
            ]
        ],
        output_filename=f"./plots/individuals/LoS_total_saptdft_extended_pres.jpg",
        # f"./plots/individuals/LoS_total_saptdft_extended_pres_violin.jpg",
        usetex=True,
        legend_loc=None,
        figure_size=(10, 5),
        grid_widths=[1.0],
        grid_heights=[
            0.08,
            1,
            0.08,
            1,
        ],
        table_fontsize=14,
        x_label_fontsize=13,
        y_label_fontsize=13,
        title_fontsize=16,
        mcure=None,
        MAE="textbf",
        RMSE=False,
        MinE=False,
        MaxE=False,
        annotations_texty=0.0,
        share_y_axis=True,
        wspace=0.05,
        table_delimiter=",",
        gridlines_linewidths=1.5,
        violin_alphas=1.0,
        quantile_linewidth=1.2,
        pm_alpha=0.5,
        zero_alpha=1.0,
        hide_ytick_label_edges=True,
        # mcure=mcure_labels,
    )

    # DIPS Components
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs_all,
        df_labels_and_columns_elst={},
        df_labels_and_columns_exch={},
        df_labels_and_columns_indu={},
        df_labels_and_columns_disp={
            "SAPT0": "SAPT0 DISP ENERGY Error",
            # "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
            # "SAPT(DFT)\\\\[B2PLYP]": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
            # "SAPT(DFT)\\\\[WB97X]": "SAPT(DFT) [WB97X] DISP ENERGY Error",
            "SAPT0-D4(S)": "SAPT0-D4 DISP ENERGY Error",
            "SAPT0-D4(I)": "SAPT0-D4 (I) DISP ENERGY Error",
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] DISP ENERGY Error",
            "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
            "PBE0-D3(SAPT)": "PBE0-D3 DISP ENERGY Error",
            "B3LYP-D3(SAPT)": "B3LYP-D3 DISP ENERGY Error",
            "SAPT(PBE0)-D4(S)": "SAPT(DFT)-D4 DISP ENERGY Error",
            "SAPT(PBE0)-D4(I)": "SAPT(PBE0)-D4 INTER DISP ENERGY Error",
            "SAPT(PBE0)-D3(S)": "SAPT(PBE0)-D3 SUPER DISP ENERGY Error",
            "SAPT(PBE0)-D3(I)": "SAPT(PBE0)-D3 INTER DISP ENERGY Error",
            # "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
            # "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
            # "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
            # "D3-ML": "D3-ML DISP ENERGY Error",
            # "SAPT2+3\\\\(CCD)DMP2": "SAPT2+3(CCD) DISP ENERGY Error",
            "SAPT2+3\\\\(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        df_labels_and_columns_total={
            # "SAPT0": "SAPT0 TOTAL ENERGY Error",
            # "SAPT(DFT)\\\\[PBE0]": "SAPT(DFT) [PBE0] TOTAL ENERGY Error",
            # "SAPT(DFT)\\\\[B3LYP]": "SAPT(DFT) [B3LYP] TOTAL ENERGY Error",
            # "SAPT(DFT)\\\\[B2PLYP]": "SAPT(DFT) [B2PLYP] TOTAL ENERGY Error",
            # "SAPT(DFT)\\\\[WB97X]": "SAPT(DFT) [WB97X] TOTAL ENERGY Error",
            # "SAPT0-D4": "SAPT0-D4 TOTAL ENERGY Error",
            # "PBE0-D4(SAPT)": "PBE0-D4 TOTAL ENERGY Error",
            # "B3LYP-D4(SAPT)": "B3LYP-D4 TOTAL ENERGY Error",
            # "B2PLYP-D4": "B2PLYP-D4 TOTAL ENERGY Error",
            # "WB97X-D4": "WB97X-D4 TOTAL ENERGY Error",
            # "SAPT2+3\\\\(CCD)DMP2": "SAPT2+3(CCD)DMP2 TOTAL ENERGY Error",
        },
        colors=[
            [
                # PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,"orange",GREY,
                # PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,PURPLE,TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,"orange",GREY,
                PURPLE,
                PURPLE,
                PURPLE,
                TEAL,
                TEAL,
                TEAL,
                TEAL,
                TEAL,  # SAPT(PBE0)-D3 (S)
                TEAL,  # SAPT(PBE0)-D3 (I)
                GREY,
            ]
        ],
        output_filename=f"./plots/individuals/LoS_total_saptdft_extended_disp_pres.jpg",
        usetex=True,
        legend_loc=None,
        figure_size=(10, 5),
        grid_widths=[1.0],
        grid_heights=[
            0.08,
            1,
            0.08,
            1,
        ],
        table_fontsize=14,
        x_label_fontsize=13,
        y_label_fontsize=13,
        title_fontsize=16,
        mcure=None,
        MAE="textbf",
        RMSE=False,
        MinE=False,
        MaxE=False,
        annotations_texty=0.0,
        share_y_axis=True,
        wspace=0.05,
        table_delimiter=",",
        gridlines_linewidths=1.5,
        violin_alphas=1.0,
        quantile_linewidth=1.2,
        pm_alpha=0.5,
        zero_alpha=1.0,
        hide_ytick_label_edges=True,
        # mcure=mcure_labels,
    )

    return
    # SAPT0-D4 vs. SAPT(DFT)-D4
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        [dfs_all[0]],
        df_labels_and_columns_elst={},
        df_labels_and_columns_exch={},
        df_labels_and_columns_indu={},
        df_labels_and_columns_disp={
            "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
            "SAPT(PBE0)-D4": "SAPT(DFT)-D4 DISP ENERGY Error",
        },
        df_labels_and_columns_total={
            "SAPT0-D4": "SAPT0-D4 TOTAL ENERGY Error",
            "SAPT(PBE0)-D4": "SAPT(DFT)-D4 TOTAL ENERGY Error",
        },
        colors=[
            [PURPLE, TEAL],
            [PURPLE, TEAL],
        ],
        output_filename=f"./plots/individuals/LoS_components_sapt0_vs_saptdft_pres.jpg",
        usetex=True,
        legend_loc=None,
        figure_size=(10, 3),
        grid_widths=[1.0, 1.0],
        grid_heights=[
            0.10,
            1,
        ],
        table_fontsize=15,
        x_label_fontsize=14,
        y_label_fontsize=13,
        title_fontsize=16,
        mcure=None,
        MAE="textbf",
        RMSE=False,
        MinE=False,
        MaxE=False,
        annotations_texty=0.1,
        share_y_axis=True,
        wspace=0.05,
        table_delimiter=",",
        gridlines_linewidths=1.5,
        violin_alphas=1.0,
        quantile_linewidth=1.2,
        pm_alpha=0.5,
        zero_alpha=1.0,
        # mcure=mcure_labels,
    )
    print(dfs_all[0]["df"]["SAPT(DFT) [PBE0] IND ENERGY Error"])
    print(dfs_all[1]["df"]["SAPT(DFT) [PBE0] IND ENERGY Error"])

    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs_all,
        df_labels_and_columns_elst={
            "SAPT0": "SAPT0 ELST ENERGY Error",
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] ELST ENERGY Error",
            "SAPT2+3DMP2": "SAPT2+(3) ELST ENERGY Error",
            # "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] ELST ENERGY Error",
            # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] ELST ENERGY Error",
            # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] ELST ENERGY Error",
        },
        df_labels_and_columns_exch={
            "SAPT0": "SAPT0 EXCH ENERGY Error",
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
            "SAPT2+3DMP2": "SAPT2 EXCH ENERGY Error",
            # "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] EXCH ENERGY Error",
            # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error",
            # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] EXCH ENERGY Error",
        },
        df_labels_and_columns_indu={
            "SAPT0": "SAPT0 IND ENERGY Error",
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] IND ENERGY Error",
            # "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
            "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
            # "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] IND ENERGY Error",
            # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] IND ENERGY Error",
            # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error",
        },
        colors=[
            [
                PURPLE,
                TEAL,
                GREY,
            ],
            [
                PURPLE,
                TEAL,
                GREY,
            ],
            [
                PURPLE,
                TEAL,
                GREY,
            ],
        ],
        output_filename=f"./plots/individuals/LoS_components_saptdft_components_nondisp_pres.jpg",
        usetex=True,
        legend_loc=None,
        figure_size=(10, 5),
        grid_widths=[3, 3, 3],
        grid_heights=[
            0.08,
            1,
            0.08,
            1,
        ],
        table_fontsize=14,
        x_label_fontsize=13,
        y_label_fontsize=13,
        title_fontsize=16,
        mcure=None,
        MAE="textbf",
        RMSE=False,
        MinE=False,
        MaxE=False,
        annotations_texty=0.0,
        share_y_axis=True,
        wspace=0.05,
        table_delimiter=",",
        gridlines_linewidths=1.5,
        violin_alphas=1.0,
        quantile_linewidth=1.2,
        pm_alpha=0.5,
        zero_alpha=1.0,
        # mcure=mcure_labels,
    )
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        [dfs_all[0]],
        df_labels_and_columns_elst={},
        df_labels_and_columns_exch={},
        df_labels_and_columns_indu={},
        df_labels_and_columns_disp={},
        df_labels_and_columns_total={
            # "SAPT0-D4": "SAPT0-D4 TOTAL ENERGY Error",
            # "PBE0": "PBE0 IE Error",
            # "B3LYP": "B3LYP IE Error",
            # "B2PLYP": "B2PLYP IE Error",
            # "WB97X": "WB97X IE Error",
            # SAPT(DFT)
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] TOTAL ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] TOTAL ENERGY Error",
            # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] TOTAL ENERGY Error",
            # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] TOTAL ENERGY Error",
            # DFT-D4
            "PBE0-D4(SAPT)": "PBE0-D4 TOTAL ENERGY Error",
            "B3LYP-D4(SAPT)": "B3LYP-D4 TOTAL ENERGY Error",
            # "B2PLYP-D4": "B2PLYP-D4 TOTAL ENERGY Error",
            # "WB97X-D4": "WB97X-D4 TOTAL ENERGY Error",
            # SAPT(DFT) D's
            # "SAPT(DFT)-D4": "SAPT(DFT)-D4 TOTAL ENERGY Error",
            # "SAPT(DFT)+D4": "SAPT(DFT)+D4 TOTAL ENERGY Error",
            # "SAPT(DFT)D3-ML": "SAPT(DFT)D3-ML TOTAL ENERGY Error",
            # Wavefunction
            # "SAPT0": "SAPT0 TOTAL ENERGY Error",
            # "SAPT2": "SAPT2 TOTAL ENERGY Error",
            # "SAPT2+": "SAPT2+ TOTAL ENERGY Error",
            # "SAPT2+(3)": "SAPT2+(3) TOTAL ENERGY Error",
            # "SAPT2+3": "SAPT2+3 TOTAL ENERGY Error",
            # "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD)DMP2 TOTAL ENERGY Error",
        },
        colors=[
            [
                # TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,
                # TEAL,LIGHT_BLUE,Medium_Sea_Green,INDIGO,
                TEAL,
                Medium_Sea_Green,
                TEAL,
                Medium_Sea_Green,
            ]
        ],
        output_filename=f"./plots/individuals/LoS_total_saptdft_simple_pres.jpg",
        usetex=True,
        legend_loc=None,
        figure_size=(12, 4),
        grid_widths=[1.0],
        grid_heights=[
            0.10,
            1,
        ],
        table_fontsize=16,
        x_label_fontsize=16,
        y_label_fontsize=18,
        title_fontsize=16,
        mcure=None,
        MAE="textbf",
        RMSE=False,
        MinE=False,
        MaxE=False,
        annotations_texty=0.0,
        share_y_axis=True,
        wspace=0.05,
        table_delimiter=",",
        gridlines_linewidths=1.5,
        violin_alphas=1.0,
        quantile_linewidth=1.2,
        pm_alpha=0.5,
        zero_alpha=1.0,
        # mcure=mcure_labels,
    )

    for dfs in dfs_all:
        dfs = [dfs]
        cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
            dfs,
            df_labels_and_columns_elst={
                "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
                "SAPT2+(3),SAPT2+3": "SAPT2+(3) ELST ENERGY Error",
                "SAPT(PBE0)": "SAPT(DFT) [PBE0] ELST ENERGY Error",
                "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] ELST ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] ELST ENERGY Error",
                # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] ELST ENERGY Error",
            },
            df_labels_and_columns_exch={
                "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
                "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 EXCH ENERGY Error",
                "SAPT(PBE0)": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
                "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] EXCH ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error",
                # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] EXCH ENERGY Error",
            },
            df_labels_and_columns_indu={
                "SAPT0": "SAPT0 IND ENERGY Error",
                "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 IND ENERGY Error",
                "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
                "SAPT(PBE0)": "SAPT(DFT) [PBE0] IND ENERGY Error",
                "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] IND ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] IND ENERGY Error",
                # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error",
            },
            df_labels_and_columns_disp={},
            colors=colors_comp,
            output_filename=f"./plots/individuals/LoS_components_{dfs[0]['name']}_nondisp_pres.jpg",
            usetex=True,
            legend_loc="lower right",
            figure_size=(12, 3),
            grid_widths=[6, 6, 7],
            grid_heights=[
                0.10,
                2,
            ],
            table_fontsize=13,
            x_label_fontsize=14,
            y_label_fontsize=14,
            title_fontsize=14,
            mcure=None,
            MAE="textbf",
            RMSE=False,
            MinE=False,
            MaxE=False,
            annotations_texty=-0.2,
            share_y_axis=True,
            wspace=0.05,
            table_delimiter=",",
            gridlines_linewidths=1.5,
            violin_alphas=1.0,
            quantile_linewidth=1.2,
            pm_alpha=0.5,
            zero_alpha=1.0,
            # mcure=mcure_labels,
        )
        cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
            dfs,
            df_labels_and_columns_elst={},
            df_labels_and_columns_exch={},
            df_labels_and_columns_indu={},
            df_labels_and_columns_disp={
                "SAPT0,SAPT2": "SAPT0 DISP ENERGY Error",
                "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
                # "sSAPT0": "SSAPT0 DISP ENERGY Error",
                "SAPT2+": "SAPT2+ DISP ENERGY Error",
                # "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
                "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
                # "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
                # "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
                "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
                "SAPT(PBE0)-D4": "SAPT(DFT)-D4 DISP ENERGY Error",
                "SAPT(PBE0)": "SAPT(DFT) [PBE0] DISP ENERGY Error",
                "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
                # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] DISP ENERGY Error",
                "SAPT(PBE0)-D4": "SAPT(DFT)-D4 DISP ENERGY Error",
                # "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error",
                "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
                "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
                # "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
                # "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
                "D3-ML": "D3-ML DISP ENERGY Error",
            },
            colors=colors_disp,
            output_filename=f"./plots/individuals/LoS_components_{dfs[0]['name']}_disp_pres.jpg",
            usetex=True,
            legend_loc="lower right",
            figure_size=(12, 3),
            grid_widths=[1.0],
            grid_heights=[
                0.10,
                1,
            ],
            table_fontsize=16,
            x_label_fontsize=16,
            y_label_fontsize=16,
            title_fontsize=16,
            mcure=None,
            MAE="textbf",
            RMSE=False,
            MinE=False,
            MaxE=False,
            annotations_texty=-0.2,
            share_y_axis=True,
            wspace=0.05,
            table_delimiter=",",
            gridlines_linewidths=1.5,
            violin_alphas=1.0,
            quantile_linewidth=1.2,
            pm_alpha=0.5,
            zero_alpha=1.0,
            # mcure=mcure_labels,
        )
        cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
            dfs,
            df_labels_and_columns_elst={},
            df_labels_and_columns_exch={},
            df_labels_and_columns_indu={},
            df_labels_and_columns_disp={},
            df_labels_and_columns_total={
                "PBE0": "PBE0 IE Error",
                "B3LYP": "B3LYP IE Error",
                # "B2PLYP": "B2PLYP IE Error",
                # "WB97X": "WB97X IE Error",
                # SAPT(DFT)
                "SAPT(PBE0)": "SAPT(DFT) [PBE0] TOTAL ENERGY Error",
                "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] TOTAL ENERGY Error",
                # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] TOTAL ENERGY Error",
                # "SAPT(WB97X)": "SAPT(DFT) [WB97X] TOTAL ENERGY Error",
                # DFT-D4
                "PBE0-D4(SAPT)": "PBE0-D4 TOTAL ENERGY Error",
                "B3LYP-D4(SAPT)": "B3LYP-D4 TOTAL ENERGY Error",
                # "B2PLYP-D4": "B2PLYP-D4 TOTAL ENERGY Error",
                # "WB97X-D4": "WB97X-D4 TOTAL ENERGY Error",
                # SAPT(DFT) D's
                "SAPT(PBE0)-D4": "SAPT(DFT)-D4 TOTAL ENERGY Error",
                # "SAPT(DFT)+D4": "SAPT(DFT)+D4 TOTAL ENERGY Error",
                "SAPT(PBE0)D3-ML": "SAPT(DFT)D3-ML TOTAL ENERGY Error",
                # Wavefunction
                "SAPT0-D4": "SAPT0-D4 TOTAL ENERGY Error",
                "SAPT0": "SAPT0 TOTAL ENERGY Error",
                "SAPT2": "SAPT2 TOTAL ENERGY Error",
                "SAPT2+": "SAPT2+ TOTAL ENERGY Error",
                "SAPT2+(3)": "SAPT2+(3) TOTAL ENERGY Error",
                "SAPT2+3": "SAPT2+3 TOTAL ENERGY Error",
                "SAPT2+3(CCD)DMP2": "SAPT2+3(CCD)DMP2 TOTAL ENERGY Error",
            },
            colors=colors_total,
            output_filename=f"./plots/individuals/LoS_components_{dfs[0]['name']}_total_pres.jpg",
            usetex=True,
            legend_loc="lower right",
            figure_size=(12, 3),
            grid_widths=[1.0],
            grid_heights=[
                0.10,
                1,
            ],
            table_fontsize=16,
            x_label_fontsize=16,
            y_label_fontsize=16,
            title_fontsize=16,
            mcure=None,
            MAE="textbf",
            RMSE=False,
            MinE=False,
            MaxE=False,
            annotations_texty=-0.2,
            share_y_axis=True,
            wspace=0.05,
            table_delimiter=",",
            gridlines_linewidths=1.5,
            violin_alphas=1.0,
            quantile_linewidth=1.2,
            pm_alpha=0.5,
            zero_alpha=1.0,
            # mcure=mcure_labels,
        )
    return


def violin_plots_multi_components_subset(
    df,
    limit_to_column_not_nan=None,
    dfs=None,
):
    if dfs is None:
        dfs = _prepare_component_violin_dfs(
            df,
            bases=("adz", "atz", "aqz"),
            limit_to_column_not_nan=limit_to_column_not_nan,
            subset_only=True,
        )

    sapt_reference = _COMPONENT_SAPT_REFERENCE
    sapt_methods = _COMPONENT_METHODS_SUBSET.copy()
    df_by_basis = {d["name"]: d["df"] for d in dfs}
    df_adz = df_by_basis["adz"]
    df_atz = df_by_basis["atz"]
    df_aqz = df_by_basis["aqz"]
    import cdsg_plot

    # fig_size = (12, 9)
    fig_size = (9, 9)
    grid_heights = [
        0.9,
        2,
        0.7,
        2,
        0.7,
        2,
    ]
    table_fontsize = 12
    x_label_fontsize = 12
    extra_label = "_slide"
    print(df_adz[sapt_reference.values()])
    print(df_atz[sapt_reference.values()])
    print(df_aqz[sapt_reference.values()])
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_elst={
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] ELST ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] ELST ENERGY Error",
            "SAPT0": "SAPT0 ELST ENERGY Error",
            "SAPT2+3": "SAPT2+(3) ELST ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] ELST ENERGY Error",
            # "SAPT(WB97X)": "SAPT(DFT) [WB97X] ELST ENERGY Error",
        },
        df_labels_and_columns_exch={
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] EXCH ENERGY Error",
            "SAPT0": "SAPT0 EXCH ENERGY Error",
            "SAPT2+3": "SAPT2 EXCH ENERGY Error",
            # "SAPT(B3LYP) dMP2": "SAPT(DFT) [B3LYP] dMP2 EXCH ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error",
            # "SAPT(WB97X)": "SAPT(DFT) [WB97X] EXCH ENERGY Error",
        },
        df_labels_and_columns_indu={
            # "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] IND ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] IND ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] IND ENERGY Error",
            # "SAPT(WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error",
            "SAPT0": "SAPT0 IND ENERGY Error",
            # "sSAPT0": "SSAPT0 IND ENERGY Error",
            "SAPT2+3": "SAPT2 IND ENERGY Error",
            # should be here but moved for dense plotting...
            # "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
            "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        colors=colors_comps_saptdftd4,
        violin_alphas=0.9,
        usetex=True,
        legend_loc="lower right",
        table_fontsize=table_fontsize,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=11,
        share_y_axis=True,
        figure_size=fig_size,
        grid_heights=grid_heights,
        grid_widths=[5, 5, 6],
        output_filename=f"./plots/LoS_components_adz_atz_aqz_subset_nondisp.jpg",
        # mcure=mcure_labels,
    )
    cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_disp={
            "SAPT(PBE0)": "SAPT(DFT) [PBE0] DISP ENERGY Error",
            "SAPT(B3LYP)": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
            # "SAPT(B2PLYP)": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
            # "SAPT(WB97X)": "SAPT(DFT) [WB97X] DISP ENERGY Error",
            # "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error",
            "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
            "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
            "PBE0-D3(SAPT)": "PBE0-D3 DISP ENERGY Error",
            "B3LYP-D3(SAPT)": "B3LYP-D3 DISP ENERGY Error",
            "SAPT(PBE0)-D4(S)": "SAPT(DFT)-D4 DISP ENERGY Error",
            "SAPT(PBE0)-D4(I)": "SAPT(PBE0)-D4 INTER DISP ENERGY Error",
            "SAPT(B3LYP)-D4(I)": "SAPT(B3LYP)-D4 INTER DISP ENERGY Error",
            "SAPT(PBE0)-D3(I)": "SAPT(PBE0)-D3 INTER DISP ENERGY Error",
            "SAPT(B3LYP)-D3(I)": "SAPT(B3LYP)-D3 INTER DISP ENERGY Error",
            # "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
            # "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
            "D3-ML": "D3-ML DISP ENERGY Error",
            "SAPT0": "SAPT0 DISP ENERGY Error",
            "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
            # "sSAPT0": "SSAPT0 DISP ENERGY Error",
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
            # "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
            "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
            # "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
            # "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
            "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        colors=colors_disp_saptdftd4,
        usetex=True,
        legend_loc="lower right",
        table_fontsize=table_fontsize,
        x_label_fontsize=x_label_fontsize,
        violin_alphas=0.9,
        y_label_fontsize=11,
        figure_size=fig_size,
        grid_heights=grid_heights,
        grid_widths=[1.0],
        output_filename=f"./plots/LoS_components_adz_atz_aqz_subset_disp.jpg",
    )

    # Compute differences between SAPT0 vs DFT and SAPT0 vs SAPT2+3(CCD),
    # determine if we can see how correlated differences in intramolecular
    # correlation of components are to each other. For this purpose, define
    # "SAPT0" as the reference even though that is not technically correct.
    sapt_reference = {
        "ELST": "SAPT0 ELST ENERGY",
        "EXCH": "SAPT0 EXCH ENERGY",
        "IND": "SAPT0 IND ENERGY",
        "DISP": "SAPT0 DISP ENERGY",
    }
    sapt_methods.extend(
        [
            "SAPT2+3(CCD)DMP2 ELST ENERGY",
            "SAPT2+3(CCD)DMP2 EXCH ENERGY",
            "SAPT2+3(CCD)DMP2 IND ENERGY",
            "SAPT2+3(CCD)DMP2 DISP ENERGY",
        ]
    )

    df_adz = sapt_error_comp(
        df_adz, df_adz, sapt_reference, sapt_methods, extra_label="0"
    )
    df_atz = sapt_error_comp(
        df_atz, df_atz, sapt_reference, sapt_methods, extra_label="0"
    )
    df_aqz = sapt_error_comp(
        df_aqz, df_aqz, sapt_reference, sapt_methods, extra_label="0"
    )

    # cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
    #     dfs,
    #     df_labels_and_columns_elst={
    #         "SAPT2,SAPT2+": "SAPT2 ELST ENERGY Error0",
    #         "SAPT2+(3),SAPT2+3": "SAPT2+(3) ELST ENERGY Error0",
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error0",
    #         "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] ELST ENERGY Error0",
    #         # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] ELST ENERGY Error0",
    #         # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] ELST ENERGY Error0",
    #     },
    #     df_labels_and_columns_exch={
    #         "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 EXCH ENERGY Error0",
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error0",
    #         "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] EXCH ENERGY Error0",
    #         "SAPT(DFT) [B3LYP] dMP2": "SAPT(DFT) [B3LYP] dMP2 EXCH ENERGY Error0",
    #         # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error0",
    #         # "SAPT(DFT) [B2PLYP] dMP2": "SAPT(DFT) [B2PLYP] dMP2 EXCH ENERGY Error0",
    #         # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] EXCH ENERGY Error0",
    #     },
    #     df_labels_and_columns_indu={
    #         "SAPT0": "SAPT0 IND ENERGY Error0",
    #         "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 IND ENERGY Error0",
    #         # should be here but moved for dense plotting...
    #         # "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error0",
    #         "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error0",
    #         "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error0",
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error0",
    #         "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] IND ENERGY Error0",
    #         "SAPT(DFT) [B3LYP] dMP2": "SAPT(DFT) [B3LYP] dMP2 IND ENERGY Error0",
    #         # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] IND ENERGY Error0",
    #         # "SAPT(DFT) [B2PLYP] dMP2": "SAPT(DFT) [B2PLYP] dMP2 IND ENERGY Error0",
    #         # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error0",
    #     },
    #     df_labels_and_columns_disp={},
    #     output_filename=f"./plots/LoS_components_adz_atz_aqz_subset_nondisp-sapt0-sapt2.jpg",
    #     table_fontsize=8,
    #     usetex=True,
    #     legend_loc="lower right",
    #     figure_size=(10, 8),
    #     grid_heights=[
    #         0.6,
    #         2,
    #         0.4,
    #         2,
    #         0.4,
    #         2,
    #     ],
    #     grid_widths=[0.75, 0.75, 1.0],
    #     # mcure=mcure_labels,
    # )
    # cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
    #     dfs,
    #     df_labels_and_columns_elst={},
    #     df_labels_and_columns_exch={},
    #     df_labels_and_columns_indu={},
    #     df_labels_and_columns_disp={
    #         "SAPT0,SAPT2": "SAPT0 DISP ENERGY Error0",
    #         "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error0",
    #         # "sSAPT0": "SSAPT0 DISP ENERGY Error0",
    #         "SAPT2+": "SAPT2+ DISP ENERGY Error0",
    #         "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error0",
    #         "SAPT2+3": "SAPT2+3 DISP ENERGY Error0",
    #         "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error0",
    #         "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error0",
    #         "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error0",
    #         "SAPT(DFT)-D4": "SAPT(DFT)-D4 DISP ENERGY Error0",
    #         "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error0",
    #         "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] DISP ENERGY Error0",
    #         # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] DISP ENERGY Error0",
    #         # "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] DISP ENERGY Error0",
    #         "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error0",
    #         "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error0",
    #         "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error0",
    #         "B3LYP-D4 dMP2": "B3LYP-D4 dMP2 DISP ENERGY Error0",
    #         # "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error0",
    #         # "B2PLYP-D4 dMP2": "B2PLYP-D4 dMP2 DISP ENERGY Error0",
    #         # "WB97X-D4": "WB97X-D4 DISP ENERGY Error0",
    #         "D3-ML": "D3-ML DISP ENERGY Error0",
    #     },
    #     output_filename=f"./plots/LoS_components_adz_atz_aqz_subset_disp-sapt0-sapt2.jpg",
    #     table_fontsize=8,
    #     usetex=True,
    #     legend_loc="lower right",
    #     figure_size=(10, 8),
    #     grid_heights=[
    #         0.6,
    #         2,
    #         0.4,
    #         2,
    #         0.4,
    #         2,
    #     ],
    #     grid_widths=[1.0],
    #     # mcure=mcure_labels,
    # )

    return


def violin_plots_multi_components_subset_individual(df, limit_to_column_not_nan=None):
    df = df[df["subset"]].copy()
    print(f"Subset: {len(df)}")
    if limit_to_column_not_nan is not None:
        size_prior = len(df)
        df = df[df[limit_to_column_not_nan].notna()].copy()
        print(
            f"Limiting to {limit_to_column_not_nan} not NaN: {size_prior} -> {len(df)}"
        )
    df = prep_saptdft_components(df, "pbe0", "adz")
    df = prep_saptdft_components(df, "pbe0", "atz")
    df = prep_saptdft_components(df, "pbe0", "aqz")
    df = prep_saptdft_components(df, "b3lyp", "adz")
    df = prep_saptdft_components(df, "b3lyp", "atz")
    df = prep_saptdft_components(df, "b3lyp", "aqz")
    df = prep_saptdft_components(df, "b2plyp", "adz")
    df = prep_saptdft_components(df, "b2plyp", "atz")
    df = prep_saptdft_components(df, "b2plyp", "aqz")
    df = prep_saptdft_components(df, "wb97x", "adz")
    df = prep_saptdft_components(df, "wb97x", "atz")
    df = prep_saptdft_components(df, "wb97x", "aqz")

    df["SAPT0-D4 DISP ENERGY adz"] = df["-D4 (SAPT0_adz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 DISP ENERGY atz"] = df["-D4 (SAPT0_atz_3_IE)"] / h2kcalmol
    df["SAPT0-D4 DISP ENERGY aqz"] = df["-D4 (SAPT0_atz_3_IE)"] / h2kcalmol

    sapt_methods = [
        "SAPT0 ELST ENERGY",
        "SAPT2 ELST ENERGY",
        "SAPT2+(3) ELST ENERGY",
        "SAPT(DFT) [PBE0] ELST ENERGY",
        "SAPT(DFT) [B3LYP] ELST ENERGY",
        "SAPT(DFT) [B2PLYP] ELST ENERGY",
        "SAPT(DFT) [WB97X] ELST ENERGY",
        "SAPT0 EXCH ENERGY",
        "SAPT2 EXCH ENERGY",
        "SAPT(DFT) [PBE0] EXCH ENERGY",
        "SAPT(DFT) [B3LYP] EXCH ENERGY",
        "SAPT(DFT) [B2PLYP] EXCH ENERGY",
        "SAPT(DFT) [WB97X] EXCH ENERGY",
        "SAPT0 IND ENERGY",
        # "SSAPT0 IND ENERGY",
        "SAPT2 IND ENERGY",
        "SAPT2+DMP2 IND ENERGY",
        "SAPT2+3DMP2 IND ENERGY",
        "SAPT(DFT) [PBE0] IND ENERGY",
        "SAPT(DFT) [B3LYP] IND ENERGY",
        "SAPT(DFT) [B2PLYP] IND ENERGY",
        "SAPT(DFT) [WB97X] IND ENERGY",
        "SAPT0 DISP ENERGY",
        # "SSAPT0 DISP ENERGY",
        "SAPT2+ DISP ENERGY",
        "SAPT2+(3) DISP ENERGY",
        "SAPT2+3 DISP ENERGY",
        "SAPT2+(CCD) DISP ENERGY",
        "SAPT2+(3)(CCD) DISP ENERGY",
        "SAPT2+3(CCD) DISP ENERGY",
        # local disp
        "SAPT0-D4 DISP ENERGY",
        "PBE0-D4 DISP ENERGY",
        "PBE0-D3 DISP ENERGY",
        "B3LYP-D4 DISP ENERGY",
        "B2PLYP-D4 DISP ENERGY",
        "WB97X-D4 DISP ENERGY",
        "SAPT(DFT) [PBE0] DISP ENERGY",
        "SAPT(DFT) [B3LYP] DISP ENERGY",
        "SAPT(DFT) [B2PLYP] DISP ENERGY",
        "SAPT(DFT) [WB97X] DISP ENERGY",
        "D3-ML DISP ENERGY",
        "SAPT(DFT)+D4 DISP ENERGY",
        "SAPT(DFT)-D4 DISP ENERGY",
    ]
    sapt_reference = {
        "ELST": "SAPT2+3(CCD)DMP2 ELST ENERGY",
        "EXCH": "SAPT2+3(CCD)DMP2 EXCH ENERGY",
        "IND": "SAPT2+3(CCD)DMP2 IND ENERGY",
        "DISP": "SAPT2+3(CCD)DMP2 DISP ENERGY",
    }

    reference = "benchmark ref energy"
    copy_cols_start = [
        "DB",
        "system_id",
        "benchmark ref energy",
        "E_R_eq",
        "R",
        "Ref_elst_aqz",
        "E_R_eq_elst_aqz",
        "Ref_exch_aqz",
        "E_R_eq_exch_aqz",
        "Ref_ind_aqz",
        "E_R_eq_ind_aqz",
        "Ref_disp_aqz",
        "E_R_eq_disp_aqz",
        # local methods
        ## saptdft
        # "SAPT_DFT_pbe0_adz_elst",
        # "SAPT_DFT_pbe0_adz_exch",
        # "SAPT_DFT_pbe0_adz_ind",
        # "SAPT_DFT_pbe0_adz_disp",
        # "SAPT_DFT_pbe0d4_adz_disp",
        # "SAPT0-D4 adz",
        # "SAPT_DFT_pbe0_atz_elst",
        # "SAPT_DFT_pbe0_atz_exch",
        # "SAPT_DFT_pbe0_atz_ind",
        # "SAPT_DFT_pbe0_atz_disp",
        # "SAPT_DFT_pbe0d4_atz_disp",
        # "SAPT0-D4 atz",
    ]
    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} adz" for c in sapt_methods])
    copy_cols.extend([f"{c} adz" for c in sapt_reference.values()])
    df_adz = df[copy_cols].copy()
    df_adz.columns = [c.replace(" adz", "") for c in df_adz.columns]

    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} atz" for c in sapt_methods])
    copy_cols.extend([f"{c} atz" for c in sapt_reference.values()])
    df_atz = df[copy_cols].copy()
    df_atz.columns = [c.replace(" atz", "") for c in df_atz.columns]

    copy_cols = copy_cols_start.copy()
    copy_cols.extend([f"{c} aqz" for c in sapt_methods])
    copy_cols.extend([f"{c} aqz" for c in sapt_reference.values()])
    df_aqz = df[copy_cols].copy()
    df_aqz.columns = [c.replace(" aqz", "") for c in df_aqz.columns]

    df_adz = sapt_error_comp(df_adz, df_aqz, sapt_reference, sapt_methods)
    df_atz = sapt_error_comp(df_atz, df_aqz, sapt_reference, sapt_methods)
    df_aqz = sapt_error_comp(df_aqz, df_aqz, sapt_reference, sapt_methods)

    adz_ylims = [
        [-3, 3],
        [-3, 3],
        [-3, 3],
        [-3, 3],
    ]

    dfs_all = [
        {
            "df": df_adz,
            "basis": "aug-cc-pVDZ",
            "label": "aug-cc-pVDZ",
            "ylim": adz_ylims,
            "name": "adz",
        },
        {
            "df": df_atz,
            "basis": "aug-cc-pVTZ",
            "label": "aug-cc-pVTZ",
            "ylim": adz_ylims,
            "name": "atz",
        },
        {
            "df": df_aqz,
            "basis": "aug-cc-pVQZ",
            "label": "aug-cc-pVQZ",
            "ylim": adz_ylims,
            "name": "aqz",
        },
    ]
    mcures_labels_start = [
        {
            "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
        },
        {
            "SAPT2,SAPT2+": "SAPT2 ELST ENERGY Error",
        },
        {
            "SAPT2+(3),SAPT2+3": "SAPT2+(3) ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] ELST ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] ELST ENERGY Error",
        },
        {
            "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] EXCH ENERGY Error",
        },
        {
            "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 EXCH ENERGY Error",
        },
        {
            "SAPT0": "SAPT0 IND ENERGY Error",
        },
        # {
        #     "sSAPT0": "SSAPT0 IND ENERGY Error",
        # },
        {
            "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 IND ENERGY Error",
        },
        {
            "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
        },
        {
            "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] IND ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] IND ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error",
        },
        {
            "SAPT0,SAPT2": "SAPT0 DISP ENERGY Error",
        },
        # {
        #     "sSAPT0": "SSAPT0 DISP ENERGY Error",
        # },
        {
            "SAPT2+": "SAPT2+ DISP ENERGY Error",
        },
        {
            "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
        },
        {
            "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
        },
        {
            "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
        },
        {
            "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
        },
        {
            "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] DISP ENERGY Error",
        },
        {
            "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
        },
        {
            "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
        },
        {
            "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
        },
        {
            "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
        },
        {
            "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
        },
        {
            "D3-ML": "D3-ML DISP ENERGY Error",
        },
        {
            "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error",
        },
        {
            "SAPT(DFT)-D4": "SAPT(DFT)-D4 DISP ENERGY Error",
        },
    ]
    mcure_labels = {
        "ELST": {},
        "EXCH": {},
        "IND": {},
        "DISP": {},
    }
    # TODO DEBUG
    df_atz["SAPT2+3(CCD)DMP2 EXCH ENERGY"] *= h2kcalmol
    debug = False
    for i in mcures_labels_start:
        k, v = list(i.items())[0]
        for d in dfs_all:
            if "ELST" in v:
                if k not in mcure_labels["ELST"]:
                    mcure_labels["ELST"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_ELST"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_elst_aqz",
                        benchmark_col_system=f"E_R_eq_elst_aqz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_ELST"].abs().mean() * 100
                mcure_labels["ELST"][k].append(mcure)
            elif "EXCH" in v:
                if k not in mcure_labels["EXCH"]:
                    mcure_labels["EXCH"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_EXCH"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_exch_aqz",
                        benchmark_col_system=f"E_R_eq_exch_aqz",
                        debug=debug,
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_EXCH"].abs().mean() * 100
                mcure_labels["EXCH"][k].append(mcure)
            elif "IND" in v:
                if k not in mcure_labels["IND"]:
                    mcure_labels["IND"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_IND"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_ind_aqz",
                        benchmark_col_system=f"E_R_eq_ind_aqz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_IND"].abs().mean() * 100
                mcure_labels["IND"][k].append(mcure)
            elif "DISP" in v:
                if k not in mcure_labels["DISP"]:
                    mcure_labels["DISP"][k] = []
                v = v.replace(" Error", "")
                d["df"]["CRE_DISP"] = d["df"].apply(
                    lambda r: compute_CRE(
                        r,
                        energy_col=v,
                        benchmark_col=f"Ref_disp_aqz",
                        benchmark_col_system=f"E_R_eq_disp_aqz",
                    ),
                    axis=1,
                )
                mcure = d["df"]["CRE_DISP"].abs().mean() * 100
                mcure_labels["DISP"][k].append(mcure)
            else:
                raise ValueError(f"Error: {v = }")
    pp(mcure_labels)

    import cdsg_plot

    for dfs in dfs_all:
        dfs = [dfs]
        cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
            dfs,
            df_labels_and_columns_elst={
                "SAPT0,sSAPT0": "SAPT0 ELST ENERGY Error",
                "SAPT2,SAPT2+": "SAPT2 ELST ENERGY Error",
                "SAPT2+(3),SAPT2+3": "SAPT2+(3) ELST ENERGY Error",
                "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] ELST ENERGY Error",
                "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] ELST ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] ELST ENERGY Error",
                "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] ELST ENERGY Error",
            },
            df_labels_and_columns_exch={
                "SAPT0,sSAPT0": "SAPT0 EXCH ENERGY Error",
                "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 EXCH ENERGY Error",
                "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] EXCH ENERGY Error",
                "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] EXCH ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] EXCH ENERGY Error",
                "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] EXCH ENERGY Error",
            },
            df_labels_and_columns_indu={
                "SAPT0": "SAPT0 IND ENERGY Error",
                # "sSAPT0": "SSAPT0 IND ENERGY Error",
                "SAPT2,SAPT2+,\\\\SAPT2+(3),SAPT2+3": "SAPT2 IND ENERGY Error",
                # should be here but moved for dense plotting...
                # "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
                "SAPT2+3DMP2": "SAPT2+3DMP2 IND ENERGY Error",
                "SAPT2+DMP2,\\\\SAPT2+(3)DMP2": "SAPT2+DMP2 IND ENERGY Error",
                "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] IND ENERGY Error",
                "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] IND ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] IND ENERGY Error",
                "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] IND ENERGY Error",
            },
            df_labels_and_columns_disp={},
            # df_labels_and_columns_disp={
            #     "SAPT0,SAPT2": "SAPT0 DISP ENERGY Error",
            #     "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
            #     # "sSAPT0": "SSAPT0 DISP ENERGY Error",
            #     "SAPT2+": "SAPT2+ DISP ENERGY Error",
            #     "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
            #     "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
            #     "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
            #     "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
            #     "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
            #     "SAPT(DFT)-D4": "SAPT(DFT)-D4 DISP ENERGY Error",
            #     "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
            #     "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
            #     "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
            #     "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] DISP ENERGY Error",
            #     "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error",
            #     "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
            #     "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
            #     "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
            #     "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
            #     "D3-ML": "D3-ML DISP ENERGY Error",
            # },
            output_filename=f"./plots/individuals/LoS_components_{dfs[0]['name']}_subset_nondisp.jpg",
            table_fontsize=8,
            usetex=True,
            legend_loc="lower right",
            figure_size=(10, 3),
            x_label_fontsize=12,
            grid_widths=[1.0, 0.9, 1.2],
            grid_heights=[
                0.50,
                2,
            ],
        )
        cdsg_plot.error_statistics.violin_plot_table_multi_SAPT_components(
            dfs,
            df_labels_and_columns_elst={},
            df_labels_and_columns_exch={},
            df_labels_and_columns_indu={},
            df_labels_and_columns_disp={
                "SAPT0,SAPT2": "SAPT0 DISP ENERGY Error",
                "SAPT0-D4": "SAPT0-D4 DISP ENERGY Error",
                # "sSAPT0": "SSAPT0 DISP ENERGY Error",
                "SAPT2+": "SAPT2+ DISP ENERGY Error",
                "SAPT2+(3)": "SAPT2+(3) DISP ENERGY Error",
                "SAPT2+3": "SAPT2+3 DISP ENERGY Error",
                "SAPT2+(CCD)": "SAPT2+(CCD) DISP ENERGY Error",
                "SAPT2+(3)(CCD)": "SAPT2+(3)(CCD) DISP ENERGY Error",
                "SAPT2+3(CCD)": "SAPT2+3(CCD) DISP ENERGY Error",
                "SAPT(DFT)-D4": "SAPT(DFT)-D4 DISP ENERGY Error",
                "SAPT(DFT) [PBE0]": "SAPT(DFT) [PBE0] DISP ENERGY Error",
                "SAPT(DFT) [B3LYP]": "SAPT(DFT) [B3LYP] DISP ENERGY Error",
                # "SAPT(DFT) [B2PLYP]": "SAPT(DFT) [B2PLYP] DISP ENERGY Error",
                "SAPT(DFT) [WB97X]": "SAPT(DFT) [WB97X] DISP ENERGY Error",
                "SAPT(DFT)+D4": "SAPT(DFT)+D4 DISP ENERGY Error",
                "PBE0-D4(SAPT)": "PBE0-D4 DISP ENERGY Error",
                "B3LYP-D4(SAPT)": "B3LYP-D4 DISP ENERGY Error",
                # "B2PLYP-D4": "B2PLYP-D4 DISP ENERGY Error",
                "WB97X-D4": "WB97X-D4 DISP ENERGY Error",
                "D3-ML": "D3-ML DISP ENERGY Error",
            },
            output_filename=f"./plots/individuals/LoS_components_{dfs[0]['name']}_subset_disp.jpg",
            table_fontsize=8,
            usetex=True,
            legend_loc="lower right",
            figure_size=(10, 3),
            x_label_fontsize=12,
            grid_widths=[1.0],
            grid_heights=[
                0.50,
                2,
            ],
            # mcure=mcure_labels,
        )
    return


def plot_components_sapt0_saptdft(df):
    df = compute_d3_from_opt_params(
        df,
        bases=[
            [
                "SAPT0_adz_3_IE",
                "SAPT0_adz_3_IE",
                "sadz",
                "SAPT0_adz_3_IE",
            ],
            [
                "SAPT0_atz_3_IE",
                "SAPT0_atz_3_IE",
                "satz",
                "SAPT0_atz_3_IE",
            ],
        ],
    )
    df = compute_d4_from_opt_params(
        df,
        bases=[
            [
                "SAPT0_adz_3_IE",
                "SAPT0_adz_3_IE_BJ_inter",
                "SAPT0_adz_3_IE_2B_BJ_inter",
                "SAPT0_adz_3_IE",
            ],
            [
                "SAPT0_atz_3_IE",
                "SAPT0_atz_3_IE_BJ_inter",
                "SAPT0_atz_3_IE_2B_BJ_inter",
                "SAPT0_atz_3_IE",
            ],
        ],
        disp_compute=locald4.compute_disp_2B_BJ_dimer_inter,
    )
    # violin_plots_multi_components_subset_sapt0d4(df)
    violin_plots_multi_components_sapt0d4(df)
    return


def d3ml_saptdft(df, functional="b3lyp"):
    for i in ["adz", "atz", "aqz"]:
        col = f"SAPT_DFT_{functional}_{i}"
        df[f"SAPT({functional.upper()})D3-ML TOTAL ENERGY {i}"] = df.apply(
            lambda r: sum(r[col][1:4]) + r[f"D3-ML"] if r[col] is not None else None,
            axis=1,
        )
        print(df[f"D3-ML"].describe())
        df[f"SAPT({functional.upper()})D3-ML TOTAL ENERGY {i}"] /= h2kcalmol
    return df


def d4_conversions(df):
    df["SAPT0-D4 TOTAL ENERGY adz"] = df.apply(
        lambda r: (r["SAPT0_adz_3_IE"] + r["-D4 (SAPT0_adz_3_IE)"]) / h2kcalmol,
        axis=1,
    )
    df["SAPT0-D4 TOTAL ENERGY atz"] = df.apply(
        lambda r: (r["SAPT0_atz_3_IE"] + r["-D4 (SAPT0_atz_3_IE)"]) / h2kcalmol,
        axis=1,
    )
    df["SAPT0-D4 TOTAL ENERGY aqz"] = df.apply(
        lambda r: (r["SAPT0_atz_3_IE"] + r["-D4 (SAPT0_atz_3_IE)"]) / h2kcalmol,
        axis=1,
    )
    df["SAPT0-D4 (I) TOTAL ENERGY adz"] = df.apply(
        lambda r: (
            (r["SAPT0_adz_3_IE"] + r["-D4 (SAPT0_adz_3_IE_2B_BJ_inter)"]) / h2kcalmol
        ),
        axis=1,
    )
    df["SAPT0-D4 (I) TOTAL ENERGY atz"] = df.apply(
        lambda r: (
            (r["SAPT0_atz_3_IE"] + r["-D4 (SAPT0_adz_3_IE_2B_BJ_inter)"]) / h2kcalmol
        ),
        axis=1,
    )
    df["SAPT0-D4 (I) TOTAL ENERGY aqz"] = df.apply(
        lambda r: (
            (r["SAPT0_atz_3_IE"] + r["-D4 (SAPT0_adz_3_IE_2B_BJ_inter)"]) / h2kcalmol
        ),
        axis=1,
    )
    df["SAPT0-D4 (I) DISP ENERGY adz"] = df.apply(
        lambda r: r["-D4 (SAPT0_adz_3_IE_2B_BJ_inter)"] / h2kcalmol,
        axis=1,
    )
    df["SAPT0-D4 (I) DISP ENERGY atz"] = df.apply(
        lambda r: r["-D4 (SAPT0_adz_3_IE_2B_BJ_inter)"] / h2kcalmol,
        axis=1,
    )
    df["SAPT0-D4 (I) DISP ENERGY aqz"] = df.apply(
        lambda r: r["-D4 (SAPT0_adz_3_IE_2B_BJ_inter)"] / h2kcalmol,
        axis=1,
    )

    df["SAPT(PBE0)-D4 INTER DISP ENERGY"] = df["-D4 (SAPT_DFT_pbe0_adz_3_IE_inter)"]
    df["SAPT(PBE0)-D4 INTER DISP ENERGY adz"] = (
        df["SAPT(PBE0)-D4 INTER DISP ENERGY"] / h2kcalmol
    )
    df["SAPT(PBE0)-D4 INTER DISP ENERGY atz"] = (
        df["SAPT(PBE0)-D4 INTER DISP ENERGY"] / h2kcalmol
    )
    df["SAPT(PBE0)-D4 INTER DISP ENERGY aqz"] = (
        df["SAPT(PBE0)-D4 INTER DISP ENERGY"] / h2kcalmol
    )
    df["SAPT(PBE0)-D4 INTER TOTAL ENERGY adz"] = (
        df["SAPT(PBE0)-D4 INTER DISP ENERGY"] / h2kcalmol
        + df["SAPT_DFT_pbe0_adz_3_IE"] / h2kcalmol
    )
    df["SAPT(PBE0)-D4 INTER TOTAL ENERGY atz"] = (
        df["SAPT(PBE0)-D4 INTER DISP ENERGY"] / h2kcalmol
        + df["SAPT_DFT_pbe0_atz_3_IE"] / h2kcalmol
    )
    df["SAPT(PBE0)-D4 INTER TOTAL ENERGY aqz"] = (
        df["SAPT(PBE0)-D4 INTER DISP ENERGY"] / h2kcalmol
        + df["SAPT_DFT_pbe0_aqz_3_IE"] / h2kcalmol
    )
    df["SAPT(B3LYP)-D4 INTER DISP ENERGY"] = df["-D4 (SAPT_DFT_b3lyp_adz_3_IE_inter)"]
    df["SAPT(B3LYP)-D4 INTER DISP ENERGY adz"] = (
        df["SAPT(B3LYP)-D4 INTER DISP ENERGY"] / h2kcalmol
    )
    df["SAPT(B3LYP)-D4 INTER DISP ENERGY atz"] = (
        df["SAPT(B3LYP)-D4 INTER DISP ENERGY"] / h2kcalmol
    )
    df["SAPT(B3LYP)-D4 INTER DISP ENERGY aqz"] = (
        df["SAPT(B3LYP)-D4 INTER DISP ENERGY"] / h2kcalmol
    )
    df["SAPT(B3LYP)-D4 INTER TOTAL ENERGY adz"] = (
        df["SAPT(B3LYP)-D4 INTER DISP ENERGY"] / h2kcalmol
        + df["SAPT_DFT_b3lyp_adz_3_IE"] / h2kcalmol
    )
    df["SAPT(B3LYP)-D4 INTER TOTAL ENERGY atz"] = (
        df["SAPT(B3LYP)-D4 INTER DISP ENERGY"] / h2kcalmol
        + df["SAPT_DFT_b3lyp_atz_3_IE"] / h2kcalmol
    )
    df["SAPT(B3LYP)-D4 INTER TOTAL ENERGY aqz"] = (
        df["SAPT(B3LYP)-D4 INTER DISP ENERGY"] / h2kcalmol
        + df["SAPT_DFT_b3lyp_aqz_3_IE"] / h2kcalmol
    )
    return df


def d3_conversions(df):
    """
    Compute D3(I) (intermolecular) and D3(S) (supermolecular) columns for SAPT(PBE0) and SAPT(B3LYP).

    D3(I) uses only intermolecular pairs (filtered D3Data).
    D3(S) uses the full supermolecular D3Data.
    """
    print("Computing D3 from optimized parameters...")
    # Get D3 BJ parameters from paramsTable
    # D3(I) params - intermolecular optimized
    params_pbe0_d3_inter = paramsTable.get_params("SAPT_DFT_pbe0_adz_3_IE_D3_inter")[0][
        1:4
    ]
    params_b3lyp_d3_inter = paramsTable.get_params("SAPT_DFT_b3lyp_adz_3_IE_D3_inter")[
        0
    ][1:4]
    # D3(S) params - supermolecular optimized
    params_pbe0_d3_super = paramsTable.get_params("SAPT_DFT_pbe0_adz_3_IE_D3_super")[0][
        1:4
    ]
    params_b3lyp_d3_super = paramsTable.get_params("SAPT_DFT_b3lyp_adz_3_IE_D3_super")[
        0
    ][1:4]
    params_pbe0_d3_ddft = paramsTable.get_params("PBE0-D3")[0][1:4]
    params_b3lyp_d3_ddft = paramsTable.get_params("B3LYP-D3")[0][1:4]

    # SAPT(PBE0)-D3(I) - Intermolecular D3
    df["SAPT(PBE0)-D3 INTER DISP ENERGY"] = df.apply(
        lambda r: (
            jeff.compute_BJ_CPP(
                params_pbe0_d3_inter,
                dftd3.filter_d3data_intermolecular(r["D3Data"], r["monAs"], r["monBs"]),
            )
            / h2kcalmol
            if r["D3Data"] is not None and len(r["D3Data"]) > 0
            else np.nan
        ),
        axis=1,
    )
    df["SAPT(PBE0)-D3 INTER DISP ENERGY adz"] = df["SAPT(PBE0)-D3 INTER DISP ENERGY"]
    df["SAPT(PBE0)-D3 INTER DISP ENERGY atz"] = df["SAPT(PBE0)-D3 INTER DISP ENERGY"]
    df["SAPT(PBE0)-D3 INTER DISP ENERGY aqz"] = df["SAPT(PBE0)-D3 INTER DISP ENERGY"]
    df["SAPT(PBE0)-D3 INTER TOTAL ENERGY adz"] = (
        df["SAPT(PBE0)-D3 INTER DISP ENERGY"] + df["SAPT_DFT_pbe0_adz_3_IE"] / h2kcalmol
    )
    df["SAPT(PBE0)-D3 INTER TOTAL ENERGY atz"] = (
        df["SAPT(PBE0)-D3 INTER DISP ENERGY"] + df["SAPT_DFT_pbe0_atz_3_IE"] / h2kcalmol
    )
    df["SAPT(PBE0)-D3 INTER TOTAL ENERGY aqz"] = (
        df["SAPT(PBE0)-D3 INTER DISP ENERGY"] + df["SAPT_DFT_pbe0_aqz_3_IE"] / h2kcalmol
    )

    # Create SAPT_DFT_pbe0_{basis}_D3_IE columns (raw D3 in hartree, like D4)
    df["SAPT_DFT_pbe0_adz_D3_IE"] = df.apply(
        lambda r: (
            jeff.compute_BJ_CPP(params_pbe0_d3_ddft, r["D3Data"])
            if r["D3Data"] is not None and len(r["D3Data"]) > 0
            else np.nan
        ),
        axis=1,
    )
    df["SAPT_DFT_pbe0_atz_D3_IE"] = df["SAPT_DFT_pbe0_adz_D3_IE"]
    df["SAPT_DFT_pbe0_aqz_D3_IE"] = df["SAPT_DFT_pbe0_adz_D3_IE"]

    # PBE0-D3 DISP ENERGY - derived from SAPT_DFT columns with dDFT/dHF corrections
    df["PBE0-D3 DISP ENERGY"] = df.apply(
        lambda r: (
            (
                r["SAPT_DFT_pbe0_adz_D3_IE"]
                + r["SAPT_DFT_pbe0_adz_dDFT"]
                - r["SAPT_DFT_pbe0_adz_dHF"]
            )
            / h2kcalmol
            if r["SAPT_DFT_D4_pbe0_adz_total"]
            and r["SAPT_DFT_pbe0_adz"]
            and not np.isnan(r["SAPT_DFT_pbe0_adz_D3_IE"])
            else np.nan
        ),
        axis=1,
    )
    # Add PBE0-D3 DISP ENERGY with basis set suffixes (with dDFT/dHF corrections)
    # SAPT_DFT_pbe0_adz_D3_IE
    df["PBE0-D3 DISP ENERGY adz"] = df.apply(
        lambda r: (
            (
                r["SAPT_DFT_pbe0_adz_D3_IE"]
                + r["SAPT_DFT_pbe0_adz_dDFT"]
                - r["SAPT_DFT_pbe0_adz_dHF"]
            )
            / h2kcalmol
            if r["SAPT_DFT_D4_pbe0_adz_total"]
            and r["SAPT_DFT_pbe0_adz"]
            and not np.isnan(r["SAPT_DFT_pbe0_adz_D3_IE"])
            else np.nan
        ),
        axis=1,
    )
    df["PBE0-D3 DISP ENERGY atz"] = df.apply(
        lambda r: (
            (
                r["SAPT_DFT_pbe0_atz_D3_IE"]
                + r["SAPT_DFT_pbe0_atz_dDFT"]
                - r["SAPT_DFT_pbe0_atz_dHF"]
            )
            / h2kcalmol
            if r["SAPT_DFT_D4_pbe0_atz_total"]
            and r["SAPT_DFT_pbe0_atz"]
            and not np.isnan(r["SAPT_DFT_pbe0_atz_D3_IE"])
            else np.nan
        ),
        axis=1,
    )
    df["PBE0-D3 DISP ENERGY aqz"] = df.apply(
        lambda r: (
            (
                r["SAPT_DFT_pbe0_aqz_D3_IE"]
                + r["SAPT_DFT_pbe0_aqz_dDFT"]
                - r["SAPT_DFT_pbe0_aqz_dHF"]
            )
            / h2kcalmol
            if r["SAPT_DFT_D4_pbe0_aqz_total"]
            and r["SAPT_DFT_pbe0_aqz"]
            and not np.isnan(r["SAPT_DFT_pbe0_aqz_D3_IE"])
            else np.nan
        ),
        axis=1,
    )
    # Add PBE0-D3 IE columns (PBE0 IE + D3 dispersion)
    df["PBE0-D3 IE adz"] = df.apply(
        lambda r: (
            r["PBE0 IE adz"] + r["SAPT_DFT_pbe0_adz_D3_IE"]
            if r["SAPT_DFT_D4_pbe0_adz_total"]
            and not np.isnan(r["PBE0-D3 DISP ENERGY adz"])
            else np.nan
        ),
        axis=1,
    )
    df["PBE0-D3 IE atz"] = df.apply(
        lambda r: (
            r["PBE0 IE atz"] + r["SAPT_DFT_pbe0_atz_D3_IE"]
            if r["SAPT_DFT_D4_pbe0_atz_total"]
            and not np.isnan(r["PBE0-D3 DISP ENERGY atz"])
            else np.nan
        ),
        axis=1,
    )
    df["PBE0-D3 IE aqz"] = df.apply(
        lambda r: (
            r["PBE0 IE aqz"] + r["SAPT_DFT_pbe0_aqz_D3_IE"]
            if r["SAPT_DFT_D4_pbe0_aqz_total"]
            and not np.isnan(r["PBE0-D3 DISP ENERGY aqz"])
            else np.nan
        ),
        axis=1,
    )
    # B3LYP-D3
    df["SAPT_DFT_b3lyp_adz_D3_IE"] = df.apply(
        lambda r: (
            jeff.compute_BJ_CPP(params_b3lyp_d3_ddft, r["D3Data"])
            if r["D3Data"] is not None and len(r["D3Data"]) > 0
            else np.nan
        ),
        axis=1,
    )
    df["SAPT_DFT_b3lyp_atz_D3_IE"] = df["SAPT_DFT_b3lyp_adz_D3_IE"]
    df["SAPT_DFT_b3lyp_aqz_D3_IE"] = df["SAPT_DFT_b3lyp_adz_D3_IE"]
    print(df[["SAPT_DFT_pbe0_adz_D3_IE", "SAPT_DFT_b3lyp_adz_D3_IE"]])
    print(df[["SAPT_DFT_b3lyp_adz_dDFT", "SAPT_DFT_b3lyp_adz_dHF"]])
    print(df[["SAPT_DFT_pbe0_adz_dDFT", "SAPT_DFT_pbe0_adz_dHF"]])

    # B3LYP-D3 DISP ENERGY - derived from SAPT_DFT columns with dDFT/dHF corrections
    # df["B3LYP-D3 DISP ENERGY"] = df.apply(
    #     lambda r: (
    #         r["SAPT_DFT_b3lyp_adz_D3_IE"]
    #         + r["SAPT_DFT_b3lyp_adz_dDFT"]
    #         - r["SAPT_DFT_b3lyp_adz_dHF"]
    #     )
    #     / h2kcalmol
    #     if r["SAPT_DFT_D4_b3lyp_adz_total"]
    #     and r["SAPT_DFT_b3lyp_adz"]
    #     and not np.isnan(r["SAPT_DFT_b3lyp_adz_D3_IE"])
    #     else np.nan,
    #     axis=1,
    # )
    # Add B3LYP-D3 DISP ENERGY with basis set suffixes (with dDFT/dHF corrections)
    df["B3LYP-D3 DISP ENERGY adz"] = df.apply(
        lambda r: (
            (
                r["SAPT_DFT_b3lyp_adz_D3_IE"]
                + r["SAPT_DFT_b3lyp_adz_dDFT"]
                - r["SAPT_DFT_b3lyp_adz_dHF"]
            )
            / h2kcalmol
            if r["SAPT_DFT_D4_b3lyp_adz_total"]
            and r["SAPT_DFT_b3lyp_adz"]
            and not np.isnan(r["SAPT_DFT_b3lyp_adz_D3_IE"])
            else np.nan
        ),
        axis=1,
    )
    print(df[["B3LYP-D3 DISP ENERGY adz"]])
    df["B3LYP-D3 DISP ENERGY atz"] = df.apply(
        lambda r: (
            (
                r["SAPT_DFT_b3lyp_atz_D3_IE"]
                + r["SAPT_DFT_b3lyp_atz_dDFT"]
                - r["SAPT_DFT_b3lyp_atz_dHF"]
            )
            / h2kcalmol
            if r["SAPT_DFT_D4_b3lyp_atz_total"]
            and r["SAPT_DFT_b3lyp_atz"]
            and not np.isnan(r["SAPT_DFT_b3lyp_atz_D3_IE"])
            else np.nan
        ),
        axis=1,
    )
    df["B3LYP-D3 DISP ENERGY aqz"] = df.apply(
        lambda r: (
            (
                r["SAPT_DFT_b3lyp_aqz_D3_IE"]
                + r["SAPT_DFT_b3lyp_aqz_dDFT"]
                - r["SAPT_DFT_b3lyp_aqz_dHF"]
            )
            / h2kcalmol
            if r["SAPT_DFT_D4_b3lyp_aqz_total"]
            and r["SAPT_DFT_b3lyp_aqz"]
            and not np.isnan(r["SAPT_DFT_b3lyp_aqz_D3_IE"])
            else np.nan
        ),
        axis=1,
    )
    # Add B3LYP-D3 IE columns (B3LYP IE + D3 dispersion)
    df["B3LYP-D3 IE adz"] = df.apply(
        lambda r: (
            r["B3LYP IE adz"] + r["SAPT_DFT_b3lyp_adz_D3_IE"]
            if r["SAPT_DFT_D4_b3lyp_adz_total"]
            and not np.isnan(r["B3LYP-D3 DISP ENERGY adz"])
            else np.nan
        ),
        axis=1,
    )
    df["B3LYP-D3 IE atz"] = df.apply(
        lambda r: (
            r["B3LYP IE atz"] + r["SAPT_DFT_b3lyp_atz_D3_IE"]
            if r["SAPT_DFT_D4_b3lyp_atz_total"]
            and not np.isnan(r["B3LYP-D3 DISP ENERGY atz"])
            else np.nan
        ),
        axis=1,
    )
    print(df[["B3LYP IE aqz", "B3LYP-D3 DISP ENERGY aqz"]])
    df["B3LYP-D3 IE aqz"] = df.apply(
        lambda r: (
            r["B3LYP IE aqz"] + r["SAPT_DFT_b3lyp_aqz_D3_IE"]
            if r["SAPT_DFT_D4_b3lyp_aqz_total"]
            and not np.isnan(r["B3LYP-D3 DISP ENERGY aqz"])
            else np.nan
        ),
        axis=1,
    )

    # SAPT(PBE0)-D3(S) - Supermolecular D3
    df["SAPT(PBE0)-D3 SUPER DISP ENERGY"] = df.apply(
        lambda r: (
            jeff.compute_BJ_CPP(params_pbe0_d3_super, r["D3Data"]) / h2kcalmol
            if r["D3Data"] is not None and len(r["D3Data"]) > 0
            else np.nan
        ),
        axis=1,
    )
    df["SAPT(PBE0)-D3 SUPER DISP ENERGY adz"] = df["SAPT(PBE0)-D3 SUPER DISP ENERGY"]
    df["SAPT(PBE0)-D3 SUPER DISP ENERGY atz"] = df["SAPT(PBE0)-D3 SUPER DISP ENERGY"]
    df["SAPT(PBE0)-D3 SUPER DISP ENERGY aqz"] = df["SAPT(PBE0)-D3 SUPER DISP ENERGY"]
    df["SAPT(PBE0)-D3 SUPER TOTAL ENERGY adz"] = (
        df["SAPT(PBE0)-D3 SUPER DISP ENERGY"] + df["SAPT_DFT_pbe0_adz_3_IE"] / h2kcalmol
    )
    df["SAPT(PBE0)-D3 SUPER TOTAL ENERGY atz"] = (
        df["SAPT(PBE0)-D3 SUPER DISP ENERGY"] + df["SAPT_DFT_pbe0_atz_3_IE"] / h2kcalmol
    )
    df["SAPT(PBE0)-D3 SUPER TOTAL ENERGY aqz"] = (
        df["SAPT(PBE0)-D3 SUPER DISP ENERGY"] + df["SAPT_DFT_pbe0_aqz_3_IE"] / h2kcalmol
    )

    # SAPT(B3LYP)-D3(I) - Intermolecular D3
    df["SAPT(B3LYP)-D3 INTER DISP ENERGY"] = df.apply(
        lambda r: (
            jeff.compute_BJ_CPP(
                params_b3lyp_d3_inter,
                dftd3.filter_d3data_intermolecular(r["D3Data"], r["monAs"], r["monBs"]),
            )
            / h2kcalmol
            if r["D3Data"] is not None and len(r["D3Data"]) > 0
            else np.nan
        ),
        axis=1,
    )
    df["SAPT(B3LYP)-D3 INTER DISP ENERGY adz"] = df["SAPT(B3LYP)-D3 INTER DISP ENERGY"]
    df["SAPT(B3LYP)-D3 INTER DISP ENERGY atz"] = df["SAPT(B3LYP)-D3 INTER DISP ENERGY"]
    df["SAPT(B3LYP)-D3 INTER DISP ENERGY aqz"] = df["SAPT(B3LYP)-D3 INTER DISP ENERGY"]
    df["SAPT(B3LYP)-D3 INTER TOTAL ENERGY adz"] = (
        df["SAPT(B3LYP)-D3 INTER DISP ENERGY"] + df["SAPT_DFT_b3lyp_adz_3_IE"]
    ) / h2kcalmol
    print(df[["SAPT(B3LYP)-D3 INTER TOTAL ENERGY adz"]])
    print(df[["SAPT_DFT_b3lyp_adz_3_IE"]])
    print(df[["SAPT(B3LYP)-D3 INTER DISP ENERGY"]])

    df["SAPT(B3LYP)-D3 INTER TOTAL ENERGY atz"] = (
        df["SAPT(B3LYP)-D3 INTER DISP ENERGY"] + df["SAPT_DFT_b3lyp_atz_3_IE"]
    ) / h2kcalmol
    df["SAPT(B3LYP)-D3 INTER TOTAL ENERGY aqz"] = (
        df["SAPT(B3LYP)-D3 INTER DISP ENERGY"] + df["SAPT_DFT_b3lyp_aqz_3_IE"]
    ) / h2kcalmol

    # SAPT(B3LYP)-D3(S) - Supermolecular D3
    df["SAPT(B3LYP)-D3 SUPER DISP ENERGY"] = df.apply(
        lambda r: (
            jeff.compute_BJ_CPP(params_b3lyp_d3_super, r["D3Data"])
            if r["D3Data"] is not None and len(r["D3Data"]) > 0
            else np.nan
        ),
        axis=1,
    )
    df["SAPT(B3LYP)-D3 SUPER DISP ENERGY adz"] = df["SAPT(B3LYP)-D3 SUPER DISP ENERGY"]
    df["SAPT(B3LYP)-D3 SUPER DISP ENERGY atz"] = df["SAPT(B3LYP)-D3 SUPER DISP ENERGY"]
    df["SAPT(B3LYP)-D3 SUPER DISP ENERGY aqz"] = df["SAPT(B3LYP)-D3 SUPER DISP ENERGY"]
    df["SAPT(B3LYP)-D3 SUPER TOTAL ENERGY adz"] = (
        df["SAPT(B3LYP)-D3 SUPER DISP ENERGY"] + df["SAPT_DFT_b3lyp_adz_3_IE"]
    ) / h2kcalmol
    df["SAPT(B3LYP)-D3 SUPER TOTAL ENERGY atz"] = (
        df["SAPT(B3LYP)-D3 SUPER DISP ENERGY"] + df["SAPT_DFT_b3lyp_atz_3_IE"]
    ) / h2kcalmol
    df["SAPT(B3LYP)-D3 SUPER TOTAL ENERGY aqz"] = (
        df["SAPT(B3LYP)-D3 SUPER DISP ENERGY"] + df["SAPT_DFT_b3lyp_aqz_3_IE"]
    ) / h2kcalmol

    for bs in ["adz", "atz", "aqz"]:
        for func in ["pbe0", "b3lyp"]:
            df[f"{func.upper()}-D3 TOTAL ENERGY {bs}"] = df.apply(
                lambda r: (
                    (
                        r[f"{func.upper()} IE {bs}"]
                        + r[f"SAPT_DFT_{func.lower()}_{bs.lower()}_D3_IE"]
                    )
                    / h2kcalmol
                    if r[f"SAPT_DFT_D4_{func}_{bs}_total"]
                    and not np.isnan(r[f"{func.upper()}-D3 DISP ENERGY {bs}"])
                    else np.nan
                ),
                axis=1,
            )
    df["PBE0-D3 TOTAL ENERGY adz"] = df["PBE0-D3 IE adz"] / h2kcalmol
    df["PBE0-D3 TOTAL ENERGY atz"] = df["PBE0-D3 IE atz"] / h2kcalmol
    df["PBE0-D3 TOTAL ENERGY aqz"] = df["PBE0-D3 IE aqz"] / h2kcalmol
    df["B3LYP-D3 TOTAL ENERGY adz"] = df["B3LYP-D3 IE adz"] / h2kcalmol
    df["B3LYP-D3 TOTAL ENERGY atz"] = df["B3LYP-D3 IE atz"] / h2kcalmol
    df["B3LYP-D3 TOTAL ENERGY aqz"] = df["B3LYP-D3 IE aqz"] / h2kcalmol
    return df


def plot_LoS_saptdft(
    df,
    presentation=False,
):
    print(f"{presentation = }")

    os.makedirs("./dfs", exist_ok=True)

    def _load_or_build(path, builder):
        if os.path.exists(path):
            print(f"Loading cached dataframe bundle: {path}")
            return pd.read_pickle(path)
        data = builder()
        pd.to_pickle(data, path)
        print(f"Saved dataframe bundle: {path}")
        return data

    limit_col = "B3LYP-D3 TOTAL ENERGY adz"
    total_full_dfs = _load_or_build(
        "./dfs/LoS_total_full_dfs.pkl",
        lambda: _prepare_total_violin_dfs(
            df,
            bases=("adz", "atz"),
            limit_to_column_not_nan=limit_col,
        ),
    )
    total_subset_dfs = _load_or_build(
        "./dfs/LoS_total_subset_dfs.pkl",
        lambda: _prepare_total_violin_dfs(
            df,
            bases=("adz", "atz", "aqz"),
            limit_to_column_not_nan=limit_col,
            subset_only=True,
        ),
    )
    components_full_dfs = _load_or_build(
        "./dfs/LoS_components_full_dfs.pkl",
        lambda: _prepare_component_violin_dfs(
            df,
            bases=("adz", "atz"),
            limit_to_column_not_nan=limit_col,
            subset_only=False,
        ),
    )
    components_subset_dfs = _load_or_build(
        "./dfs/LoS_components_subset_dfs.pkl",
        lambda: _prepare_component_violin_dfs(
            df,
            bases=("adz", "atz", "aqz"),
            limit_to_column_not_nan=limit_col,
            subset_only=True,
        ),
    )
    if presentation:
        # # return
        # violin_plots_multi_components_df_individual(df, limit_to_column_not_nan="D3-ML")
        # # return
        # violin_plots_multi_components_subset_individual(
        #     df, limit_to_column_not_nan="D3-ML"
        # )
        # violin_plots_multi_individual(df)
        # violin_plots_multi_components_df_individual(df)
        # violin_plots_multi_subset_individual(df)
        violin_plots_multi(df, dfs=total_full_dfs)
        violin_plots_multi_subset(df, dfs=total_subset_dfs)
        violin_plots_multi_components(df, dfs=components_full_dfs)
        violin_plots_multi_components_subset(df, dfs=components_subset_dfs)
    else:
        # return
        # return
        violin_plots_multi_components_df_individual(df, limit_to_column_not_nan="D3-ML")
        violin_plots_multi_individual(df)
        # return
        violin_plots_multi_components(df, limit_to_column_not_nan="D3-ML", slide=True)
        return
        violin_plots_multi(df)
        violin_plots_multi_subset(df)
        return
        # return
        return
    return
