import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import src
from qm_tools_aw import tools


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
    plt.savefig(f"{pfn}")
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
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return


def plot_dbs_d3_d4_two(df, c1, c2, l1, l2, title_name, pfn, first=True) -> None:
    """ """
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
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
        label=r"$\pm$1 $kcal\cdot mol^{-1}$",
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
    ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
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
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return


def get_charged_df(df) -> pd.DataFrame:
    df = df.copy()
    def_charge = np.array([[0, 1] for i in range(3)])
    print(df.columns.values)
    inds = []
    for i, row in df.iterrows():
        if np.all(row["charges"] == def_charge):
            inds.append(i)
    df = df.drop(inds)
    return df


def plotting_setup(df, build_df=False, df_out: str = "plots/plot.pkl", compute_d3=True):
    df, selected = df
    selected = selected.split("/")[-1].split(".")[0]
    df_out = f"plots/{selected}.pkl"
    if build_df:
        print(df.columns.values)
        for i in [j for j in df.columns.values if "SAPT0_" in j if j not in ["SAPT0", "SAPT0_jdz", "SAPT0_aqz"] if "_IE" not in j]:
            df[i + "_IE"] = df.apply(lambda r: r[i][0], axis=1)
            df[i + "_diff"] = df["Benchmark"] - df[i + "_IE"]

        df = compute_D3_D4_values_for_params_for_plotting(df, "adz", compute_d3)
        df = compute_D3_D4_values_for_params_for_plotting(df, "jdz", compute_d3)

        df["SAPT0-D4/aug-cc-pVDZ"] = df.apply(
            lambda row: row["HF_adz"] + row["-D4 (adz)"],
            axis=1,
        )
        df["SAPT0-D4(ATM)/aug-cc-pVDZ"] = df.apply(
            lambda row: row["HF_adz"] + row["-D4 (adz) ATM"],
            axis=1,
        )
        df["SAPT0-D4/jun-cc-pVDZ"] = df.apply(
            lambda row: row["HF_jdz"] + row["-D4 (jdz)"],
            axis=1,
        )
        df["SAPT0-D4(ATM)/jun-cc-pVDZ"] = df.apply(
            lambda row: row["HF_jdz"] + row["-D4 (jdz) ATM"],
            axis=1,
        )
        df["adz_diff_d4"] = df["Benchmark"] - df["SAPT0-D4/aug-cc-pVDZ"]
        df["adz_diff_d4_ATM"] = df["Benchmark"] - df["SAPT0-D4(ATM)/aug-cc-pVDZ"]
        df["adz_diff_d4_ATM_G"] = df["Benchmark"] - (
            df["HF_adz"] + df["-D4 (adz) ATM G"]
        )
        df["jdz_diff_d4"] = df["Benchmark"] - df["SAPT0-D4/jun-cc-pVDZ"]
        df["jdz_diff_d4_ATM"] = df["Benchmark"] - df["SAPT0-D4(ATM)/jun-cc-pVDZ"]
        df["SAPT0_jdz_diff"] = df["Benchmark"] - df["SAPT0"]
        df["SAPT0_jdz_diff"] = df["Benchmark"] - df["SAPT0"]

        if compute_d3:
            df["adz_diff_d4_2B@ATM_G"] = df["Benchmark"] - (
                df["HF_adz"] + df["-D4 2B@ATM_params (adz) G"]
            )
            df["SAPT0-D3/jun-cc-pVDZ"] = df.apply(
                lambda row: row["HF_jdz"] + row["-D3 (adz)"],
                axis=1,
            )
            df["SAPT0-D3/aug-cc-pVDZ"] = df.apply(
                lambda row: row["HF_adz"] + row["-D3 (adz)"],
                axis=1,
            )
            df["SAPT0-D3/aug-cc-pVDZ"] = df.apply(
                lambda row: row["HF_adz"] + row["-D3 (adz)"],
                axis=1,
            )
        df["adz_diff_d3"] = df["Benchmark"] - df["SAPT0-D3/aug-cc-pVDZ"]
        df["jdz_diff_d3"] = df["Benchmark"] - df["SAPT0-D3/jun-cc-pVDZ"]

        # D3 binary results
        df["jdz_diff_d3mbj"] = df["Benchmark"] - (df["HF_jdz"] + df["D3MBJ"])
        df["adz_diff_d3mbj"] = df["Benchmark"] - (df["HF_adz"] + df["D3MBJ"])
        df["jdz_diff_d3mbj_atm"] = df["Benchmark"] - (df["HF_jdz"] + df["D3MBJ ATM"])
        df["adz_diff_d3mbj_atm"] = df["Benchmark"] - (df["HF_adz"] + df["D3MBJ ATM"])
        df.to_pickle(df_out)
    else:
        df = pd.read_pickle(df_out)
    # Non charged
    plot_dbs_d3_d4(
        df,
        "adz_diff_d4",
        "adz_diff_d4_ATM_G",
        "-D4 (2B)",
        "-D4 (ATM_G)",
        title_name=f"DB Breakdown SAPT0-D4/aug-cc-pVDZ ({selected})",
        pfn=f"{selected}_db_breakdown_2B_ATM",
    )
    df_charged = get_charged_df(df)
    plot_violin_d3_d4_ALL(
        df,
        {
            "-D3/jun-cc-pVDZ": "jdz_diff_d3",
            "-D3MBJ(ATM)/jun-cc-pVDZ": "jdz_diff_d3mbj_atm",
            "-D3/aug-cc-pVDZ": "adz_diff_d3",
            "-D3MBJ(ATM)/aug-cc-pVDZ": "adz_diff_d3mbj_atm",
            "-D4/aug-cc-pVDZ": "adz_diff_d4",
            "-D4(ATM)/jun-cc-pVDZ": "jdz_diff_d4_ATM",
            "-D4(ATM)/aug-cc-pVDZ": "adz_diff_d4_ATM",
            "-D4(2B@ATM_params_G)/aug-cc-pVDZ": "adz_diff_d4_2B@ATM_G",
            "-D4(ATM_G)/aug-cc-pVDZ": "adz_diff_d4_ATM_G",
            "/jun-cc-pVDZ": "SAPT0_jdz_diff",
            "/aug-cc-pVDZ": "SAPT0_adz_diff",
        },
        f"All Dimers with SAPT0 ({selected})",
        f"{selected}_adz_d3_d4_total_sapt0",
    )
    # Basis Set Performance: SAPT0
    plot_violin_d3_d4_ALL(
        df,
        {
            "SAPT0-D4/cc-pVDZ": "SAPT0_dz_diff",
            "SAPT0-D4/jun-cc-pVDZ": "jdz_diff_d4",
            "SAPT0/jun-cc-pVDZ": "SAPT0_jdz_diff",
            "SAPT0-D4/aug-cc-pVDZ": "adz_diff_d4",
            "SAPT0/aug-cc-pVDZ": "SAPT0_adz_diff",
            # "SAPT0/cc-pVTZ": "SAPT0_tz_diff",
            "SAPT0/may-cc-pVTZ": "SAPT0_mtz_diff",
            "SAPT0/jun-cc-pVTZ": "SAPT0_jtz_diff",
            "SAPT0/aug-cc-pVTZ": "SAPT0_atz_diff",
        },
        f"All Dimers with SAPT0 ({selected})",
        f"{selected}_basis_set",
    )
    plot_violin_d3_d4_ALL(
        df_charged,
        {
            "SAPT0-D3/aug-cc-pVDZ": "adz_diff_d3",
            "SAPT0-D4/aug-cc-pVDZ": "adz_diff_d4",
            "SAPT0-D4(ATM)/jun-cc-pVDZ": "jdz_diff_d4_ATM",
            "SAPT0-D4(ATM)/aug-cc-pVDZ": "adz_diff_d4_ATM",
            "SAPT0/jun-cc-pVDZ": "SAPT0_jdz_diff",
            "SAPT0/aug-cc-pVDZ": "SAPT0_adz_diff",
        },
        f"Charged Dimers ({selected})",
        f"{selected}_adz_d3_d4_total_sapt0_charged",
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
        print(df.columns.values)

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
        title_name=f"DB Breakdown SAPT0-D4/aug-cc-pVDZ ({selected})",
        pfn=f"{selected}_db_breakdown_2B_ATM",
    )
    df_charged = get_charged_df(df)
    plot_violin_d3_d4_ALL(
        df,
        {
            "-D3MBJ(ATM)/aug-cc-pVDZ": "qz_diff_d3mbj_atm",
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
) -> None:
    """ """
    print(f"Plotting {pfn}")
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []

    for k, v in vals.items():
        df[v] = pd.to_numeric(df[v])
        vData.append(df[v].to_list())
        vLabels.append(k)

    # print(df[vals.values()].describe(include="all"))
    # transparent figure
    fig = plt.figure(dpi=1000)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)

    colors = [
        "blue",
        "red",
        "purple",
        "brown",
        "green",
        "orange",
        "pink",
        "grey",
        "yellow",
        "teal",
        "cyan",
        "navy",
    ]
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
        label=r"$\pm$1 $kcal\cdot mol^{-1}$",
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
        zorder=0,
    )
    navy_blue = (0.0, 0.32, 0.96)
    ax.set_xticks(xs)
    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="6")
    ax.set_xlim((0, len(vLabels)))
    lg = ax.legend(loc="upper left", edgecolor="black")
    lg.get_frame().set_alpha(None)
    lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    ax.set_xlabel("Level of Theory", color="k")
    ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)", color="k")
    ax.grid(color="gray", which="major", linewidth=0.5, alpha=0.3)
    ax.grid(color="gray", which="minor", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color("k")

    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=0.4)
    plt.savefig(f"plots/{pfn}_dbs_violin.png", transparent=False)
    plt.clf()
    return


def plot_dbs_d3_d4(df, c1, c2, l1, l2, title_name, pfn, outlier_cutoff=3) -> None:
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
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
            for n, r in df3.iterrows():
                print(
                    f"\n{r['DB']}, {r['System']}\nid: {r['id']}, error: {c2}={r[c2]:.2f}, {c1}={r[c1]:.2f}"
                )
                tools.print_cartesians(r["Geometry"])

        else:
            vDataErrors.append([])
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
        label=r"$\pm$1 $kcal\cdot mol^{-1}$",
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
    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="5")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    ax.grid(color="gray", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):
        if n % 2 != 0:
            xtick.set_color("blue")
        else:
            xtick.set_color("red")

    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return
