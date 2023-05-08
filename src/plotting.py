import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
        df2 = df[df['DB'] == d]
        vData.append(df2[df_col].to_list())
        vLabels.append(d)

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(['cbars', 'cmins', 'cmaxes', 'cmeans']):
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
    plt.setp(ax.set_xticklabels(vLabels), rotation=45, fontsize="5")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel("Error (kcal/mol)")
    plt.title(f"{title_name}")
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return

def plot_dbs_d3_d4_two(df, c1, c2, l1, l2, title_name, pfn, first=True) -> None:
    """
    """
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    if first:
        # dbs = [dbs[i] for i in range(len(dbs)//2)]
        dbs = dbs[:len(dbs)//2]
    else:
        dbs = dbs[len(dbs)//2:]
    print(dbs)

    for d in dbs:
        df2 = df[df['DB'] == d]
        vData.append(df2[c1].to_list())
        vData.append(df2[c2].to_list())
        vLabels.append(f"{d} - {l1}")
        vLabels.append(f"{d} - {l2}")

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(['cbars', 'cmins', 'cmaxes', 'cmeans']):
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
    ax.set_ylim((-12,6))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    ax.grid(color='gray', linewidth=0.5, alpha=0.3)
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

def plot_dbs_d3_d4(df, c1, c2, l1, l2, title_name, pfn) -> None:
    """
    """
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    for d in dbs:
        df2 = df[df['DB'] == d]
        vData.append(df2[c1].to_list())
        vData.append(df2[c2].to_list())
        vLabels.append(f"{d} - {l1}")
        vLabels.append(f"{d} - {l2}")

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(['cbars', 'cmins', 'cmaxes', 'cmeans']):
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
    plt.setp(ax.set_xticklabels(vLabels), rotation=60, fontsize="5")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    ax.grid(color='gray', linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):

        if n % 2 != 0:
            xtick.set_color("blue")
        else:
            xtick.set_color("red")

    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=0.2)
    # plt.show()
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return


def plot_violin_d3_d4_total(df, c1, c2, l1, l2, title_name, pfn) -> None:
    """
    """
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    vData.append(df[c1].to_list())
    vData.append(df[c2].to_list())
    vLabels.append(f"{l1}")
    vLabels.append(f"{l2}")

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(['cbars', 'cmins', 'cmaxes', 'cmeans']):
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
    plt.setp(ax.set_xticklabels(vLabels), rotation=0, fontsize="8")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    # set yaxis as log
    # ax.set_yscale("log")
    ax.set_xlabel("Level of Theory")
    ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    # ax.grid(color='gray', linewidth=0.5, alpha=0.3)
    ax.grid(color='gray', which="both", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):

        if n % 2 != 0:
            xtick.set_color("blue")
        else:
            xtick.set_color("red")

    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=0.2)
    # plt.show()
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return


def plot_violin_d3_d4_total(
    df,
    c1s: [],
    l1s: [],
    title_name: str,
    pfn: str,
) -> None:
    """
    """
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    for i in range(len(c1s)):
        vData.append(df[c1s[i]].to_list())
        vLabels.append(f"{l1s[i]}")

    fig = plt.figure(dpi=800)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(['cbars', 'cmins', 'cmaxes', 'cmeans']):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)

    colors = ['blue', 'red', 'green']
    for n, pc in enumerate(vplot["bodies"], 1):
        # if n % 2 != 0:
        #     pc.set_facecolor("blue")
        # else:
        #     pc.set_facecolor("red")
        pc.set_facecolor(colors[n-1])
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
    plt.setp(ax.set_xticklabels(vLabels), rotation=15, fontsize="9")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    # set yaxis as log
    # ax.set_yscale("log")
    ax.set_xlabel("Level of Theory")
    ax.set_ylabel(r"Error ($kcal\cdot mol^{-1}$)")
    # ax.set_ylabel(r"Error ($\frac{kcal}{mol}$)")
    ax.grid(color='gray', which="major", linewidth=0.5, alpha=0.3)
    ax.grid(color='gray', which="minor", linewidth=0.5, alpha=0.3)
    for n, xtick in enumerate(ax.get_xticklabels()):

        # if n % 2 != 0:
        #     xtick.set_color("blue")
        # else:
        #     xtick.set_color("red")
        xtick.set_color(colors[n-1])

    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=0.2)
    # plt.show()
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return
