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


def plot_dbs(df, df_col, title_name, pfn) -> None:
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

    fig = plt.figure(dpi=400)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(['cbars', 'cmins', 'cmaxes', 'cmeans']):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)

    for n, pc in enumerate(vplot["bodies"], 1):
        if n%2 != 0:
            pc.set_facecolor("blue")
        else:
            pc.set_facecolor("blue")
        pc.set_alpha(0.5)
        # pc.set_edgecolor("black")


    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        label="+-1 kcal/mol",
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
        label="+-1 kcal/mol",
        zorder=0,
    )
    ax.set_xticks(xs)
    plt.setp(ax.set_xticklabels(vLabels), rotation=30, fontsize="6")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel("Error (kcal/mol)")
    plt.title(f"{title_name} DBs Violin")
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return


def plot_dbs_d3_d4(df, c1, c2, l1, l2, title_name, pfn) -> None:
    """
    """
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    for d in dbs:
        df2 = df[df['DB'] == d]
        vData.append(df2[c1].to_list())
        vData.append(df2[c2].to_list())
        vLabels.append(f"{d} - {l1}")
        vLabels.append(f"{d} - {l2}")

    fig = plt.figure(dpi=400)
    ax = plt.subplot(111)
    vplot = ax.violinplot(vData, showmeans=True, showmedians=False)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(['cbars', 'cmins', 'cmaxes', 'cmeans']):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)

    for n, pc in enumerate(vplot["bodies"], 1):
        if n%2 != 0:
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
        label="+-1 kcal/mol",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        label="0 kcal/mol",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        label="+-1 kcal/mol",
        zorder=0,
    )
    ax.set_xticks(xs)
    plt.setp(ax.set_xticklabels(vLabels), rotation=30, fontsize="6")
    ax.set_xlim((0, len(vLabels)))
    ax.legend(loc="upper left")
    ax.set_xlabel("Database")
    ax.set_ylabel("Error (kcal/mol)")
    plt.title(f"{title_name} DBs Violin")
    # plt.show()
    plt.savefig(f"plots/{pfn}_dbs_violin.png")
    plt.clf()
    return
