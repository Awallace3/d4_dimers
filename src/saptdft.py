import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from qcelemental import constants

from . import locald4


def failed(
    df,
) -> None:
    """
    failed
    """
    df = pd.read_pickle("data/schr_dft.pkl")
    print(df.columns.values)
    t = df[df["pbe0_adz_saptdft"].isna()]
    print(len(t))
    # print(t)
    t2 = t[t["pbe0_adz_cation_A"].isna()]
    # t2 = t[t["pbe0_adz_grac_A"].isna()]
    print("cation_A")
    print(t2[["pbe0_adz_cation_A", "pbe0_adz_cation_B"]])
    print(t2["main_id"].to_list())
    # t3 = t[t["pbe0_adz_grac_B"].isna()]
    t3 = t[t["pbe0_adz_cation_B"].isna()]
    print("cation_B")
    print(t3[["pbe0_adz_cation_A", "pbe0_adz_cation_B"]])
    print(t3["main_id"].to_list())
    errors = list(set(t2["main_id"].to_list()).union(t3["main_id"].to_list()))
    print(errors)
    print(len(errors))


def df_empty_cols(df, col="pbe0_adz_grac_A") -> None:
    """
    df_empty_gracs
    """
    a = df[df[col].isna()]
    print(len(a))
    return


def plot_BJ_damping(
    s8,
    a1,
    a2,
    s6=1.0,
    R_0_AB=3.9520731774875273,  # H interacting w/ O
    # R_0_AB=4.492276114883366, # O O
    # R_0_AB=5.37789343195611 , # C C
    # R_0_AB=4.324121142851815 , # C H
    # R_0_AB=4.915178756949189 , # C O
    # R_0_AB = sqrt(3 * Q_A * Q_B)
) -> None:
    """
    plot_BJ_damping
    """
    # R_0_ABs = [
    #     {v: 3.9520731774875273, label: "H O"},
    #     {v: 4.492276114883366, label: "O O"},
    #     {v: 5.37789343195611, label: "C C"},
    #     {v: 4.324121142851815, label: "C H"},
    #     {v: 4.915178756949189, label: "C O"},
    # ]
    R_0_ABs = [
        3.9520731774875273,
        4.492276114883366,
        5.37789343195611,
        4.324121142851815,
        4.915178756949189,
    ]

    dis = np.arange(0.01, 2, 0.01)
    # The parametrized function to be plotted
    def f(dis, a1, a2, R_0_AB):
        # return a1 * np.sin(2 * np.pi * a2 * dis)
        return dis / (dis + a1 * (R_0_AB + a2))

    # Define initial parameters
    init_a1 = a1
    init_a2 = a2

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    (line,) = ax.plot(dis, f(dis, init_a1, init_a2, R_0_ABs[0]), lw=2)
    # (line2,) = ax.plot(dis, f(dis, init_a1, init_a2, R_0_ABs[1]), lw=2)
    ax.set_xlabel("Time [s]")

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the a2.
    axa2 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    a2_slider = Slider(
        ax=axa2,
        label="a2",
        valmin=0.1,
        valmax=8,
        valinit=init_a2,
    )

    # Make a vertically oriented slider to control the a1
    axa1 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    a1_slider = Slider(
        ax=axa1,
        label="a1",
        valmin=0,
        valmax=2,
        valinit=init_a1,
        orientation="vertical",
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(f(dis, a1_slider.val, a2_slider.val, R_0_ABs[0]))
        # line2.set_ydata(f(dis, a1_slider.val, a2_slider.val, R_0_ABs[1]))
        fig.canvas.draw_idle()

    # register the update function with each slider
    a2_slider.on_changed(update)
    a1_slider.on_changed(update)

    def reset(event):
        a2_slider.reset()
        a1_slider.reset()

    plt.show()
    return
