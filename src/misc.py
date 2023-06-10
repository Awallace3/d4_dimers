import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qm_tools_aw import tools


def find_charged_systems():
    df = pd.read_pickle("data/d4.pkl")
    c = np.array([[0, 1] for i in range(3)])
    cnt_charged = 0
    for i, r in df.iterrows():
        if not np.array_equal(r["charges"], c):
            print(i, r["charges"])
            cnt_charged += 1
    print(f"Found {cnt_charged} charged systems")
    return


def print_geom_by_id(df, id):
    tools.print_cartesians(df.loc[id]["Geometry"], symbols=True)
    return


def main():
    # find_charged_systems()
    print_geom_by_id(pd.read_pickle("data/d4.pkl"), 4926 )
    return


if __name__ == "__main__":
    main()
