import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
import qcelemental as qcel
from src import dftd3

def setup_los_df():
    df = pd.read_pickle("./plots/LoS.pkl")
    df["Geometry"] = df["Geometry"].apply(lambda x: np.array(x))
    df["monAs"] = df["monAs"].apply(lambda x: np.array(x))
    df["monBs"] = df["monBs"].apply(lambda x: np.array(x))
    df["charges"] = df["charges"].apply(lambda x: np.array(x))
    df.dropna(inplace=True, subset=["benchmark ref energy"])
    print(df)
    for n, i in df.iterrows():
        pos, carts = i["Geometry"][:, 0], i["Geometry"][:, 1:]
        monAs, monBs = i["monAs"], i["monBs"]
        C6s = dftd3.collect_bjm_d3data_dimer(pos, carts, monAs, monBs)
        print(C6s)
        break
    return


def main():
    setup_los_df()
    return


if __name__ == "__main__":
    main()
