import numpy as np
import pandas as pd
import pickle
from pprint import pprint as pp
from qm_tools_aw import tools
import pymol

def molecule_figures(df):
    """
    Plots figures for each molecule
    """
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []
    loc = range(5)
    for d in dbs:
        df2 = df[df['DB'] == d]
        for i in loc:
            g = df2.iloc[i]['Geometry']
            fd = d.lower().replace(" ", "_")
            fname = f"molImages/{fd}_{i}.xyz"
            print(fname)
            gs = tools.write_cartesians_to_xyz(g[:,0], g[:,1:], fname)
    return

def main():
    """
    docstring
    """
    pkl_name = "data/schr_dft.pkl"
    df = pd.read_pickle(pkl_name)
    print(df.columns)
    molecule_figures(df)
    return


if __name__ == "__main__":
    main()
