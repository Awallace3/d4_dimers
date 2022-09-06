import pickle
import numpy as np
import pandas as pd


def np_carts_to_string(carts):
    w = ""
    for n, r in enumerate(carts):
        e, x, y, z = r
        line = "{:d}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(int(e), x, y, z)
        w += line
    return w


def print_cartesians(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for i, elem in enumerate(a):
            if i == 0:
                print("{} ".format(int(elem)), end="\t")
            else:
                print("{:.10f} ".format(elem).rjust(3), end="\t")
        print(end="\n")


def print_cartesians_pos_carts(pos: np.array, carts: np.array):
    """
    prints a 2-D numpy array in a nicer format
    """
    print()
    for n, r in enumerate(carts):
        x, y, z = r
        line = "{}\t{:.10f}\t{:.10f}\t{:.10f}".format(int(pos[n]), x, y, z)
        print(line)
    print()


def write_pickle(data, fname="data.pickle"):
    with open(fname, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fname="data.pickle"):
    with open(fname, "rb") as handle:
        return pickle.load(handle)


def df_to_latex_table_round(
    df,
    index_col="method",
    cols_round={"RMSE": 4, "MAX_E": 4, "MAD": 4, "MD": 4},
    l_out="out",
):
    df = df.set_index(index_col)
    s = df.style
    for k, v in cols_round.items():
        s.format(subset=k, precision=v)
    print(s.to_latex())
    with open(f"latex_outfiles/{l_out}.tex", "w") as f:
        f.write(s.to_latex())
    return
