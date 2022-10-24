import pickle
import numpy as np
import pandas as pd
import qcelemental


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


def convert_geom_to_bohr(geom):
    c = qcelemental.constants.conversion_factor("angstrom", "bohr")
    # c = qcelemental.constants.conversion_factor("bohr", "angstrom")
    print(c)
    return np.array(geom, copy=True) * c


def hf_key_to_latex_cmd(hf_key, d4=True) -> str:
    """
    hf_key_to_latex_cmd
    """
    out = ""
    if d4:
        if hf_key == "HF_dz":
            out = "\\sdz"
        elif hf_key == "HF_jdz":
            out = "\\sjdz"
        elif hf_key == "HF_adz":
            out = "\\sadz"
        elif hf_key == "HF_tz":
            out = "\\stz"
        elif hf_key == "HF_atz":
            out = "\\satz"
        elif hf_key == "HF_jtz":
            out = "\\sjtz"
    else:
        if hf_key == "HF_dz":
            out = "\\sddz"
        elif hf_key == "HF_jdz":
            out = "\\sdjdz"
        elif hf_key == "HF_adz":
            out = "\\sdadz"
        elif hf_key == "HF_tz":
            out = "\\sdtz"
        elif hf_key == "HF_atz":
            out = "\\sdatz"
        elif hf_key == "HF_jtz":
            out = "\\sdjtz"
    return out


def stats_to_latex_row(name, rmse, max_e, mad, md):
    v = f"    {name}  &  {rmse:.4f}  &  {max_e:.4f}  &  {mad:.4f}  &  {md:.4f}\n"
    return v
