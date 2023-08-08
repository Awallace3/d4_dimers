import numpy as np
import pandas as pd
from qm_tools_aw import tools
import subprocess
import os
from .locald4 import hartree_to_kcalmol


def set_dftd3_params(param_label):
    if param_label == "D3MBJ":
        params = [1.000000, 0.079643, 0.713108, 3.627271, 0.000000, 6.00000]
    elif param_label == "D3MBJ ATM":
        params = [1.000000, 0.079643, 0.713108, 3.627271, 1.000000, 6.00000]
    else:
        raise ValueError(f"param_label {param_label} not recognized")
    hostname = os.uname()[1]
    home = os.environ["HOME"]
    fn = f"{home}/.dftd3par.{hostname}"
    with open(fn, "w") as f:
        f.write(f"{' '.join([f'{i:.6f}' for i in params])}\n")
    return


def dftd3_bjm(pos, carts):
    with open("tmp.xyz", "w") as f:
        f.write(tools.carts_to_xyz(pos, carts))
    proc1 = subprocess.Popen(["dftd3", "tmp.xyz", "-bjm"], stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(
        ["grep", "Edisp"], stdin=proc1.stdout, stdout=subprocess.PIPE
    )
    proc1.stdout.close()
    proc3 = subprocess.Popen(
        ["awk", "{print $4}"], stdin=proc2.stdout, stdout=subprocess.PIPE
    )
    proc2.stdout.close()
    output = proc3.communicate()[0]
    e_disp = float(output.decode("utf-8").strip())
    os.remove("tmp.xyz")
    return e_disp


def compute_dftd3(df, out_pkl, geom_column, param_label="D3MBJ ATM"):
    """
    compute_dftd3 computes D3MBJ energy for each protein
    """
    dftd3 = []
    set_dftd3_params(param_label)
    for n, i in df.iterrows():
        mol = i[geom_column]
        geom, ma, mb, charges = i["Geometry"], i["monAs"], i["monBs"], i["charges"]
        pD, cD = geom[:, 0], geom[:, 1:]
        dftd3_d = dftd3_bjm(pD, cD)
        dftd3_a = dftd3_bjm(pD[ma], cD[ma, :])
        dftd3_b = dftd3_bjm(pD[mb], cD[mb, :])
        dftd3_ie = (dftd3_d - (dftd3_a + dftd3_b)) * hartree_to_kcalmol
        print(f"{n}, {param_label}: {dftd3_ie}")
        dftd3.append(dftd3_ie)
    df[param_label] = dftd3
    df.to_pickle(out_pkl)
    return
