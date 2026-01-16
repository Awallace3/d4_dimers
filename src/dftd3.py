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
    print(f"Setting DFTD3 parameters to...\n{params}")
    hostname = os.uname()[1]
    home = os.environ["HOME"]
    fn = f"{home}/.dftd3par.{hostname}"
    with open(fn, "w") as f:
        f.write(f"{' '.join([f'{i:.6f}' for i in params])}\n")
    return params


def s_dftd3_params(param_label):
    if param_label == "D3MBJ":
        params = [1.000000, 0.713108, 0.079643, 3.627271]
    elif param_label == "D3MBJ ATM":
        params = [1.000000, 0.713108, 0.079643, 3.627271]
    else:
        raise ValueError(f"param_label {param_label} not recognized")
    return params


def dftd3_bjm(pos, carts, params, ATM=False):
    with open("tmp.xyz", "w") as f:
        f.write(tools.carts_to_xyz(pos, carts))
    params = [str(i) for i in params]
    if ATM:
        cmd = ["s-dftd3", "--bj", "hf", "--bj-param", *params, "--atm", "tmp.xyz"]
    else:
        cmd = ["s-dftd3", "--bj", "hf", "--bj-param", *params, "tmp.xyz"]
    proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(
        ["grep", "Dispersion energy:"], stdin=proc1.stdout, stdout=subprocess.PIPE
    )
    proc1.stdout.close()
    proc3 = subprocess.Popen(
        ["awk", "{ print $3 }"], stdin=proc2.stdout, stdout=subprocess.PIPE
    )
    proc2.stdout.close()
    output = proc3.communicate()[0]
    e_disp = float(output.decode("utf-8").strip())
    os.remove("tmp.xyz")
    return e_disp


def collect_bjm_d3data(pos, carts, ATM=False, s_dftd3_bin=None):
    if s_dftd3_bin is None:
        s_dftd3_bin = "s-dftd3"
    with open("tmp.xyz", "w") as f:
        f.write(tools.carts_to_xyz(pos, carts))
    if ATM:
        cmd = [s_dftd3_bin, "--bj", "hf", "--pair-resolved", "--atm", "tmp.xyz"]
    else:
        cmd = [s_dftd3_bin, "--bj", "hf", "--pair-resolved", "tmp.xyz"]
    # proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # proc1.wait()
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # print(proc1.stdout.read())
    data = tools.json_to_dict("d3data.json")
    # os.remove("tmp.xyz")
    os.remove("d3data.json")
    # read energy from .EDISP file
    with open(".EDISP", "r") as f:
        lines = f.readlines()
    e_disp = float(lines[-1].strip().split()[-1]) * hartree_to_kcalmol
    output = []
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            output.append(
                [
                    i + 1,
                    j + 1,
                    data["rs"][j, i],
                    data["r0s"][j, i],
                    data["c6s"][j, i],
                    data["c8s"][j, i],
                ]
            )
    return data, np.array(output), e_disp


def collect_bjm_d3data_dimer_intermolecular(pos, carts, monAs, monBs, ATM=False, s_dftd3_bin=None):
    if s_dftd3_bin is None:
        s_dftd3_bin = "s-dftd3"
    with open("tmp.xyz", "w") as f:
        f.write(tools.carts_to_xyz(pos, carts))
    if ATM:
        cmd = [s_dftd3_bin, "--bj", "hf", "--pair-resolved", "--atm", "tmp.xyz"]
    else:
        cmd = [s_dftd3_bin, "--bj", "hf", "--pair-resolved", "tmp.xyz"]
    # proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # proc1.wait()
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # print(proc1.stdout.read())
    data = tools.json_to_dict("d3data.json")
    os.remove("tmp.xyz")
    os.remove("d3data.json")
    output = []
    for i in monAs:
        for j in monBs:
            output.append(
                [
                    i + 1,
                    j + 1,
                    data["rs"][j, i],
                    data["r0s"][j, i],
                    data["c6s"][j, i],
                    data["c8s"][j, i],
                ]
            )
    # return data, np.array(output), e_disp
    return np.array(output)


def filter_d3data_intermolecular(d3data, monAs, monBs):
    """
    Filter d3data to include only intermolecular pairs (one atom from A, one from B).
    
    Parameters
    ----------
    d3data : np.ndarray
        Output from collect_bjm_d3data with columns [i, j, r, r0, c6, c8]
        where i, j are 1-indexed atom indices
    monAs : array-like
        0-indexed atom indices for monomer A
    monBs : array-like
        0-indexed atom indices for monomer B
    
    Returns
    -------
    np.ndarray
        Filtered d3data with only intermolecular pairs
    """
    if d3data is None or len(d3data) == 0:
        return np.array([])
    # Convert monAs/monBs to sets for O(1) lookup, and to 1-indexed
    monAs_set = set(i + 1 for i in monAs)
    monBs_set = set(i + 1 for i in monBs)
    
    intermolecular = []
    for row in d3data:
        i, j = int(row[0]), int(row[1])
        # Check if one atom is in A and other is in B (either direction)
        if row[-1] < 0:
            break
        if (i in monAs_set and j in monBs_set):
            intermolecular.append(row)
    
    return np.array(intermolecular)


def collect_bjm_d3data_dimer(pos, carts, monAs, monBs, ATM=False, s_dftd3_bin=None):
    _, dimer_d3data, _ = collect_bjm_d3data(
        pos, carts, ATM=False, s_dftd3_bin=s_dftd3_bin
    )
    v = [dimer_d3data]
    _, monA_d3data, _ = collect_bjm_d3data(
        pos[monAs], carts[monAs], ATM=False, s_dftd3_bin=s_dftd3_bin
    )
    # check if monA_d3data is empty, if so skip multiplying by -1. Ions don't
    # have any meaningful pairwise interactions, so collect_bjm_d3data returns
    # an empty array. We can just skip it
    if len(monA_d3data) > 0:
        monA_d3data[:, -2:] *= -1
        v.append(monA_d3data)
    _, monB_d3data, _ = collect_bjm_d3data(
        pos[monBs], carts[monBs], ATM=False, s_dftd3_bin=s_dftd3_bin
    )
    if len(monB_d3data) > 0:
        monB_d3data[:, -2:] *= -1
        v.append(monB_d3data)
    return np.concatenate(v)


def dftd3_bjm_og(pos, carts, ATM=False):
    with open("tmp.xyz", "w") as f:
        f.write(tools.carts_to_xyz(pos, carts))
    if ATM:
        cmd = ["dftd3", "--bj", "--atm", "tmp.xyz"]
    else:
        cmd = ["dftd3", "--bjm", "tmp.xyz"]
    print(" ".join(cmd))
    proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
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
    params = s_dftd3_params(param_label)
    ATM = False
    if param_label == "D3MBJ ATM":
        ATM = True

    for n, i in df.iterrows():
        mol = i[geom_column]
        geom, ma, mb, charges = i["Geometry"], i["monAs"], i["monBs"], i["charges"]
        pD, cD = geom[:, 0], geom[:, 1:]
        dftd3_d = dftd3_bjm(pD, cD, params, ATM=ATM)
        dftd3_a = dftd3_bjm(pD[ma], cD[ma, :], params, ATM=ATM)
        dftd3_b = dftd3_bjm(pD[mb], cD[mb, :], params, ATM=ATM)
        dftd3_ie = (dftd3_d - (dftd3_a + dftd3_b)) * hartree_to_kcalmol
        print(f"{n}, {param_label}: {dftd3_ie}")
        dftd3.append(dftd3_ie)
    df[param_label] = dftd3
    df.to_pickle(out_pkl)
    return
