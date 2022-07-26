import numpy as np
import pandas as pd
from periodictable import elements
from .r4r2 import get_Q, r4r2_from_elements_call, r4r2_vals, r4r2_ls
from .tools import print_cartesians, print_cartesians_pos_carts, np_carts_to_string
import subprocess
import json
import math
from .constants import Constants
from tqdm import tqdm
from psi4.driver.qcdb.bfs import BFS
import os
from .harvest import ssi_bfdb_data, harvest_data
import psi4
from qcelemental import constants


def inpsect_master_regen():
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    pkl_path = "master-regen.pkl"
    ms = pd.read_pickle(pkl_path)
    # ms = ms[ms["DB"] != "PCONF"]
    # ms = ms[ms["DB"] != "SCONF"]
    # ms = ms[ms["DB"] != "ACONF"]
    # ms = ms[ms["DB"] != "CYCONF"]
    # for i in ms.columns.values:
    #     print(i)
    # ms = ms[ms["DB"] == "SSI"]
    # for idx, i in ms.iterrows():
    #     if int(ms.loc[idx, "R"]) != 1:
    #         print(ms.loc[idx])
    print(ms.columns.values)
    # for i in ms.columns.values:
    #     if "disp" in i.lower():
    #         print(i, 'NaNs =', ms[i].isna().sum())
    #         ms["t"] = ms.apply(
    #             lambda r: abs(r["Benchmark"] - (r["HF INTERACTION ENERGY"] + r[i])),
    #             axis=1,
    #         )
    #         print(i, ms["t"].mean(), (ms["t"] ** 2).mean() ** 0.5, ms["t"].max())
    return ms


def gather_data(
    pkl_path: str = "master-regen.pkl",
    csv_path: str = "SAPT-D3-Refit-Data.csv",
    out_pkl: str = "data.pkl",
):
    """
    gather_data collects pkl data

    use gather_data3 for best output data from master-regen for other functions
    """
    ms = pd.read_pickle(pkl_path)
    ms = ms[ms["DB"] != "PCONF"]
    ms = ms[ms["DB"] != "SCONF"]
    ms = ms[ms["DB"] != "ACONF"]
    ms = ms[ms["DB"] != "CYCONF"]

    df = pd.read_csv("SAPT-D3-Refit-Data.csv")
    out_df = ms[["Benchmark", "HF INTERACTION ENERGY", "Geometry"]]
    energies = ms[["Benchmark", "HF INTERACTION ENERGY"]].to_numpy()
    carts = ms["Geometry"].to_list()
    atom_order = [np.array(i[:, 0]) for i in carts]
    carts = [i[:, 1:] for i in carts]

    data = [energies, atom_order, carts]
    C6s, C8s, Qs = [], [], []
    for i in tqdm(
        range(len(data[0])),
        desc="DFTD4 Props",
        ascii=True,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        C6, Q = calc_dftd4_props(data[1][i], data[2][i])
        C6s.append(C6)
        Qs.append(Q)
    C6s = C6s
    Qs = Qs
    d4_vals = d4_values(C6s, Qs, C8s)
    data_out = mols_data(energies, atom_order, carts, d4_vals)
    write_pickle(data_out, out_pkl)
    return data_out


def write_xyz_from_np(atom_numbers, carts, outfile="dat.xyz", charges=[0, 1]) -> None:
    """
    write_xyz_from_np
    """
    with open(outfile, "w") as f:
        f.write(str(len(carts)) + "\n\n")
        for n, i in enumerate(carts):
            el = str(int(atom_numbers[n]))
            v = "    ".join(["%.16f" % k for k in i])
            line = "%s    %s\n" % (el, v)
            f.write(line)
    return


def mol_testing(mol):
    params = [1.61679827, 0.44959224, 3.35743605]
    geom = mol["Geometry"]
    Ma = mol["monAs"]
    Mb = mol["monBs"]

    atoms = geom[:, 0]
    carts = geom[:, 1:]
    C6s, C8s = calc_dftd4_props(atoms, carts)
    print(np.array_equal(C6s, mol["C6s"]))
    print(C6s[0])
    print(mol["C6s"][0])
    energy = compute_bj_pairs(params, atoms, carts, Ma, Mb, C6s)
    print("HF_jdz", mol["HF_jdz"])
    print("pairs =", energy)
    print("EI =", mol["HF_jdz"] + energy)


def compute_pairwise_dispersion(m):
    pos = m["Geometry"][:, 0]
    carts = m["Geometry"][:, 1:]
    Ma = m["monAs"]
    Mb = m["monBs"]
    C6s, pairs, e = calc_dftd4_pair_resolved(pos, carts)
    mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
    C6_A, pairs_a, e = calc_dftd4_pair_resolved(mon_pa, mon_ca)
    mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
    C6_B, pairs_b, e = calc_dftd4_pair_resolved(mon_pb, mon_cb)
    dp = pairs.sum()
    ap = pairs_a.sum()
    bp = pairs_b.sum()
    disp = dp - (ap + bp)
    print(m["main_id"], disp, m["dftd4_disp_ie_grimme_params"], m["Benchmark"])
    # print("dp:", dp)
    # print("ap:", ap)
    # print("bp:", bp)
    # print("disp:", disp)
    return pairs, pairs_a, pairs_b, disp


def calc_dftd4_pair_resolved(
    atom_numbers: np.array,
    carts: np.array,
    input_xyz: str = "dat.xyz",
    output_data: str = "dat.txt",
    p: [] = [1.61679827, 0.44959224, 3.35743605],
):
    # carts = convert_geom_to_bohr(carts)
    write_xyz_from_np(atom_numbers, carts, outfile=input_xyz)
    args = [
        # "dftd4",
        "/theoryfs2/ds/amwalla3/miniconda3/bin/dftd4",
        input_xyz,
        "--pair-resolved",
        "--property",
        "--mbdscale",
        "0.0",
        "--param",
        "1.0",
        str(p[0]),
        str(p[1]),
        str(p[2]),
        "--json",
        "C_n.json",
    ]
    out = subprocess.run(
        args=args,
        shell=False,
        capture_output=True,
    ).stdout.decode("utf-8")
    # print(out)
    d = out.split("Pairwise representation")[-1]
    d = d.split("\n")
    clean = []
    start = False
    vs = []
    for i in d[4:]:
        if "------" in i:
            break
        else:
            start = True
        i = i.split()
        p1, p2, e = int(i[0]), int(i[3]), float(i[-1])
        vs.append([p1, p2, e])
    M = vs[-1][0]
    m = np.zeros((M, M))
    for i in vs:
        m[i[0] - 1, i[1] - 1] = i[2]
    with open("C_n.json") as f:
        dat = json.load(f)
        # C6s = np.array(dat["c6"])
        C6s = np.array(dat["c6 coefficients"])
    with open(".EDISP", "r") as f:
        e = float(f.read().rstrip())
    os.remove("C_n.json")
    os.remove(".EDISP")
    return C6s, m, e


def calc_dftd4_props_params(
    atom_numbers: np.array,
    carts: np.array,
    input_xyz: str = "dat.xyz",
    output_json: str = "",
    p: [] = [1.61679827, 0.44959224, 3.35743605],
    s9: str = "0.0",
    dftd4_p: str = "dftd4",
):
    # mult_out=constants.conversion_factor("hartree", "kcal / mol"),
    write_xyz_from_np(atom_numbers, carts, outfile=input_xyz)
    args = [
        dftd4_p,
        input_xyz,
        "--pair-resolved",
        "--property",
        "--mbdscale",
        s9,
        "--param",
        "1.0",
        str(p[0]),
        str(p[1]),
        str(p[2]),
    ]
    out = subprocess.run(
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    with open("C_n.json") as f:
        dat = json.load(f)
        C6s = np.array(dat["c6"])
        C8s = np.array(dat["c8"])
    with open(".EDISP") as f:
        e = float(f.read().replace("\n", ""))
    if output_json != "":
        os.remove(input_xyz)
    return C6s, C8s, e


def calc_dftd4_c6_c8_pairDisp2(
    atom_numbers: np.array,
    carts: np.array,
    charges: np.array,
    input_xyz: str = "dat.xyz",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
):
    """
    Ensure that dftd4 binary is from compiling git@github.com:Awallace3/dftd4
        - this is used to generate more decimal places on values for c6, c8,
          and pairDisp2
    """

    write_xyz_from_np(atom_numbers, carts, outfile=input_xyz, charges=charges)
    args = [
        "/theoryfs2/ds/amwalla3/.local/bin/dftd4",
        input_xyz,
        "--property",
        "--param",
        str(p[0]),
        str(p[1]),
        str(p[2]),
        str(p[3]),
        "--mbdscale",
        "0.0",
        "-c",
        str(charges[0]),
        "--pair-resolved",
    ]
    subprocess.call(
        # cmd,
        # shell=True,
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    output_json = "C_n.json"
    with open(output_json) as f:
        cs = json.load(f)
    C6s = np.array(cs["c6"])
    C8s = np.array(cs["c8"])
    output_json = "pairs.json"
    with open(output_json) as f:
        pairs = json.load(f)
        pairs = np.array(pairs["pairs2"])
    os.remove(input_xyz)
    os.remove("C_n.json")
    os.remove("pairs.json")
    return C6s, C8s, pairs


def calc_dftd4_props_psi4_dftd4(
    atom_numbers: np.array,
    carts: np.array,
    charges: np.array,
    input_xyz: str = "dat.xyz",
    output_json: str = "tmp.json",
):
    write_xyz_from_np(atom_numbers, carts, outfile=input_xyz, charges=charges)
    if output_json == "":
        # args = ["dftd4", input_xyz, "--property", "--mbdscale", "0.0"]
        args = [
            "/theoryfs2/ds/amwalla3/miniconda3/bin/dftd4",
            input_xyz,
            "--property",
            "--mbdscale",
            "0.0",
            "-c",
            str(charges[0]),
        ]
        # cmd = "~/.local/bin/dftd4 %s --property" % (input_xyz)
    else:
        args = [
            "/theoryfs2/ds/amwalla3/miniconda3/bin/dftd4",
            input_xyz,
            "--property",
            "--json",
            output_json,
            "--mbdscale",
            "0.0",
            "-c",
            str(charges[0]),
        ]
        # cmd = "~/.local/bin/dftd4 %s --property --json %s" % (input_xyz, output_json)
    subprocess.call(
        # cmd,
        # shell=True,
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    with open(output_json) as f:
        dat = json.load(f)
        C6s = np.array(dat["c6 coefficients"])
        n = int(np.sqrt(len(C6s)))
        C6s = np.reshape(C6s, (n, n))
        polariz = np.array(dat["polarizibilities"])
        # C8s = np.array(dat["c8"])
    if output_json != "":
        os.remove(input_xyz)
    return C6s, polariz


def calc_dftd4_props(
    atom_numbers: np.array,
    carts: np.array,
    input_xyz: str = "dat.xyz",
    output_json: str = "",
):
    write_xyz_from_np(atom_numbers, carts, outfile=input_xyz)
    if output_json == "":
        # args = ["dftd4", input_xyz, "--property", "--mbdscale", "0.0"]
        args = ["dftd4", input_xyz, "--property", "--mbdscale", "0.0"]
        # cmd = "~/.local/bin/dftd4 %s --property" % (input_xyz)
    else:
        args = [
            "dftd4",
            input_xyz,
            "--property",
            "--json",
            output_json,
            "--mbdscale",
            "0.0",
        ]
        # cmd = "~/.local/bin/dftd4 %s --property --json %s" % (input_xyz, output_json)
    subprocess.call(
        # cmd,
        # shell=True,
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    with open("C_n.json") as f:
        dat = json.load(f)
        C6s = np.array(dat["c6"])
        C8s = np.array(dat["c8"])
    if output_json != "":
        os.remove(input_xyz)
    return C6s, C8s


def read_master_regen(
    pkl_path: str = "master-regen.pkl",
    csv_path: str = "SAPT-D3-Refit-Data.csv",
    out_pkl: str = "data.pkl",
):
    ms = pd.read_pickle(pkl_path)
    ms = ms[ms["DB"] != "PCONF"]
    ms = ms[ms["DB"] != "SCONF"]
    ms = ms[ms["DB"] != "ACONF"]
    ms = ms[ms["DB"] != "CYCONF"]
    print(ms["Name"])
    for i in ms.columns.values:
        print(i)
        # print(ms[i])
    return


def databases_dimers():
    names = [
        "ACHC",
        "C2H4_NT",
        "CH4_PAH",
        "CO2_NPHAC",
        "CO2_PAH",
        "HBC6",
        "NBC10ext",
        "S22by7",
        "S66by10",
        "Water2510",
        "X31by10",
    ]
    return names


def databases():
    names = [
        "ACHC",
        "ACONF",
        "C2H4_NT",
        "CH4_PAH",
        "CO2_NPHAC",
        "CO2_PAH",
        "CYCONF",
        "HBC6",
        "NBC10ext",
        "PCONF",
        "S22by7",
        "S66by10",
        "SCONF",
        "Water2510",
        "X31by10",
    ]
    return names


def combine_dimer_csvs():
    path_db = "data/Databases/"
    dbs = databases_dimers()
    frames = []
    for i in dbs:
        p = "%s%s/data.csv" % (path_db, i)
        df = pd.read_csv(p)
        frames.append(df)
    df = pd.concat(frames)
    print(df.head())
    print(len(df["DB"]))
    # df.to_pickle("condensed.pkl")
    return df


def create_data_csv():
    path_db = "data/Databases/"
    dbs = databases()
    for i in dbs:
        db_start = path_db + i + "/"
        csv = db_start + "benchmark_data.csv"
        condensed_csv = db_start + "data.csv"
        fp = db_start + "Geometries/*dimer*"
        df = pd.read_csv(csv)
        df = df[["DB", "System #", "z", "Benchmark", "System"]]
        df[
            [
                "sapt0_N",
                "HF_jun_DZ",
                "HF_aug_DZ",
                "HF_cc_PVDZ",
                "HF_cc_PTDZ",
                "HF_aug_cc_PTDZ",
            ]
        ] = np.nan

        xyz = db_start + "Geometries/" + i + "_"
        df = generate_xyz_lists(df, xyz)
        for i in df["xyz_d"]:
            if i != "":
                with open(i, "r") as f:
                    pass

        df.to_csv(condensed_csv)
    df = combine_dimer_csvs()
    return


def convert_zs_HBC6(
    z,
) -> str:
    """
    convert_zs returns z for HBC6
    """
    z2 = "%.2f" % z
    if z2[-1] == "0" and z2[-2] != ".":
        v = z2[:-1]
    else:
        v = z2
    return v


def generate_xyz_lists(df: pd.DataFrame, xyz: str):
    """
    generates the xyz lists for a given db from the pandas dataframe
    """
    if (
        df["DB"].iat[0] == "SCONF"
        or df["DB"].iat[0] == "PCONF"
        or df["DB"].iat[0] == "CYCONF"
        or df["DB"].iat[0] == "ACONF"
    ):
        df["xyz_d"] = ""
        df["xyz1"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_reagentA.xyz", axis=1
        )
        df["xyz2"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_reagentB.xyz", axis=1
        )
    elif df["DB"].iat[0] == "HBC6" or df["DB"].iat[0] == "NBC10ext":
        df["xyz_d"] = df.apply(
            lambda x: xyz
            + str(x["System #"])
            + "_"
            + convert_zs_HBC6(x["z"])
            + "_dimer.xyz",
            axis=1,
        )
        df["xyz1"] = df.apply(
            lambda x: xyz
            + str(x["System #"])
            + "_"
            + convert_zs_HBC6(x["z"])
            + "_monomerA.xyz",
            axis=1,
        )
        df["xyz2"] = df.apply(
            lambda x: xyz
            + str(x["System #"])
            + "_"
            + convert_zs_HBC6(x["z"])
            + "_monomerB.xyz",
            axis=1,
        )
        # df["xyz2"] = ""
    elif df["z"].isna().sum() > 1 or df["DB"].iat[0] == "Water2510":
        df["xyz_d"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_dimer.xyz", axis=1
        )
        df["xyz1"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_monomerA.xyz", axis=1
        )
        df["xyz2"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_monomerB.xyz", axis=1
        )
    else:
        df["xyz_d"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_" + str(x["z"]) + "_dimer.xyz",
            axis=1,
        )
        df["xyz1"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_" + str(x["z"]) + "_monomerA.xyz",
            axis=1,
        )
        df["xyz2"] = df.apply(
            lambda x: xyz + str(x["System #"]) + "_" + str(x["z"]) + "_monomerB.xyz",
            axis=1,
        )
    return df


def construct_xyz_lookup(
    pkl_path: str = "master-regen.pkl",
    csv_path: str = "SAPT-D3-Refit-Data.csv",
    out_pkl: str = "data.pkl",
):
    ms = pd.read_pickle(pkl_path)
    ms = ms[ms["DB"] != "PCONF"]
    ms = ms[ms["DB"] != "SCONF"]
    ms = ms[ms["DB"] != "ACONF"]
    ms = ms[ms["DB"] != "CYCONF"]

    df = pd.read_csv("SAPT-D3-Refit-Data.csv")
    out_df = ms[
        [
            "DB",
            "System",
            "System #",
            "Benchmark",
            "HF INTERACTION ENERGY",
            "Geometry",
        ]
    ]
    create_data_csv()

    # energies = ms[["Benchmark", "HF INTERACTION ENERGY"]].to_numpy()
    # carts = ms["Geometry"].to_list()
    return


def remove_extra_wb(line: str):
    """
    Removes extra whitespace in a string for better splitting.
    """
    line = (
        line.replace("    ", " ")
        .replace("   ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("\n ", "\n")
    )
    return line


def create_pt_dict():
    """
    create_pt_dict creates dictionary for string elements to atomic number.
    """
    el_dc = {}
    for el in elements:
        el_dc[el.symbol] = el.number
    return el_dc


def convert_str_carts_np_carts(carts: str, el_dc: dict = create_pt_dict()):
    """
    This will take Cartesian coordinates as a string and convert it to a numpy
    array.
    """
    carts = remove_extra_wb(carts)
    carts = carts.split("\n")
    if carts[0] == "":
        carts = carts[1:]
    if carts[-1] == "":
        carts = carts[:-1]
    ca = []
    for n, line in enumerate(carts):
        a = line.split()
        for j in range(len(a)):
            if a[j].isalpha():
                if len(a[j]) > 1:
                    a[j] = a[j][:-1] + a[j][-1].lower()
                a[j] = el_dc[a[j]]
            else:
                a[j] = float(a[j])
        ca.append(a)
    ca = np.array(ca)
    return ca


def read_xyz(xyz_path: str, el_dc: dict = create_pt_dict()) -> np.array:
    """
    read_xyz takes a path to xyz and returns np.array
    """
    with open(xyz_path, "r") as f:
        dat = "".join(f.readlines()[2:])
    geom = convert_str_carts_np_carts(dat, el_dc)
    return geom


def distance_3d(r1, r2):
    return np.linalg.norm(r1 - r2)


def target_C8_AA() -> None:
    """
    target_C8_AA
    """
    return [
        666.6492,
        726.1834,
        683.1565,
        680.7375,
        721.4485,
        680.8078,
        671.4554,
        734.7153,
        670.8260,
        17.2193,
        760.6473,
        8.7298,
        8.6837,
        19.0139,
        10.5469,
        655.3271,
        607.8381,
        643.4876,
        681.1332,
        897.9817,
        834.4209,
        9.9954,
        414.9061,
        768.0699,
        9.1092,
        9.8194,
        19.2866,
        18.7959,
    ]


def print_C6s_C8s(
    C6s,
    C8s,
    M_tot,
    pos,
) -> None:
    """
    print_C6s_C8s prints C6s and C8s next to target diagonals
    """
    print("Z     C6s\tC8s\t\tC8_target")
    t_C8 = target_C8_AA()
    for i in range(M_tot):
        l = "%d     %.4f\t%.4f\t\t%.4f" % (int(pos[i]), C6s[i, i], C8s[i, i], t_C8[i])
        print(l)


def compute_bj_opt(
    params: [],
    pos: np.array,
    carts: np.array,
    C6s: np.array,
    C8s: np.array,
    monAs: np.array,
    monBs: np.array,
) -> float:
    """
    compute_bj_opt computes energy from C6s, cartesian coordinates, and C8s for
    optimization from sklearn
    """
    s8, a1, a2 = params
    s6 = 1
    N_tot = len(carts)
    aatoau = Constants().g_aatoau()
    energy = 0
    cs = aatoau * np.array(carts, copy=True)
    # print(s8, a1, a2, "\nmolecule\n")
    for i in monAs:
        # print(i)
        for j in monBs:
            # print(i, j)
            C6 = C6s[i, j]
            C8 = C8s[i, j]
            r1, r2 = cs[i, :], cs[j, :]
            R = np.linalg.norm(r1 - r2)
            R0 = np.sqrt(C8 / C6)
            e6 = C6 / (R**6.0 + (a1 * R0 + a2) ** 6.0)
            # print(e6)
            e8 = s8 * C8 / (R**8.0 + (a1 * R0 + a2) ** 8.0)
            # print(e8, C8)
            energy += e6
            energy += e8

    energy *= -1
    # energy *= -1
    return energy


def create_mon_geom(
    pos,
    carts,
    M,
) -> (np.array, np.array):
    """
    create_mon_geom creates pos and carts from dimer
    """
    # mon_carts = np.zeros((len(M), 3))
    # mon_pos = np.zeros(len(M))
    # for n, i in enumerate(M):
    #     mon_carts[n] = carts[i]
    #     mon_pos[n] = pos[i]
    mon_carts = carts[M]
    mon_pos = pos[M]
    return mon_pos, mon_carts


def compute_bj_mons(
    params: [],
    pos: np.array,
    carts: np.array,
    M: [],  # number of atoms in monomer
    C6s: np.array,
) -> float:
    """
    compute_bj_mon computes energy from C6s, cartesian coordinates, and monomers.
    """
    energy = 0
    mon_carts = np.zeros((len(M), 3))
    mon_pos = np.zeros(len(M))
    C6_mon = np.zeros((len(M), len(M)))
    for n, i in enumerate(M):
        mon_carts[n] = carts[i]
        mon_pos[n] = pos[i]

    s8, a1, a2 = params
    s6 = 1.0
    M_tot = len(mon_carts)
    energies = np.zeros(M_tot)
    lattice_points = 1
    aatoau = Constants().g_aatoau()
    cs = aatoau * np.array(mon_carts, copy=True)
    for i in range(M_tot):
        el1 = int(pos[i])
        el1_r4r2 = r4r2_vals(el1)
        Q_A = np.sqrt(np.sqrt(el1) * el1_r4r2)
        for j in range(i):
            if i == j:
                continue
            for k in range(lattice_points):
                el2 = int(pos[j])
                el2_r4r2 = r4r2_vals(el2)
                Q_B = np.sqrt(np.sqrt(el2) * el2_r4r2)
                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                # TODO: read correct C6s
                c6_i, c6_j = M[i], M[j]

                C6 = C6s[c6_i, c6_j]
                C6_mon[i, j] = C6
                r1, r2 = cs[i, :], cs[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))
                # R value is not a match since dftd4 converts input carts
                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)
                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6 * edisp * 0.5
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy, C6_mon


def compute_bj_pairs(
    params: [],
    pos: np.array,
    carts: np.array,
    Ma: int,  # number of atoms in monomer A
    Mb: int,  # number of atoms in monomer B
    C6s: np.array,
    index: int = 1,
    mult_out=constants.conversion_factor("hartree", "kcal / mol"),
) -> float:
    """
    compute_bj_pairs computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    s8, a1, a2 = params
    s6 = 1.0
    C8s = np.zeros(np.shape(C6s))

    aatoau = Constants().g_aatoau()
    energy = 0
    cs = aatoau * np.array(carts, copy=True)
    # cs = np.array(carts)
    for i in Ma:
        el1 = int(pos[i])
        el1_r4r2 = r4r2_vals(el1)
        Q_A = np.sqrt(el1) * el1_r4r2
        for j in Mb:
            el2 = int(pos[j])
            el2_r4r2 = r4r2_vals(el2)
            Q_B = np.sqrt(el2) * el2_r4r2
            C8s[i, j] = 3 * C6s[i, j] * np.sqrt(Q_A * Q_B)
            C6 = C6s[i, j]
            C8 = C8s[i, j]

            r1, r2 = cs[i, :], cs[j, :]
            R = np.linalg.norm(r1 - r2)
            R0 = np.sqrt(C8 / C6)

            energy += C6 / (R**6.0 + (a1 * R0 + a2) ** 6.0)
            energy += s8 * C8 / (R**8.0 + (a1 * R0 + a2) ** 8.0)

    energy *= -mult_out
    if index == 1466:
        print(index, energy)
    return energy


def compute_bj_alt(
    params: [],
    pos: np.array,
    carts: np.array,
    Ma: int,  # number of atoms in monomer A
    Mb: int,  # number of atoms in monomer B
    C6s: np.array,
) -> float:
    """
    compute_bj_alt computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    s8, a1, a2 = params
    s6 = 1.0
    C8s = np.zeros(np.shape(C6s))
    energies = np.zeros(np.shape(C6s))
    N_tot = len(carts)

    aatoau = Constants().g_aatoau()
    energy = 0
    cs = aatoau * np.array(carts, copy=True)
    for i in range(N_tot):
        el1 = int(pos[i])
        el1_r4r2 = r4r2_vals(el1)
        Q_A = np.sqrt(el1) * el1_r4r2

        for j in range(i):

            el2 = int(pos[j])
            el2_r4r2 = r4r2_vals(el2)
            Q_B = np.sqrt(el2) * el2_r4r2
            C8s[i, j] = 3 * C6s[i, j] * np.sqrt(Q_A * Q_B)
            C6 = C6s[i, j]
            C8 = C8s[i, j]

            r1, r2 = cs[i, :], cs[j, :]
            R = np.linalg.norm(r1 - r2)
            R0 = np.sqrt(C8 / C6)

            energy += C6 / (R**6.0 + (a1 * R0 + a2) ** 6.0)
            energy += s8 * C8 / (R**8.0 + (a1 * R0 + a2) ** 8.0)
            energies[i, j] += C6 / (R**6.0 + (a1 * R0 + a2) ** 6.0)
            energies[i, j] += s8 * C8 / (R**8.0 + (a1 * R0 + a2) ** 8.0)

    # print(energies)
    energy = np.sum(energies)
    # print(energy)
    energy *= -1
    return energy


def compute_C8s(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
) -> float:
    C8s = np.zeros(np.shape(C6s))
    N_tot = len(carts)
    aatoau = Constants().g_aatoau()
    cs = aatoau * np.array(carts, copy=True)
    for i in range(N_tot):
        el1 = int(pos[i])
        el1_r4r2 = r4r2_vals(el1)
        Q_A = np.sqrt(el1) * el1_r4r2
        # for j in range(i):
        for j in range(N_tot):
            el2 = int(pos[j])
            el2_r4r2 = r4r2_vals(el2)
            Q_B = np.sqrt(el2) * el2_r4r2
            C8s[i, j] = 3 * C6s[i, j] * np.sqrt(Q_A * Q_B)
            C6 = C6s[i, j]
            C8 = C8s[i, j]
    return C8s


# /theoryfs2/ds/amwalla3/projects/dftd4/src/dftd4/damping/rational.f90
def compute_bj_f90(
    params: [],
    pos: np.array,
    carts: np.array,
    C6s: np.array,
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    energy = 0
    s8, a1, a2 = params
    s6 = 1.0
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1
    aatoau = Constants().g_aatoau()
    cs = aatoau * np.array(carts, copy=True)
    for i in range(M_tot):
        el1 = int(pos[i])
        el1_r4r2 = r4r2_vals(el1)
        Q_A = np.sqrt(np.sqrt(el1) * el1_r4r2)

        for j in range(i):
            if i == j:
                continue
            for k in range(lattice_points):
                el2 = int(pos[j])
                el2_r4r2 = r4r2_vals(el2)
                Q_B = np.sqrt(np.sqrt(el2) * el2_r4r2)

                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6 = C6s[i, j]

                r1, r2 = cs[i, :], cs[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))
                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)
                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6 * edisp * 0.5
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy


def compute_bj_f90_exact(
    params: [],
    pos: np.array,
    carts: np.array,
    C6s: np.array,
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    r4r2 = r4r2_ls()
    energy = 0
    s8, a1, a2 = params
    s6 = 1.0
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1
    aatoau = Constants().g_aatoau()
    cs = aatoau * np.array(carts, copy=True)
    cutoff2 = 60
    for i in range(M_tot):
        # el1 = int(pos[i])
        # el1_r4r2 = r4r2_vals(el1)
        # Q_A = np.sqrt(np.sqrt(el1) * el1_r4r2)

        for j in range(i):
            rrij = 3 * r4r2[int(pos[i]) - 1] * r4r2[int(pos[j]) - 1]
            r0ij = a1 * np.sqrt(rrij) + a2
            C6 = C6s[i, j]
            r1, r2 = cs[i, :], cs[j, :]
            r2 = np.subtract(r1, r2)
            r2 = np.sum(np.multiply(r2, r2))
            if r2 > cutoff2 or r2 < 2.2204460492503131e-016:
                continue
            for k in range(lattice_points):
                # el2 = int(pos[j])
                # el2_r4r2 = r4r2_vals(el2)
                # Q_B = np.sqrt(np.sqrt(el2) * el2_r4r2)
                # rrij = 3 * Q_A * Q_B

                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)
                edisp = s6 * t6 + s8 * rrij * t8
                de = -C6 * edisp * 0.5
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy


def compute_bj_dftd4(
    params,
    atom_numbers: np.array,
    carts: np.array,
):
    s8, a1, a2 = params
    s6 = 1.0
    energy = 0.0
    # disp = DispersionModel(
    #     numbers=atom_numbers,
    #     positions=carts,
    # )
    # param = DampingParam(s6=s6, s8=s8, a1=a1, a2=a2)
    # res = disp.get_dispersion(param, grad=False)
    # energy = res.get("energy")

    write_xyz_from_np(atom_numbers, carts)
    subprocess.call(
        "dftd4 dat.xyz --verbose --pair-resolved --verbose --property --param %.16f %.16f %.16f %.16f > dat.txt"
        % (s6, s8, a1, a2),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    with open("dat.txt", "r") as f:
        data = f.readlines()
    for i in data:
        if "2b" in i:
            e_2b = float(i.split()[-1])
    with open(".EDISP") as f:
        # energy = json.load(f)["energy"]
        energy = float(f.read())
    # C6s_json = np.array(C6s_json).reshape((len(C6s), len(C6s)))
    # converts to kcal/mol
    return energy, e_2b


def build_dummy() -> None:
    """
    build_dummy builds and runs all elements to get r4r2

    """
    for i in elements:
        e = i.number
        atom_numbers = np.array([e, e])
        carts = np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
            ],
        )
        write_xyz_from_np(atom_numbers, carts, outfile="diatomic.xyz")
        subprocess.call(
            "dftd4 diatomic.xyz > diatomic.txt",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        with open("diatomic.txt", "r") as f:
            d = f.readlines()
            print(d[15:16])


def split_dimer(geom, Ma, Mb) -> (np.array, np.array):
    """
    split_dimer
    """
    ma = []
    mb = []
    for i in Ma:
        ma.append(geom[i, :])
    for i in Mb:
        mb.append(geom[i, :])
    return np.array(ma), np.array(mb)


def compute_psi4_d4(geom, Ma, Mb, memory: str = "4 GB", basis="jun-cc-pvdz"):
    ma, mb = split_dimer(geom, Ma, Mb)
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
    geom = "0 1\n%s--\n0 1\n%s" % (ma, mb)
    # geom = '%s--\n%s' % (A, B)
    print(geom)
    psi4.geometry(geom)
    psi4.set_memory(memory)
    psi4.set_options(
        {
            "basis": basis,
            "freeze_core": "true",
            "guess": "sad",
            "scf_type": "df",
        }
    )
    v = psi4.energy("hf-d4", bsse_type="cp")
    print(v)

    return


def compute_bj_from_dimer_AB_with_C6s(
    params,
    pos,
    carts,
    Ma,
    Mb,
    C6s,
    mult_out=constants.conversion_factor("hartree", "kcal / mol"),
) -> float:
    """
    compute_bj_from_dimer_AB_with_C6s computes dftd4 for dimer and each monomer and returns
    subtraction after running dftd4 on monomers
    """
    C6_n, n, e = calc_dftd4_props_params(pos, carts, p=params)
    f90 = compute_bj_f90(params, pos, carts, C6s)
    print(f90, e, f90 - e)

    mon_carts = np.zeros((len(Ma), 3))
    mon_pos = np.zeros(len(Ma))
    for n, i in enumerate(Ma):
        mon_carts[n] = carts[i]
        mon_pos[n] = pos[i]

    C6_a, n, e = calc_dftd4_props_params(mon_pos, mon_carts, p=params)
    monA = compute_bj_f90(params, mon_pos, mon_carts, C6_a)
    print(monA, e, monA - e)

    mon_carts = np.zeros((len(Mb), 3))
    mon_pos = np.zeros(len(Mb))
    for n, i in enumerate(Mb):
        mon_carts[n] = carts[i]
        mon_pos[n] = pos[i]

    C6_b, n, e = calc_dftd4_props_params(mon_pos, mon_carts, p=params)
    monB = compute_bj_f90(params, mon_pos, mon_carts, C6_b)

    v, C6a = compute_bj_mons(params, pos, carts, Ma, C6s)
    # v, C6b = compute_bj_mons(params, pos, carts, Mb, C6s)
    # print("monA")
    # print(C6_a[-1])
    # print(C6a[-1])
    # TODO: just tally C6s for monomers as well

    # print(np.subtract(C6_a[0], C6a[0]))

    # print(np.subtract(C6_b, C6b))

    print(monB, e, monB - e)
    AB = monA + monB
    disp = f90 - (AB)
    return disp * mult_out
    # return f90* mult_out


def compute_bj_from_dimer_AB_all_C6s(
    params,
    pos,
    carts,
    Ma,
    Mb,
    C6s,
    C6_A,
    C6_B,
    mult_out=constants.conversion_factor("hartree", "kcal / mol"),
) -> float:
    """
    compute_bj_from_dimer_AB computes dftd4 for dimer and each monomer and returns
    subtraction.
    """
    f90 = compute_bj_f90(params, pos, carts, C6s)
    # print_cartesians_pos_carts(pos, carts)

    mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
    monA = compute_bj_f90(params, mon_pa, mon_ca, C6_A)

    mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
    monB = compute_bj_f90(params, mon_pb, mon_cb, C6_B)

    AB = monA + monB
    disp = f90 - (AB)
    return disp * mult_out


def compute_bj_from_dimer_AB(
    params,
    pos,
    carts,
    Ma,
    Mb,
    C6s,
    mult_out=627.509,
) -> float:
    """
    compute_bj_from_dimer_AB computes dftd4 for dimer and each monomer and returns
    subtraction.
    """
    f90 = compute_bj_f90(params, pos, carts, C6s)
    monA, C6 = compute_bj_mons(params, pos, carts, Ma, C6s)
    monB, C6 = compute_bj_mons(params, pos, carts, Mb, C6s)
    AB = monA + monB
    disp = f90 - (AB)
    return disp * mult_out
    # return f90* mult_out


# self :     -26.86350371919017
# DFTD4 2b : -26.86350371919017


def gather_data2_testing_mol(mol):
    params = [1.61679827, 0.44959224, 3.35743605]
    print(params)
    g3 = mol["Geometry"]
    Ma, Mb = mol["monAs"], mol["monBs"]
    pos = g3[:, 0]
    carts = g3[:, 1:]
    HF_jdz = mol["HF_jdz"]
    C6s, C8s = calc_dftd4_props(pos, carts)
    pairs = compute_bj_pairs(params, pos, carts, Ma, Mb, C6s)
    alt = compute_bj_alt(params, pos, carts, Ma, Mb, C6s)
    f90 = compute_bj_f90(params, pos, carts, C6s)
    monA, C6 = compute_bj_mons(params, pos, carts, Ma, C6s)
    monB, C6 = compute_bj_mons(params, pos, carts, Mb, C6s)
    d_ab = compute_bj_from_dimer_AB(params, pos, carts, Ma, Mb, C6s)
    AB = monA + monB
    conv = constants.conversion_factor("hartree", "kcal / mol")

    energy, e_2b = compute_bj_dftd4(params, pos, carts)
    print("DFTD4 full:", energy)
    print("pairs     :", pairs)
    print("f90       :", f90)
    print("DFTD4 2b  :", e_2b)
    print("alt       :", alt)
    print("f90  - DFTD4 2b = ", f90 - e_2b)
    print("alt  - DFTD4 2b = ", alt - e_2b)
    print("monA :", monA)
    print("monB :", monB)
    print("AB:", AB)
    print("f90 - AB =", f90 - AB)
    print("HF_jdz = ", HF_jdz)
    print("BM     = ", mol["Benchmark"])
    d4_1 = f90 - (AB)
    ie1 = HF_jdz + d4_1 * conv
    d4_2 = alt - (AB)
    ie2 = HF_jdz + d4_2 * conv
    ie3 = HF_jdz + d_ab
    d4 = pairs
    ie4 = HF_jdz + d4 * conv
    print("DISP (f90 )  =", d4_1 * conv)
    print("DISP (alt )  =", d4_2 * conv)
    print("DISP (d_ab ) =", d_ab)
    print("DISP (pairs) =", pairs * conv)
    print("IE (f90 )    =", ie1)
    print("IE (alt )    =", ie2)
    print("IE (d_ab )   =", ie3)
    print("IE (pairs)   =", ie4)
    print(Ma, Mb)
    print_cartesians(g3)
    return


def gather_data2_testing(
    condensed_path="condensed.pkl",
):
    params = [1.0000, 0.9, 0.4, 5.0]
    el_dc = create_pt_dict()
    df = pd.read_pickle(condensed_path)
    xyzs = df[["xyz1", "xyz2", "xyz_d"]].to_numpy()
    for sys in xyzs[:1]:
        g1, g2, g3 = sys[0], sys[1], sys[2]
        g1 = read_xyz(g1, el_dc)
        g2 = read_xyz(g2, el_dc)
        gt = np.vstack((g1, g2))
        g3 = read_xyz(g3, el_dc)
        if not np.all((gt == g3) == True):
            print("not a match")
            continue
        pos = g3[:, 0]
        carts = g3[:, 1:]
        C6s, C8s = calc_dftd4_props(pos, carts)
        Ma, Mb = len(g1), len(g2)
        pairs = compute_bj_alt(params, pos, carts, Ma, Mb, C6s)
        print("pairs     :", pairs)
        energy = compute_bj_f90(params, pos, carts, C6s)
        print("self :", energy)
        energy, e_2b = compute_bj_dftd4(params, pos, carts)
        print("DFTD4 full:", energy)
        print("DFTD4 2b  :", e_2b)
        print("pairs - DFTD4 2b = ", pairs - e_2b)
        print()
    return


"""
self : -0.03421315617528683
DFTD4 full: -0.0338246418555
self :      -0.03421315617528684
DFTD4 2b  : -0.03421315617528684
"""


def gather_data2(condensed_path="condensed.pkl", output_path="opt2.pkl"):
    """
    use gather_data3 for best output data from master-regen for other functions
    """
    params = [0.9, 0.5, 5.0]
    el_dc = create_pt_dict()
    df = pd.read_pickle(condensed_path)
    xyzs = df["xyz_d"].to_numpy()
    C6s = [i for i in range(len(xyzs))]
    C8s = [i for i in range(len(xyzs))]

    for n, g3 in enumerate(tqdm(xyzs[:1], desc="DFTD4 Props", ascii=True)):
        g3 = read_xyz(g3, el_dc)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        C6, C8 = calc_dftd4_props(pos, carts)
        C6s[n] = C6
        C8s[n] = C8
    df["C6s"] = C6s
    df["C8s"] = C6s
    df.to_pickle(output_path)
    return


class FailedToSplit(Exception):
    """DID NOT BREAK INTO DIMER"""

    def __init__(self, position):
        self.position
        super().__init__(self.message)


def split_Hs_carts(
    geom: np.array,
    verbose: bool = False,
) -> (np.array, np.array,):
    """
    removes Hs from carts
    """
    g = geom.tolist()
    Hs = []
    Core = []
    for i, r1 in enumerate(g):
        if int(r1[0]) == 1:
            Hs.append(np.array(r1))
        else:
            Core.append(r1)
    g2 = np.array(Core)
    frags = BFS(g2[:, 1:], g2[:, 0], bond_threshold=0.40)
    f1, f2 = [], []
    for n, i in enumerate(frags):
        for j in i:
            if n == 0:
                f1.append(g2[j, :])
            else:
                f2.append(g2[j, :])
    for i, r1 in enumerate(Hs):
        l1, l2 = 100, 100
        for j, r2 in enumerate(f1):
            if int(r2[0]) != 1:
                R = np.linalg.norm(r1[1:] - r2[1:])
                if R < l1:
                    l1 = R
        for j, r2 in enumerate(f2):
            if int(r2[0]) != 1:
                R = np.linalg.norm(r1[1:] - r2[1:])
                if R < l2:
                    l2 = R
        if l1 < l2:
            f1.append(r1)
        else:
            f2.append(r1)
    monAs, monBs = [], []
    f1 = np.array(f1)
    f2 = np.array(f2)
    for n, i in enumerate(f1):
        monAs.append(n)
    for n, i in enumerate(f2):
        monBs.append(n + len(monAs))
    ft = np.vstack((f1, f2))
    if verbose:
        print("monA", monAs)
        print_cartesians(f1)
        print("monB", monBs)
        print_cartesians(f2)
        print("\ncombined")
        print_cartesians(ft)
        print(len(ft) == len(geom))
        print(len(monAs) > 0 and len(monBs) > 0)
    return ft, np.array(monAs), np.array(monBs)


def gather_data3_dimer_splits(
    df_og: pd.DataFrame,
) -> pd.DataFrame:
    """
    gather_data3_dimer_splits creates dimers from failed splits in gather_data3
    from BFS
    """
    df = df_og[["Geometry", "monAs", "monBs"]].copy()
    ind1 = df["monBs"].index[df["monBs"].isna()]
    print(f"indexes: {ind1}")
    for i in ind1:
        g3 = df.loc[i, "Geometry"]
        g3 = np.array(g3)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        frags = BFS(carts, pos, bond_threshold=0.35)
        if len(frags) == 2:
            f1 = frags[0]
            f2 = frags[1]
            df.loc[i, "monAs"] = f1
            df.loc[i, "monBs"] = f2

    df1 = df.copy()
    ind2 = df["monBs"].index[df["monBs"].isna()]
    for i in ind2:
        g3 = df.loc[i, "Geometry"]
        geom, monA, monB = split_Hs_carts(g3)
        df.loc[i, "Geometry"] = geom
        df.loc[i, "monAs"] = monA
        df.loc[i, "monBs"] = monB
    geoms = df["Geometry"].to_list()
    monAs = df["monAs"].to_list()
    monBs = df["monBs"].to_list()
    del df_og["Geometry"]
    del df_og["monAs"]
    del df_og["monBs"]
    df_og["Geometry"] = geoms
    df_og["monAs"] = monAs
    df_og["monBs"] = monBs
    assert df["monAs"].isna().sum() == 0, "Not all dimers split!"
    assert df["monBs"].isna().sum() == 0, "Not all dimers split!"
    return df_og, ind1


def expand_opt_df(
    df,
    columns_to_add: list = ["HF_dz", "HF_dt", "HF_adz", "HF_adt", "HF_jdz_dftd4"],
    prefix: str = "",
    replace_HF: bool = False,
) -> pd.DataFrame:
    """
    expand_opt_df adds columns with nan
    """
    if replace_HF:
        df = replace_hf_int_HF_jdz(df)
    columns_to_add = ["%s%s" % (prefix, i) for i in columns_to_add]
    for i in columns_to_add:
        if i not in df:
            df[i] = np.nan
    # print(df.columns.values)
    return df


def gather_data3(
    master_path="master-regen.pkl",
    output_path="opt3.pkl",
    verbose=False,
    HF_columns=[
        "HF_dz",
        "HF_adz",
        "HF_atz",
        "HF_tz",
        "HF_jtz",
    ],
    from_master: bool = True,
):
    """
    **NOTE** This damages ordering of geometries for C6s causing incorrect results.
    collects data from master-regen.pkl from jeffschriber's scripts for D3
    (https://aip.scitation.org/doi/full/10.1063/5.0049745)
    """
    if from_master:
        df = pd.read_pickle(master_path)
        df["SAPT0"] = df["SAPT0 TOTAL ENERGY"]
        df["SAPT"] = df["SAPT TOTAL ENERGY"]
        df = df[
            [
                "DB",
                "System",
                "System #",
                "Benchmark",
                "HF INTERACTION ENERGY",
                "Geometry",
                "SAPT0",
                "SAPT",
            ]
        ]
        df = replace_hf_int_HF_jdz(df)
        xyzs = df["Geometry"].to_list()
        C6s = [np.array([]) for i in range(len(xyzs))]
        C8s = [np.array([]) for i in range(len(xyzs))]
        monAs = [np.nan for i in range(len(xyzs))]
        monBs = [np.nan for i in range(len(xyzs))]

        ones, twos, clear = [], [], []
        for n, c in enumerate(tqdm(xyzs[:], desc="DFTD4 Props", ascii=True)):
            g3 = np.array(c)
            pos = g3[:, 0]
            carts = g3[:, 1:]
            C6, C8 = calc_dftd4_props(pos, carts)
            C6s[n] = C6
            C8s[n] = C8
            frags = BFS(carts, pos, bond_threshold=0.4)
            if len(frags) > 2:
                ones.append(n)
            elif len(frags) == 1:
                twos.append(n)
                # monAs[n] = np.array(frags[0])
            else:
                monAs[n] = np.array(frags[0])
                monBs[n] = np.array(frags[1])
                clear.append(n)

        print("total =", len(xyzs))
        print("ones =", len(ones))
        print("twos =", len(twos))
        print("clear =", len(clear))
        df["C6s"] = C6s
        df["C8s"] = C8s
        df["monAs"] = monAs
        df["monBs"] = monBs
        df = df.reset_index(drop=True)
        df.to_pickle(output_path)
        df, inds = gather_data3_dimer_splits(df)
        df = expand_opt_df(df, HF_columns)
        df = ssi_bfdb_data(df)
    else:
        df = pd.read_pickle(output_path)
        df = expand_opt_df(df, HF_columns)
    for i in HF_columns:
        df = harvest_data(df, i.split("_")[-1])
    df.to_pickle(output_path)
    return df


def reorganize_carts_to_split_middle(geoms: [], mAs: [], mBs: []) -> []:
    """
    reorganize_carts_to_split_middle
    """
    for n, g in enumerate(geoms):
        print(f"Mol {n}\n")
        ma, mb = mAs[n], mBs[n]
        print(ma, mb)


def gather_data4(
    master_path="master-regen.pkl",
    output_path="opt4.pkl",
    verbose=False,
    HF_columns=[
        "HF_dz",
        "HF_adz",
        "HF_atz",
        "HF_tz",
        "HF_jtz",
    ],
    from_master: bool = True,
):
    """
    collects data from master-regen.pkl from jeffschriber's scripts for D3
    (https://aip.scitation.org/doi/full/10.1063/5.0049745)
    """
    if from_master:
        df = pd.read_pickle(master_path)
        df["SAPT0"] = df["SAPT0 TOTAL ENERGY"]
        df["SAPT"] = df["SAPT TOTAL ENERGY"]
        df = df[
            [
                "DB",
                "System",
                "System #",
                "Benchmark",
                "HF INTERACTION ENERGY",
                "Geometry",
                "SAPT0",
                "SAPT",
            ]
        ]
        df = replace_hf_int_HF_jdz(df)
        xyzs = df["Geometry"].to_list()
        monAs = [np.nan for i in range(len(xyzs))]
        monBs = [np.nan for i in range(len(xyzs))]

        ones, twos, clear = [], [], []
        for n, c in enumerate(tqdm(xyzs[:], desc="Dimer Splits", ascii=True)):
            g3 = np.array(c)
            pos = g3[:, 0]
            carts = g3[:, 1:]
            frags = BFS(carts, pos, bond_threshold=0.4)
            if len(frags) > 2:
                ones.append(n)
            elif len(frags) == 1:
                twos.append(n)
                # monAs[n] = np.array(frags[0])
            else:
                monAs[n] = np.array(frags[0])
                monBs[n] = np.array(frags[1])
                clear.append(n)
        print("total =", len(xyzs))
        print("ones =", len(ones))
        print("twos =", len(twos))
        print("clear =", len(clear))
        df["monAs"] = monAs
        df["monBs"] = monBs
        df = df.reset_index(drop=True)
        df, inds = gather_data3_dimer_splits(df)
        df = df.reset_index(drop=True)
        # TODO: possibly add check to ensure geoms are splitted in carts before C6s

        xyzs = df["Geometry"].to_list()
        C6s = [np.array([]) for i in range(len(xyzs))]
        C8s = [np.array([]) for i in range(len(xyzs))]
        for n, c in enumerate(tqdm(xyzs[:], desc="DFTD4 Props", ascii=True)):
            g3 = np.array(c)
            pos = g3[:, 0]
            carts = g3[:, 1:]
            C6, C8 = calc_dftd4_props(pos, carts)
            C6s[n] = C6
            C8s[n] = C8
        df["C6s"] = C6s
        df["C8s"] = C8s
        df.to_pickle(output_path)
        df = expand_opt_df(df, HF_columns)
        df = ssi_bfdb_data(df)
    else:
        df = pd.read_pickle(output_path)
        df = expand_opt_df(df, HF_columns)
    for i in HF_columns:
        df = harvest_data(df, i.split("_")[-1])
    df.to_pickle(output_path)
    return df


def replace_hf_int_HF_jdz(df):
    df["HF_jdz"] = df["HF INTERACTION ENERGY"]
    df = df.drop(columns="HF INTERACTION ENERGY")
    return df


def assign_charges(df, path_SSI="data/SSI_xyzfiles/combined/") -> pd.DataFrame:
    c = np.array([[0, 1], [0, 1], [0, 1]])
    charges = [c for i in range(len(df))]
    ind1 = df["DB"].index[df["DB"] == "SSI"]
    cnt = 0
    for i in ind1:
        sys = df.loc[i, "System"]
        sys = f"{path_SSI}SSI-{sys[8:14]}-{sys[19:25]}-{sys[-1]}"
        f_d = f"{sys}-dimer.xyz"
        f_A = f"{sys}-monoA-unCP.xyz"
        f_B = f"{sys}-monoB-unCP.xyz"
        with open(f_d, "r") as f:
            c_d = f.readlines()[1].rstrip()
            c_d = [int(i) for i in c_d.split()]
        with open(f_A, "r") as f:
            c_A = f.readlines()[1].rstrip()
            c_A = [int(i) for i in c_A.split()]
        with open(f_B, "r") as f:
            c_B = f.readlines()[1].rstrip()
            c_B = [int(i) for i in c_B.split()]
        charge = np.array([c_d, c_A, c_B])
        charges[i] = charge
    df["charges"] = charges
    return df


def split_mons(xyzs) -> []:
    """
    split_mons takes xyzs and splits
    """
    monAs = [np.nan for i in range(len(xyzs))]
    monBs = [np.nan for i in range(len(xyzs))]
    ones, twos, clear = [], [], []
    for n, c in enumerate(
        tqdm(
            xyzs[:],
            desc="Dimer Splits",
            ascii=True,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
    ):
        g3 = np.array(c)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        frags = BFS(carts, pos, bond_threshold=0.4)
        if len(frags) > 2:
            ones.append(n)
        elif len(frags) == 1:
            twos.append(n)
            # monAs[n] = np.array(frags[0])
        else:
            monAs[n] = np.array(frags[0])
            monBs[n] = np.array(frags[1])
            clear.append(n)
    print("total =", len(xyzs))
    print("ones =", len(ones))
    print("twos =", len(twos))
    print("clear =", len(clear))
    return monAs, monBs


def calc_c6s_for_df(xyzs, monAs, monBs, charges) -> ([], [], []):
    """
    calc_c6s_for_df
    """
    C6s = [np.array([]) for i in range(len(xyzs))]
    C6_A = [np.array([]) for i in range(len(xyzs))]
    C6_B = [np.array([]) for i in range(len(xyzs))]
    pas = [np.array([]) for i in range(len(xyzs))]
    pbs = [np.array([]) for i in range(len(xyzs))]
    pds = [np.array([]) for i in range(len(xyzs))]
    for n, c in enumerate(
        tqdm(
            xyzs[:],
            desc="DFTD4 Props",
            ascii=True,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
    ):
        g3 = np.array(c)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        c = charges[n]
        # C6, na = calc_dftd4_props(pos, carts)
        C6, pd = calc_dftd4_props_psi4_dftd4(pos, carts, c[0])
        C6s[n] = C6
        pds[n] = pd

        Ma = monAs[n]
        mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
        # C6a, na = calc_dftd4_props(mon_pa, mon_ca)
        C6a, pa = calc_dftd4_props_psi4_dftd4(mon_pa, mon_ca, c[1])
        C6_A[n] = C6a
        pas[n] = pa

        Mb = monBs[n]
        mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
        # C6b, na = calc_dftd4_props(mon_pb, mon_cb)
        C6b, pb = calc_dftd4_props_psi4_dftd4(mon_pb, mon_cb, c[2])
        C6_B[n] = C6b
        pbs[n] = pb
    return C6s, C6_A, C6_B, pds, pas, pbs


def calc_c6_c8_pairs_for_df(xyzs, monAs, monBs, charges) -> ([], [], []):
    """
    calc_c6_c8_pairs_for_df
    """
    C6s = [np.array([]) for i in range(len(xyzs))]
    C6_A = [np.array([]) for i in range(len(xyzs))]
    C6_B = [np.array([]) for i in range(len(xyzs))]
    pas = [np.array([]) for i in range(len(xyzs))]
    pbs = [np.array([]) for i in range(len(xyzs))]
    pds = [np.array([]) for i in range(len(xyzs))]
    for n, c in enumerate(
        tqdm(
            xyzs[:],
            desc="DFTD4 Props",
            ascii=True,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
    ):
        g3 = np.array(c)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        c = charges[n]
        # C6, na = calc_dftd4_props(pos, carts)
        C6, pd = calc_c6s_c8s_polariz_pairDisp2_for_df(pos, carts, c[0])
        C6s[n] = C6
        pds[n] = pd

        Ma = monAs[n]
        mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
        # C6a, na = calc_dftd4_props(mon_pa, mon_ca)
        C6a, pa = calc_dftd4_props_psi4_dftd4(mon_pa, mon_ca, c[1])
        C6_A[n] = C6a
        pas[n] = pa

        Mb = monBs[n]
        mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
        # C6b, na = calc_dftd4_props(mon_pb, mon_cb)
        C6b, pb = calc_dftd4_props_psi4_dftd4(mon_pb, mon_cb, c[2])
        C6_B[n] = C6b
        pbs[n] = pb
    return C6s, C6_A, C6_B, pds, pas, pbs


def calc_c6s_c8s_pairDisp2_for_df(xyzs, monAs, monBs, charges) -> ([], [], []):
    """
    calc_c6s_for_df
    """
    C6s = [np.array([]) for i in range(len(xyzs))]
    C6_A = [np.array([]) for i in range(len(xyzs))]
    C6_B = [np.array([]) for i in range(len(xyzs))]
    C8s = [np.array([]) for i in range(len(xyzs))]
    C8_A = [np.array([]) for i in range(len(xyzs))]
    C8_B = [np.array([]) for i in range(len(xyzs))]
    disp_d = [np.array([]) for i in range(len(xyzs))]
    disp_a = [np.array([]) for i in range(len(xyzs))]
    disp_b = [np.array([]) for i in range(len(xyzs))]
    for n, c in enumerate(
        tqdm(
            xyzs[:],
            desc="DFTD4 Props",
            ascii=True,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
    ):
        g3 = np.array(c)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        c = charges[n]
        C6, C8, dispd = calc_dftd4_c6_c8_pairDisp2(pos, carts, c[0])
        C6s[n] = C6
        C8s[n] = C8
        disp_d[n] = dispd

        Ma = monAs[n]
        mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
        C6a, C8a, dispa = calc_dftd4_c6_c8_pairDisp2(mon_pa, mon_ca, c[1])
        C6_A[n] = C6a
        C8_A[n] = C8a
        disp_a[n] = dispa

        Mb = monBs[n]
        mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
        C6b, C8b, dispb = calc_dftd4_c6_c8_pairDisp2(mon_pb, mon_cb, c[2])
        C6_B[n] = C6b
        C8_B[n] = C8b
        disp_b[n] = dispb
    return C6s, C6_A, C6_B, C8s, C8_A, C8_B, disp_d, disp_a, disp_b


def gather_data5(
    master_path="master-regen.pkl",
    output_path="opt5.pkl",
    verbose=False,
    HF_columns=[
        "HF_dz",
        "HF_adz",
        "HF_atz",
        "HF_tz",
    ],
    from_master: bool = True,
    overwrite: bool = False,
    replace_hf: bool = False,
):
    """
    collects data from master-regen.pkl from jeffschriber's scripts for D3
    (https://aip.scitation.org/doi/full/10.1063/5.0049745)
    """
    if from_master:
        df = pd.read_pickle(master_path)
        df["SAPT0"] = df["SAPT0 TOTAL ENERGY"]
        df["SAPT"] = df["SAPT TOTAL ENERGY"]
        df = df[
            [
                "DB",
                "System",
                "System #",
                "Benchmark",
                "HF INTERACTION ENERGY",
                "Geometry",
                "SAPT0",
                "SAPT",
                "Disp20",
                "SAPT DISP ENERGY",
            ]
        ]

        if replace_hf:
            df = replace_hf_int_HF_jdz(df)
        xyzs = df["Geometry"].to_list()
        monAs, monBs = split_mons(xyzs)
        df["monAs"] = monAs
        df["monBs"] = monBs
        df = df.reset_index(drop=True)
        df, inds = gather_data3_dimer_splits(df)
        monAs = df["monAs"].to_list()
        monBs = df["monBs"].to_list()
        # t = [x for x in monAs if type(x) != type(np.array)]
        df = df.reset_index(drop=True)
        xyzs = df["Geometry"].to_list()
        df = assign_charges(df)
        charges = df["charges"]
        C6s, C6_A, C6_B, pds, pas, pbs = calc_c6s_for_df(xyzs, monAs, monBs, charges)
        df["C6s"] = C6s
        df["C6_A"] = C6_A
        df["C6_B"] = C6_B
        df["pds"] = pds
        df["pas"] = pas
        df["pbs"] = pbs
        df.to_pickle(output_path)
        df = expand_opt_df(df, HF_columns)
        df = ssi_bfdb_data(df)
        df.to_pickle(output_path)
    else:
        df = pd.read_pickle(output_path)
        df = expand_opt_df(df, HF_columns)
    for i in HF_columns:
        df = harvest_data(df, i.split("_")[-1], overwrite=overwrite)
    df.to_pickle(output_path)
    return df

def ram_data():
    df = pd.read_pickle('master-regen.pkl')
    df = df[
        [
            "Name",
            "DB",
            "Benchmark",
            "SAPT TOTAL ENERGY",
            "SAPT IND ENERGY",
            "SAPT EXCH ENERGY",
            "SAPT ELST ENERGY",
            "SAPT DISP ENERGY",
            "SAPT0 TOTAL ENERGY",
            "SAPT0 IND ENERGY",
            "SAPT0 EXCH ENERGY",
            "SAPT0 ELST ENERGY",
            "SAPT0 DISP ENERGY",
            "Geometry",
            'R'
            # TODO: add RA and RB
        ]
    ]
    xyzs = df["Geometry"].to_list()
    monAs, monBs = split_mons(xyzs)
    ra, rb = [], []
    for n in range(len(xyzs)):
        a = xyzs[0][monAs[0]]
        b = xyzs[0][monBs[0]]
        ra.append(a)
        rb.append(b)
    df['RA'] = ra
    df['RB'] = rb
    df.to_pickle("rm.pkl")
    print(df)
    return



def gather_data6(
    master_path="master-regen.pkl",
    output_path="opt5.pkl",
    verbose=False,
    HF_columns=[
        "HF_dz",
        "HF_adz",
        "HF_atz",
        "HF_tz",
    ],
    from_master: bool = True,
    overwrite: bool = False,
    replace_hf: bool = False,
):
    """
    collects data from master-regen.pkl from jeffschriber's scripts for D3
    (https://aip.scitation.org/doi/full/10.1063/5.0049745)
    """
    if from_master:
        df = pd.read_pickle(master_path)
        # df = inpsect_master_regen()
        df["SAPT0"] = df["SAPT0 TOTAL ENERGY"]
        df["SAPT"] = df["SAPT TOTAL ENERGY"]
        df = df[
            [
                "DB",
                "System",
                "System #",
                "Benchmark",
                "HF INTERACTION ENERGY",
                "Geometry",
                "SAPT0",
                "SAPT",
                "Disp20",
                "SAPT DISP ENERGY",
                "SAPT DISP20 ENERGY",
                "D3Data",
            ]
        ]
        if replace_hf:
            df = replace_hf_int_HF_jdz(df)
        xyzs = df["Geometry"].to_list()
        monAs, monBs = split_mons(xyzs)
        df["monAs"] = monAs
        df["monBs"] = monBs
        df = df.reset_index(drop=True)
        df, inds = gather_data3_dimer_splits(df)
        monAs = df["monAs"].to_list()
        monBs = df["monBs"].to_list()
        # t = [x for x in monAs if type(x) != type(np.array)]
        df = df.reset_index(drop=True)
        xyzs = df["Geometry"].to_list()
        df = assign_charges(df)
        charges = df["charges"]
        (
            C6s,
            C6_A,
            C6_B,
            C8s,
            C8_A,
            C8_B,
            disp_d,
            disp_a,
            disp_b,
        ) = calc_c6s_c8s_pairDisp2_for_df(xyzs, monAs, monBs, charges)

        df["C6s"] = C6s
        df["C6_A"] = C6_A
        df["C6_B"] = C6_B
        df["C8s"] = C8s
        df["C8_A"] = C8_A
        df["C8_B"] = C8_B
        df['disp_d'] = disp_d
        df['disp_a'] = disp_a
        df['disp_b'] = disp_b

        df.to_pickle(output_path)
        df = expand_opt_df(df, HF_columns)
        df = ssi_bfdb_data(df)
        df.to_pickle(output_path)
    else:
        df = pd.read_pickle(output_path)
        df = expand_opt_df(df, HF_columns)
    for i in HF_columns:
        df = harvest_data(df, i.split("_")[-1], overwrite=overwrite)
    df.to_pickle(output_path)
    return df
