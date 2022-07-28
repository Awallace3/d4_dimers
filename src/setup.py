import numpy as np
import pandas as pd
from periodictable import elements
from .r4r2 import get_Q, r4r2_from_elements_call, r4r2_vals
from .tools import print_cartesians, print_cartesians_pos_carts
import subprocess
import json
import math
from .constants import Constants
from tqdm import tqdm
from psi4.driver.qcdb.bfs import BFS


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
    for i in tqdm(range(len(data[0])), desc="DFTD4 Props", ascii=True):
        C6, Q = calc_dftd4_props(data[1][i], data[2][i])
        C6s.append(C6)
        Qs.append(Q)
    C6s = C6s
    Qs = Qs
    d4_vals = d4_values(C6s, Qs, C8s)
    data_out = mols_data(energies, atom_order, carts, d4_vals)
    write_pickle(data_out, out_pkl)
    return data_out


def write_xyz_from_np(atom_numbers, carts, outfile="dat.xyz") -> None:
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


def calc_dftd4_props(
    atom_numbers: np.array,
    carts: np.array,
    input_xyz: str = "dat.xyz",
    output_json: str = "",
):
    write_xyz_from_np(atom_numbers, carts, outfile=input_xyz)
    if output_json == "":
        args = ["dftd4", input_xyz, "--property"]
        # cmd = "~/.local/bin/dftd4 %s --property" % (input_xyz)
    else:
        args = ["dftd4", input_xyz, "--property", "--json", output_json]
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


def read_xyz(xyz_path: str, el_dc: dict) -> np.array:
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

    energy *= -627.509
    # energy *= -1
    return energy


def compute_bj_pairs(
    params: [],
    pos: np.array,
    carts: np.array,
    Ma: int,  # number of atoms in monomer A
    Mb: int,  # number of atoms in monomer B
    C6s: np.array,
) -> float:
    """
    compute_bj_self computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    s6, s8, a1, a2 = params
    C8s = np.zeros(np.shape(C6s))
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

    energy *= -1
    return energy


def compute_C8s(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
) -> float:
    """
    compute_bj_self computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
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


def compute_bj_self(
    params: [],
    pos: np.array,
    carts: np.array,
    C6s: np.array,
) -> float:
    """
    compute_bj_self computes energy from C6s, cartesian coordinates, and dimer sizes.
    """

    energy = 0
    s6, s8, a1, a2 = params
    C8s = np.zeros(np.shape(C6s))
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
                # C8s[i, j] = 3 * C6s[i, j] * np.sqrt(Q_A * Q_B)
                C6 = C6s[i, j]
                # C8 = C8s[i, j]

                r1, r2 = cs[i, :], cs[j, :]
                # R = np.linalg.norm(r1 - r2)
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))
                # R value is not a match since dftd4 converts input carts
                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)
                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6 * edisp * 0.5
                # print(carts[i, 0], i, j, R, edisp, de)

                # i + 1 to match f90 indexing
                # print("carts1 =", r1)
                # print("carts2 =", r2)

                # print *, iat, jat, r2, edisp, dE

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
    s6, s8, a1, a2 = params
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
        "dftd4 dat.xyz --verbose --pair-resolved --verbose --property --param %.4f %.4f %.4f %.4f > dat.txt"
        % (s6, s8, a1, a2),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    with open(".EDISP") as f:
        # energy = json.load(f)["energy"]
        energy = float(f.read())
    # C6s_json = np.array(C6s_json).reshape((len(C6s), len(C6s)))
    # converts to kcal/mol
    return energy


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
        energy = compute_bj_pairs(params, pos, carts, Ma, Mb, C6s)
        print("self :", energy)
        energy = compute_bj_self(params, pos, carts, C6s)
        print("self :", energy)
        energy = compute_bj_dftd4(params, pos, carts)
        print("DFTD4:", energy)
        print()
    return


def gather_data2(condensed_path="condensed.pkl", output_path="opt.pkl"):
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
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    gather_data3_dimer_splits creates dimers from failed splits in gather_data3
    from BFS
    """
    ind1 = df["monBs"].index[df["monBs"].isna()]
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
    return df, ind1


def gather_data3(
    master_path="master-regen.pkl",
    output_path="opt3.pkl",
    verbose=False,
):
    """
    collects data from master-regen.pkl from jeffschriber's scripts for D3
    (https://aip.scitation.org/doi/full/10.1063/5.0049745)
    """
    el_dc = create_pt_dict()
    df = pd.read_pickle(master_path)
    df = df[
        ["DB", "System", "System #", "Benchmark", "HF INTERACTION ENERGY", "Geometry"]
    ]
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
            monAs[n] = np.array(frags[0])
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
    df, inds = gather_data3_dimer_splits(df)
    df.to_pickle(output_path)
    return
