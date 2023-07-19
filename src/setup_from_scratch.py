import numpy as np
import pandas as pd
from periodictable import elements
from . import r4r2
from qm_tools_aw import tools
from .constants import Constants
from tqdm import tqdm
from psi4.driver.qcdb.bfs import BFS
from .harvest import ssi_bfdb_data, harvest_data
import psi4
import qcelemental as qcel
from . import locald4

ang_to_bohr = Constants().g_aatoau()
bohr_to_ang = Constants().g_autoaa()
hartree_to_kcal_mol = qcel.constants.conversion_factor("hartree", "kcal / mol")


def inpsect_master_regen():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pkl_path = "master-regen.pkl"
    ms = pd.read_pickle(pkl_path)
    ms = ms[ms["DB"] != "PCONF"]
    ms = ms[ms["DB"] != "SCONF"]
    ms = ms[ms["DB"] != "ACONF"]
    ms = ms[ms["DB"] != "CYCONF"]
    for i in ms.columns.values:
        print(i)
    ms = ms[ms["DB"] == "SSI"]
    for idx, i in ms.iterrows():
        if int(ms.loc[idx, "R"]) != 1:
            print(ms.loc[idx])
    print(ms.columns.values)
    for i in ms.columns.values:
        if "disp" in i.lower():
            print(i, "NaNs =", ms[i].isna().sum())
            ms["t"] = ms.apply(
                lambda r: abs(r["Benchmark"] - (r["HF INTERACTION ENERGY"] + r[i])),
                axis=1,
            )
            print(i, ms["t"].mean(), (ms["t"] ** 2).mean() ** 0.5, ms["t"].max())
    return ms


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


def create_mon_geom(
    pos,
    carts,
    M,
    r4r2_ls=r4r2.r4r2_vals_ls(),
) -> (np.array, np.array):
    """
    create_mon_geom creates pos and carts from dimer
    """
    mon_carts = carts[M]
    mon_pos = pos[M]
    return mon_pos, mon_carts


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
        Q_A = (0.5 * el1**0.5 * r4r2_ls[el1 - 1]) ** 0.5
        # for j in range(i):
        for j in range(N_tot):
            el2 = int(pos[j])
            Q_B = (0.5 * el2**0.5 * r4r2_ls[el2 - 1]) ** 0.5
            C8s[i, j] = 3 * C6s[i, j] * np.sqrt(Q_A * Q_B)
            C6 = C6s[i, j]
            C8 = C8s[i, j]
    return C8s


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
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
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

