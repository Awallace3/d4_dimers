import numpy as np
import pandas as pd
from qm_tools_aw import tools
from .constants import Constants
from tqdm import tqdm
from psi4.driver.qcdb.bfs import BFS
from .harvest import ssi_bfdb_data, harvest_data
import qcelemental as qcel
from . import locald4
from periodictable import elements

ang_to_bohr = Constants().g_aatoau()

def create_pt_dict():
    """
    create_pt_dict creates dictionary for string elements to atomic number.
    """
    el_dc = {}
    for el in elements:
        el_dc[el.symbol] = el.number
    return el_dc

def create_mon_geom(
    pos,
    carts,
    M,
) -> (np.array, np.array):
    """
    create_mon_geom creates pos and carts from dimer
    """
    mon_carts = carts[M]
    mon_pos = pos[M]
    return mon_pos, mon_carts


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
        tools.print_cartesians(f1)
        print("monB", monBs)
        tools.print_cartesians(f2)
        print("\ncombined")
        tools.print_cartesians(ft)
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
    return df


def replace_hf_int_HF_jdz(df):
    df["HF_jdz"] = df["HF INTERACTION ENERGY"]
    df = df.drop(columns="HF INTERACTION ENERGY")
    return df


def assign_charge_single(mol, path_SSI="data/SSI_xyzfiles/combined/") -> pd.DataFrame:
    c = np.array([[0, 1], [0, 1], [0, 1]])
    sys = mol["System"]
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
    return charge


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


def calc_c6s_c8s_pairDisp2_for_df(xyzs, monAs, monBs, charges) -> ([], [], []):
    """
    runs pairDisp2 for all xyzs to accumulate C6s
    """
    C6s = [np.array([]) for i in range(len(xyzs))]
    C6_A = [np.array([]) for i in range(len(xyzs))]
    C6_B = [np.array([]) for i in range(len(xyzs))]
    C6_ATMs = [np.array([]) for i in range(len(xyzs))]
    C6_ATM_A = [np.array([]) for i in range(len(xyzs))]
    C6_ATM_B = [np.array([]) for i in range(len(xyzs))]
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
        C6, _, _, dispd, C6_ATM = locald4.calc_dftd4_c6_c8_pairDisp2(
            pos, carts, c[0], C6s_ATM=True
        )
        C6s[n] = C6
        C6_ATMs[n] = C6_ATM
        disp_d[n] = dispd

        Ma = monAs[n]
        mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
        C6a, _, _, dispa, C6_ATMa = locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pa, mon_ca, c[1], C6s_ATM=True
        )
        C6_A[n] = C6a
        C6_ATM_A[n] = C6_ATMa
        disp_a[n] = dispa

        Mb = monBs[n]
        mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
        C6b, _, _, dispb, C6_ATMb = locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pb, mon_cb, c[2], C6s_ATM=True
        )
        C6_B[n] = C6b
        C6_ATM_B[n] = C6_ATMb
        disp_b[n] = dispb
    return C6s, C6_A, C6_B, C6_ATMs, C6_ATM_A, C6_ATM_B, disp_d, disp_a, disp_b


def generate_D4_data(df):
    xyzs = df["Geometry"].to_list()
    monAs = df["monAs"].to_list()
    monBs = df["monBs"].to_list()
    charges = df["charges"].to_list()
    (
        C6s,
        C6_A,
        C6_B,
        C6_ATMs,
        C6_ATM_A,
        C6_ATM_B,
        disp_d,
        disp_a,
        disp_b,
    ) = calc_c6s_c8s_pairDisp2_for_df(xyzs, monAs, monBs, charges)
    df["C6"] = C6s
    df["C6_A"] = C6_A
    df["C6_B"] = C6_B
    df["C6_ATM"] = C6_ATMs
    df["C6_ATM_A"] = C6_ATM_A
    df["C6_ATM_B"] = C6_ATM_B
    df["disp_d"] = disp_d
    df["disp_a"] = disp_a
    df["disp_b"] = disp_b
    return df


def r_z_tq_to_mol(r, tq, mult) -> qcel.models.Molecule:
    """
    r_z_tq_to_mol takes in carts, charges, and total charge
    to create qcel Molecule.
    """
    geom = f"""
{int(tq)} {int(mult)}
"""
    geom = f""""""
    for n, i in enumerate(r):
        s = f"{int(i[0])} {i[1]:.8f} {i[2]:.8f} {i[3]:.8f}\n"
        geom += s
    return geom


def ram_data_2():
    df = pd.read_pickle("rm.pkl")
    df = df[df["DB"] == "SSI"].reset_index(inplace=False)
    # mol = df.iloc[[0, 1]]
    # df = assign_charges(df)
    mol = df.iloc[2]
    print(mol)
    assign_charge_single(mol)
    c = mol["charges"]
    g1 = r_z_tq_to_mol(mol["RA"], c[1][0], c[1][1])
    g2 = r_z_tq_to_mol(mol["RB"], c[2][0], c[2][1])
    geom = g1 + "--\n" + g2
    print("GEOM")
    print(geom)
    dimer = qcel.models.Molecule.from_data(geom)
    return


def ram_data():
    df = pd.read_pickle("master-regen.pkl")
    df = df[
        [
            "Name",
            "DB",
            "Benchmark",
            "SAPT TOTAL ENERGY",
            "System",
            "SAPT IND ENERGY",
            "SAPT EXCH ENERGY",
            "SAPT ELST ENERGY",
            "SAPT DISP ENERGY",
            "SAPT0 TOTAL ENERGY",
            # "SAPT0 IND ENERGY",
            # "SAPT0 EXCH ENERGY",
            # "SAPT0 ELST ENERGY",
            # "SAPT0 DISP ENERGY",
            "Geometry",
            "R"
            # TODO: add RA and RB
        ]
    ]
    xyzs = df["Geometry"].to_list()
    monAs, monBs = split_mons(xyzs)
    df["monAs"] = monAs
    df["monBs"] = monBs
    df = df.reset_index(drop=True)
    df, inds = gather_data3_dimer_splits(df)
    monAs = df["monAs"].to_list()
    monBs = df["monBs"].to_list()
    ra, rb = [], []
    for n in range(len(xyzs)):
        a = xyzs[n][monAs[n]]
        b = xyzs[n][monBs[n]]
        ra.append(a)
        rb.append(b)
    df["RA"] = ra
    df["RB"] = rb
    xyzs = df["Geometry"].to_list()
    # monAs = df["monAs"].to_list()
    # monBs = df["monBs"].to_list()
    # # t = [x for x in monAs if type(x) != type(np.array)]
    # df = df.reset_index(drop=True)
    # xyzs = df["Geometry"].to_list()
    df = assign_charges(df)
    # df = df.drop(["monAs", "monBs"])
    # return
    df.to_pickle("rm.pkl")
    return


def gather_data6(
    master_path="data/master-regen.pkl",
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

        def convert_coords(geom, mult):
            g2 = geom.copy()
            g2[:, 1:] = g2[:, 1:] * mult
            return g2

        id_test = 512
        print("Pre-manipulation")
        tools.print_cartesians(df.iloc[id_test]["Geometry"])
        df["Geometry"] = df.apply(lambda r: r["Geometry"].copy(), axis=1)
        df["Geometry_bohr"] = df.apply(
            lambda r: convert_coords(r["Geometry"], ang_to_bohr),
            axis=1,
        )
        print("A")
        tools.print_cartesians(df.iloc[id_test]["Geometry"])
        print("Bohr")
        tools.print_cartesians(df.iloc[id_test]["Geometry_bohr"])

        if replace_hf:
            df = replace_hf_int_HF_jdz(df)
        xyzs = df["Geometry"].to_list()
        monAs, monBs = split_mons(xyzs)
        df["monAs"] = monAs
        df["monBs"] = monBs
        df = df.reset_index(drop=True)
        df, inds = gather_data3_dimer_splits(df)
        df = df.reset_index(drop=True)
        df = assign_charges(df)
        xyzs = df["Geometry"].to_list()
        monAs = df["monAs"].to_list()
        monBs = df["monBs"].to_list()
        charges = df["charges"]
        (
            C6s,
            C6_A,
            C6_B,
            C6_ATMs,
            C6_ATM_A,
            C6_ATM_B,
            disp_d,
            disp_a,
            disp_b,
        ) = calc_c6s_c8s_pairDisp2_for_df(xyzs, monAs, monBs, charges)

        df["C6s"] = C6s
        df["C6_A"] = C6_A
        df["C6_B"] = C6_B
        df["C6_ATMs"] = C6_ATMs
        df["C6_ATM_A"] = C6_ATM_A
        df["C6_ATM_B"] = C6_ATM_B
        df["disp_d"] = disp_d
        df["disp_a"] = disp_a
        df["disp_b"] = disp_b

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


def extend_df():
    df1 = pd.read_pickle("./dfs/master-regen.pkl")
    df2 = pd.read_pickle("./dfs/schr_dft2.pkl")
    df1.reset_index(drop=True, inplace=True)
    print(df1.columns.values)
    print(df2.columns.values)
    print(len(df1), len(df2))

    include_cols = [
        "Geometry",
        "QcdbSys",
        "R",
        "DB",
        "System",
        "System #",
    ]

    df1_premerge = df1[include_cols].copy()
    def find_geometry_matches(df1, df2):
        matches = []
        for db in df1["DB"].unique():
            print(f"{db = }")
            df1_tmp = df1[df1["DB"] == db].copy()
            df2_tmp = df2[df2["DB"] == db].copy()
            for system in df1_tmp["System"].unique():
                df1_tmp2 = df1_tmp[df1_tmp["System"] == system]
                df2_tmp2 = df2_tmp[df2_tmp["System"] == system]
                if db == "SSI" or db == "Water2510":
                    for system_number in df1_tmp2["System #"].unique():
                        if system_number:
                            df1_tmp3 = df1_tmp2[df1_tmp2["System #"] == system_number]
                            df2_tmp3 = df2_tmp2[df2_tmp2["System #"] == system_number]
                        for i, row1 in df1_tmp3.iterrows():
                            for j, row2 in df2_tmp3.iterrows():
                                g1 = row1["Geometry"][np.lexsort((row1["Geometry"][:, 0], row1["Geometry"][:, 1]))]
                                g2 = row2["Geometry"][np.lexsort((row2["Geometry"][:, 0], row2["Geometry"][:, 1]))]
                                if np.array_equal(g1, g2):
                                    matches.append((i, j))
                else:
                    for i, row1 in df1_tmp2.iterrows():
                        for j, row2 in df2_tmp2.iterrows():
                            g1 = row1["Geometry"][np.lexsort((row1["Geometry"][:, 0], row1["Geometry"][:, 1]))]
                            g2 = row2["Geometry"][np.lexsort((row2["Geometry"][:, 0], row2["Geometry"][:, 1]))]
                            if np.array_equal(g1, g2):
                                matches.append((i, j))
        return matches

    matches = find_geometry_matches(df1_premerge, df2)
    print(len(matches))
    for df1_idx, df2_idx in matches:
        df1_premerge.at[df1_idx, "temp_merge_key"] = df1_idx
        df2.at[df2_idx, "temp_merge_key"] = df1_idx
    for i in range(len(df2)):
        row1 = df1_premerge.iloc[i]
        row2 = df2.iloc[i]
        if np.isnan(row2["temp_merge_key"]):
            print(f"{row1['DB'] = }, {row1['System'] = }")
            print(f"{row2['DB'] = }, {row2['System'] = }")
            g1 = row1["Geometry"][np.lexsort((row1["Geometry"][:, 0], row1["Geometry"][:, 1]))]
            g2 = row2["Geometry"][np.lexsort((row2["Geometry"][:, 0], row2["Geometry"][:, 1]))]
            print(f"{np.subtract(g1, g2) = }")

    df1_premerge = df1_premerge[[ "QcdbSys", "R", 'temp_merge_key' ]]
    df2_merged = pd.merge(df2, df1_premerge, on="temp_merge_key", how="inner")
    df2_merged.drop(columns=["temp_merge_key"], inplace=True)
    # print(df2_merged)
    # print(df2_merged.columns.values)
    df2_merged.to_pickle("./data_files/schr_dft3.pkl")
    return
