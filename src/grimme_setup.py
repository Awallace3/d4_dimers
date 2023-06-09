import pandas as pd
from tqdm import tqdm
from .setup import (
    create_pt_dict,
    calc_dftd4_props,
    create_mon_geom,
    expand_opt_df,
    calc_dftd4_props_params,
    read_xyz,
    create_pt_dict,
    split_mons,
    calc_c6s_for_df,
    harvest_data,
)
from .optimization import compute_int_energy_stats
import numpy as np
import psi4
from qcelemental import constants
from psi4.driver.qcdb.bfs import BFS

"""
From Grimme's first citation...

Within the DFT calculations, we applied standard exchange-correlation
functional integration grids (m4) and typical self-consistent field
convergence criteria (10−7 Eh) as well as the resolution of the
identity integral approximation.44–46 Ahlrich’s type quadruple-zeta
basis sets (def2-QZVP) are used for all single-point calculations. The
density functional specific damping parameters are obtained by
least-squares Levenberg-Marquardt minimization to the reference
interaction energies in the three investigated benchmark sets.

https://aip.scitation.org/doi/10.1063/1.4993215
"""


def gather_BLIND_geoms() -> None:
    """
    gather_BLIND_geoms
    """
    el_dc = create_pt_dict()
    df = pd.read_csv("./data/Grimme/NCIBLIND10/data.csv", delimiter="\t")
    print(df)
    with open("./data/Grimme/NCIBLIND10/geometries.txt", "r") as f:
        data = f.readlines()
    stat = 0
    MAs, MBs, geoms = [], [], []
    MA, MB, geom = [], [], []
    geom_pos = 0
    for l in data:
        if "Monomer A" in l:
            stat = 1
            geom_pos = 0
            continue
        if stat == 1 and "Monomer B" in l:
            stat = 2
            continue
        if stat == 2 and "-----END-------" in l:
            stat = 0
            geom = np.array(geom, dtype="float32")
            MA = np.array(MA)
            MB = np.array(MB)
            geoms.append(geom)
            MAs.append(MA)
            MBs.append(MB)
            MA, MB, geom = [], [], []
            continue
        if stat == 1:
            MA.append(geom_pos)
            l = l.split()
            l[0] = el_dc[l[0]]
            geom.append(l)
        elif stat == 2:
            MB.append(geom_pos)
            l = l.rstrip().split()
            l[0] = int(el_dc[l[0]])
            l = np.array([float(i) for i in l])
            geom.append(l)
        geom_pos += 1
    df["monAs"] = MAs
    df["monBs"] = MBs
    df["Geometry"] = geoms
    df["Benchmark"] = df["CBS"]
    C6s = [np.array([]) for i in range(len(geoms))]
    C6_A = [np.array([]) for i in range(len(geoms))]
    C6_B = [np.array([]) for i in range(len(geoms))]
    for n, c in enumerate(tqdm(geoms[:], desc="DFTD4 Props", ascii=True)):
        g3 = np.array(c)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        C6, na = calc_dftd4_props(pos, carts)
        C6s[n] = C6

        Ma = MAs[n]
        mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
        C6a, na = calc_dftd4_props(mon_pa, mon_ca)
        C6_A[n] = C6a

        Mb = MBs[n]
        mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
        C6b, na = calc_dftd4_props(mon_pb, mon_cb)
        C6_B[n] = C6b

    df["C6s"] = C6s
    df["C6_A"] = C6_A
    df["C6_B"] = C6_B
    bases = ["dz", "tz"]
    df = expand_opt_df(df, bases, prefix="HF_", replace_HF=False)
    df.to_pickle("grimme.pkl")
    return df


def create_hf_energies_grimme(
    df_p: "grimme.pkl",
    bases: [] = ["dz"],
    memory: str = "4gb",
) -> None:
    """
    run_hf_binding_energies uses psi4 to calculate monA, monB, and dimer energies with HF
    with a specified basis set.

    The inputted df will be saved to out_df after each computation finishes.
    """
    df = pd.read_pickle(df_p)
    df = expand_opt_df_jobs(df, bases, prefix="HF_", replace_HF=False)
    pd.to_pickle(df, df_p)

    def_dir = os.getcwd()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    int_dir = os.getcwd()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)

    jobs = []
    for idx, item in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Creating Inputs",
        ascii=True,
    ):
        for basis in bases:
            basis_set, meth_basis_dir = basis_labels(basis)
            method = "hf/%s" % basis_set
            col = "HF_%s" % basis
            v = df.loc[idx, col]
            if not np.isnan(v):
                continue
            p = "%d_%s" % (idx, item["DB"].replace(" - ", "_"))
            job_p = "%s/%s/%s.dat" % (p, meth_basis_dir, in_file)

            if os.path.exists(job_p):
                out_p = "%s/%s/%s.out" % (p, meth_basis_dir, in_file)
                if os.path.exists(out_p):
                    continue
            else:
                if not os.path.exists(p):
                    os.mkdir(p)
                os.chdir(p)
                c = item["Geometry"]
                monA = item["monAs"]
                monB = item["monBs"]
                cm = item["charges"]
                mA, mB = [], []
                for i in monA:
                    mA.append(c[i, :])
                for i in monB:
                    mB.append(c[i, :])
                mA = np_carts_to_string(mA)
                mB = np_carts_to_string(mB)
                write_psi4_sapt0(
                    mA,
                    mB,
                    meth_basis_dir=meth_basis_dir,
                    basis=basis_set,
                    in_file=in_file,
                    charge_mult=cm,
                )
                os.chdir("..")
            jobs.append(job_p)
    os.chdir(int_dir)
    create_pylauncher(
        jobs=jobs,
        data_dir=data_dir,
        basis=basis,
        name=in_file,
        memory=memory,
        ppn=nodes,
        nodes=nodes,
        walltime=walltime,
    )
    os.chdir(def_dir)
    return


def create_Grimme_db(bases=["dz", "tz"]) -> pd.DataFrame:
    """
    create_Grimme_db
    """
    df = pd.read_pickle("s22s66.pkl")
    start = len(df)
    df1 = df[df["DB"] == "S22by7"]
    df1 = df1.reset_index(drop=True)
    df1["m"] = df1.apply(lambda r: "%.4f" % r["Benchmark"], axis=1)
    df2 = pd.read_csv("./data/Databases/S22by7/benchmark_data.csv")
    df2 = df2[["System #", "z", "Benchmark"]]
    df3 = df2.groupby("Benchmark").mean().reset_index()
    df3["m"] = df3.apply(lambda r: "%.4f" % r["Benchmark"], axis=1)
    df_s22 = pd.merge(df1, df3, on=["m"], how="outer")
    # print("s22by7", len(df1))
    # print("s22by7", len(df_s22))

    df1 = df[df["DB"] == "S66by10"]
    df1 = df1.reset_index(drop=True)
    df1["m"] = df1.apply(lambda r: "%.4f" % r["Benchmark"], axis=1)
    df2 = pd.read_csv("./data/Databases/S66by10/benchmark_data.csv")
    df2 = df2[["z", "Benchmark", "System #"]]
    df3 = df2.groupby("Benchmark").mean().reset_index()
    df3["m"] = df3.apply(lambda r: "%.4f" % r["Benchmark"], axis=1)
    df_s66 = pd.merge(df1, df3, on=["m"], how="left")
    df = pd.concat([df_s22, df_s66])
    print("s66by10", len(df1))
    print("s66by10", len(df_s66))
    assert start == len(df)

    # print(df_s22["z"].head(20))
    # print(len(df_s22))
    df_s22 = df_s22[df_s22["z"].isin(np.array([0.9, 1.0, 1.2, 1.5, 2.0]))]
    # print(df_s22["z"].head(20))
    print("s22by5 length:", len(df_s22))

    print(len(df_s66))
    # print(df_s66["z"].head(20))
    # df_s66 = df_s66[
    #     df_s66["z"].isin(np.array([0.9, 0.95, 1.0, 1.05, 1.10, 1.25, 1.5, 2.0]))
    # ]
    df_s66 = df_s66[df_s66["z"] >= 0.9]
    print("s66by8 length:", len(df_s66))
    print(df_s66["z"].head(20))

    df = pd.concat([df_s22, df_s66])
    df["System #"] = df["System #_x"]
    df["Benchmark"] = df["Benchmark_x"]
    del df["System #_x"]
    del df["Benchmark_x"]
    del df["System #_y"]
    del df["Benchmark_y"]
    df = expand_opt_df(df, bases, prefix="HF_", replace_HF=False)
    df.to_pickle("data/Grimme/s22s66_Grimme.pkl")
    return df


def create_grimme_s22s66blind_self() -> None:
    """
    create_grimme_s22s66blind_self
    """
    df1 = pd.read_pickle("./data/Grimme/s22s66_Grimme.pkl")
    # df1 = create_Grimme_db()
    df2 = pd.read_pickle("./data/Grimme/grimme_out.pkl")
    df2["DB"] = ["NCIBLIND10" for i in range(len(df2))]
    df2["charges"] = [np.array([[0, 1], [0, 1], [0, 1]]) for i in range(len(df2))]
    df2["System"] = df2["Dimer"]
    for i in df1.columns.values:
        if i not in df2.columns.values:
            df2[i] = [0 for i in range(len(df2))]
    for i in df2.columns.values:
        if i not in df1.columns.values:
            del df2[i]
    frames = [df1, df2]
    df = pd.concat(frames)
    df = collect_dftd4_atm_for_grimme_parameters(df)
    df.to_pickle("grimme_db.pkl")
    print(len(df), len(df1), len(df2))
    return df


def collect_atm_disp_e(atom_numbers, carts, Mas, Mbs):
    x, z, d = calc_dftd4_props_params(atom_numbers, carts, s9="1.0")
    x, z, ma = calc_dftd4_props_params(atom_numbers[Mas], carts[Mas, :], s9="1.0")
    x, z, mb = calc_dftd4_props_params(atom_numbers[Mbs], carts[Mbs, :], s9="1.0")
    v = d - (ma + mb)
    v *= constants.conversion_factor("hartree", "kcal / mol")
    return v


def collect_dftd4_atm_for_grimme_parameters(df):
    print(df.columns)
    df["dftd4_atm"] = df.apply(
        lambda r: collect_atm_disp_e(
            r["Geometry"][:, 0],
            r["Geometry"][:, 1:],
            r["monAs"],
            r["monBs"],
        ),
        axis=1,
    )
    return df


def create_grimme_s22s66blind() -> None:
    """
    create_grimme_s22s66blind
    """
    global el_dc
    el_dc = create_pt_dict()
    frames = []
    g_ps = [
        "./data/Grimme/d4fitset/NCIBLIND10",
        "./data/Grimme/d4fitset/S66x8",
        "./data/Grimme/d4fitset/S22x5",
    ]
    for i in g_ps:
        df = pd.read_csv(f"{i}/ref.csv")
        db = i.split("/")[-1]
        df["System"] = df["dimer"]
        df["Benchmark"] = df["reference"]
        df["DB"] = db
        del df["dimer"]
        del df["monomerA"]
        del df["monomerB"]
        del df["reference"]
        df["Geometry"] = df.apply(
            lambda r: read_xyz(f'{i}/{r["System"]}/mol.xyz', el_dc), axis=1
        )
        geoms = df["Geometry"].tolist()
        monAs, monBs = split_mons(geoms)
        df["monAs"] = monAs
        df["monBs"] = monBs
        C6s, C6_A, C6_B = calc_c6s_for_df(geoms, monAs, monBs)
        df["C6s"] = C6s
        df["C6_A"] = C6_A
        df["C6_B"] = C6_B
        frames.append(df)
    df = pd.concat(frames)
    df.to_pickle("data/grimme_fitset.pkl")
    # for i in HF_columns:
    #     df = harvest_data(df, i.split("_")[-1], overwrite=overwrite)
    return


def gather_grimme_from_db():
    df = pd.read_pickle("data/grimme_fitset.pkl")
    df["DB"] = "G"
    df = expand_opt_df(df)
    print(df.columns)
    df = harvest_data(df, "jdz_dftd4", data_dir="calcgrimme", overwrite=True)
    df.to_pickle("data/grimme_fitset_db.pkl")
    return df



