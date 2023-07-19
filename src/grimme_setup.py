import pandas as pd
from tqdm import tqdm
from .setup import (
    create_mon_geom,
    expand_opt_df,
    create_pt_dict,
    split_mons,
    harvest_data,
)
from . import locald4
from .optimization import compute_int_energy_stats
import numpy as np
import psi4
import qcelemental as qcel
from qm_tools_aw import tools

hartree_to_kcal_mol = qcel.constants.conversion_factor("hartree", "kcal / mol")

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

def read_xyz(xyz_path: str, el_dc: dict = create_pt_dict()) -> np.array:
    """
    read_xyz takes a path to xyz and returns np.array
    """
    with open(xyz_path, "r") as f:
        dat = "".join(f.readlines()[2:])
    geom = convert_str_carts_np_carts(dat, el_dc)
    return geom



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
        c = charges[n]
        C6, _, dispd, C6_ATM = locald4.calc_dftd4_c6_c8_pairDisp2(
            pos, carts, c[0], C6s_ATM=True
        )
        C6s[n] = C6
        C6_ATMs[n] = C6_ATM
        disp_d[n] = dispd

        Ma = monAs[n]
        mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
        C6a, _, dispa, C6_ATMa = locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pa, mon_ca, c[1], C6s_ATM=True
        )
        C6_A[n] = C6a
        C6_ATM_A[n] = C6_ATMa
        disp_a[n] = dispa

        Mb = monBs[n]
        mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
        C6b, _, dispb, C6_ATMb = locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pb, mon_cb, c[2], C6s_ATM=True
        )
        C6_B[n] = C6b
        C6_ATM_B[n] = C6_ATMb
        disp_b[n] = dispb

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
    x, y, p, d = locald4.calc_dftd4_c6_c8_pairDisp2(atoms, geom, p=params, s9=s9)
    x, y, p, a = locald4.calc_dftd4_c6_c8_pairDisp2(
        atoms[ma], geom[ma, :], p=params, s9=s9
    )
    x, y, p, b = locald4.calc_dftd4_c6_c8_pairDisp2(
        atoms[mb], geom[mb, :], p=params, s9=s9
    )
    v = d - (a + b)
    v *= hartree_to_kcal_mol
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
        df["charges"] = df.apply(lambda r: np.array([[0, 1], [0, 1], [0, 1]]), axis=1)
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
        ) = calc_c6s_c8s_pairDisp2_for_df(geoms, monAs, monBs, charges)
        df["C6s"] = C6s
        df["C6_A"] = C6_A
        df["C6_B"] = C6_B
        df["d4Ds"] = d4Ds
        df["d4As"] = d4As
        df["d4Bs"] = d4Bs
        frames.append(df)
    df = pd.concat(frames)
    df.to_pickle("data/grimme_fitset_test.pkl")
    # for i in HF_columns:
    #     df = harvest_data(df, i.split("_")[-1], overwrite=overwrite)
    return


def gather_grimme_from_db():
    df = pd.read_pickle("data/grimme_fitset.pkl")
    # df["DB"] = "G"
    df = expand_opt_df(df)
    print(df.columns)
    df = harvest_data(df, "jdz_dftd4", data_dir="calcgrimme", overwrite=True)
    # df.to_pickle("data/grimme_fitset_db.pkl")
    return df


def combine_data_with_new_df():
    create_grimme_s22s66blind()
    df = pd.read_pickle("data/grimme_fitset_total.pkl")
    # geoms = df["Geometry"].tolist()
    # monAs, monBs = split_mons(geoms)
    # df["monAs"] = monAs
    # df["monBs"] = monBs
    # df["charges"] = df.apply(lambda r: np.array([[0, 1], [0, 1], [0, 1]]), axis=1)
    # C6s, C6_A, C6_B, d4Ds, d4As, d4Bs = calc_c6s_for_df(
    #     geoms, monAs, monBs, df["charges"].to_list()
    # )
    # df['d4Ds'] = d4Ds
    # df['d4As'] = d4As
    # df['d4Bs'] = d4Bs
    # df.to_pickle("data/grimme_fitset_total.pkl")
    df2 = pd.read_pickle("data/grimme_fitset_test.pkl")
    df["m_g"] = df.apply(
        lambda r: tools.print_cartesians_pos_carts(
            r["Geometry"][:, 0], r["Geometry"][:, 1:], True
        ),
        axis=1,
    )
    df2["m_g"] = df.apply(
        lambda r: tools.print_cartesians_pos_carts(
            r["Geometry"][:, 0], r["Geometry"][:, 1:], True
        ),
        axis=1,
    )
    df2["main_id"] = df2.index
    print(df.columns)
    print(df2.columns)
    # df = pd.merge(df, df2, on=["main_id"], how="inner", suffixes=("", "_y"))
    df = pd.merge(df, df2, on=["m_g"], how="inner", suffixes=("", "_y"))
    print(df)
    df.to_pickle("data/grimme_fitset_test2.pkl")
    return df2


def read_grimme_dftd4_paper_HF_energies(path="dftd4-fitdata/data/hf.csv") -> None:
    """
    read_grimme_dftd4_paper_HF_energies
    """
    df = pd.read_csv(path)
    df2 = pd.read_pickle("data/gf1.pkl")
    print(df.columns.values)
    print(df2.columns.values)
    print(df2[["DB", "System", "HF_qz_no_cp"]].head())
    s_e_dict = {
        "DB": [],
        "System": [],
        "HF_qz_dimer": [],
        "HF_qz_monA": [],
        "HF_qz_monB": [],
    }
    db_mons = {
        "A": {"System": [], "HF_qz_monA": [], "DB": []},
        "B": {"System": [], "HF_qz_monB": [], "DB": []},
    }
    for n, i in df.iterrows():
        db, sys = i["system"].split("/")
        # print(db, sys)
        if "A" not in sys and "B" not in sys:
            s_e_dict["System"].append(sys)
            s_e_dict["HF_qz_dimer"].append(i["HF/def2-QZVP/TM"])
            s_e_dict["DB"].append(db)
            s_e_dict["HF_qz_monA"].append(np.nan)
            s_e_dict["HF_qz_monB"].append(np.nan)

        elif "A" in sys:
            db_mons["A"]["System"].append(sys)
            db_mons["A"]["HF_qz_monA"].append(i["HF/def2-QZVP/TM"])
            db_mons["A"]["DB"].append(db)

        elif "B" in sys:
            db_mons["B"]["System"].append(sys)
            db_mons["B"]["HF_qz_monB"].append(i["HF/def2-QZVP/TM"])
            db_mons["B"]["DB"].append(db)
        else:
            print("ERROR")
    for n, j in enumerate(s_e_dict["System"]):
        for i in range(len(db_mons["A"]["System"])):
            sys_name = db_mons["A"]["System"][i]
            if db_mons["A"]["DB"][i] == s_e_dict["DB"][n]:
                # TODO: fix matching... not working correctly
                s_c = sys_name[:-2]
                db_c = j[:2]
                if s_c == db_c:
                    s_e_dict["HF_qz_monA"][n] = db_mons["A"]["HF_qz_monA"][i]
                    s_e_dict["HF_qz_monB"][n] = db_mons["B"]["HF_qz_monB"][i]
    pd.set_option("display.max_rows", None)
    df_dimers = pd.DataFrame(s_e_dict)
    df_dimers["HF_qz_Grimme"] = (
        df_dimers["HF_qz_dimer"] - df_dimers["HF_qz_monA"] - df_dimers["HF_qz_monB"]
    ) * hartree_to_kcal_mol
    # print(df_dimers)
    df_compare = pd.merge(df_dimers, df2, on=["DB", "System"], how="inner")
    print(df_compare.columns.values)
    df_compare["HF_qz_dif"] = df_compare["HF_qz_Grimme"] - df_compare["HF_qz_no_cp"]
    df_compare["HF_qz_G_d4"] = -(
        df_compare["Benchmark"]
        - (
            df_compare["HF_qz_Grimme"]
            + df_compare["d4Ds"]
            - df_compare["d4As"]
            - df_compare["d4Bs"]
        )
    )
    df_compare["HF_qz_d4"] = -(
        df_compare["Benchmark"]
        - (
            df_compare["HF_qz_no_cp"]
            + df_compare["d4Ds"]
            - df_compare["d4As"]
            - df_compare["d4Bs"]
        )
    )
    # df_compare['HF_qz_no_cp_corrected_D'] = df_compare.apply(lambda r: r['HF_qz_no_cp_D'] + r['HF_qz_no_cp_correction'], axis=1)
    # df_compare['HF_qz_no_cp_corrected_D'] = df_compare.apply(lambda r: r['HF_qz_no_cp_D'] + r['HF_qz_no_cp_correction'], axis=1)
    df_compare['HF_qz_no_cp_dif_D'] = df_compare.apply(lambda r: r['HF_qz_dimer'] - r['HF_qz_no_cp_D'], axis=1)
    compare_cols = [
        # "HF_qz_dif",
        # "HF_qz_Grimme",
        # "HF_qz_no_cp",
        "HF_qz_d4",
        "HF_qz_G_d4",
        # "HF_qz_no_cp_dif_D",
    ]
    pd.set_option("display.float_format", lambda x: "%.4f" % x)
    print(df_compare[compare_cols].describe())
    longest = sorted([len(i) for i in compare_cols])[-1]
    print(longest)
    for c in compare_cols:
        label = c
        if len(c) < longest:
            label += " " * (longest - len(c))
        RMSE = np.sqrt(np.mean(df_compare[c] ** 2))
        MAD = abs(df_compare[c] - df_compare[c].mean()).mean()
        print(f"{label}  {RMSE = :.4f}  {MAD = :.4f}")
    print(
        """
                                      RMSE                MAD      MD
\qz \cite{caldeweyher2019generally} & 0.49719 & -         & 0.34732 & -0.02597 \\
"""
    )

    #                                       RMSE                MAD      MD
    # \qz \cite{caldeweyher2019generally} & 0.4972 &          & 0.3473 & -0.0260 \\

    df_compare.to_pickle("data/grimme_paper_HF.pkl")
    print(df_compare.columns.values)
    pd.set_option('display.float_format', '{:.10f}'.format)
    for n, r in df_compare.iterrows():
        # print(r['DB'], r["System"], r["HF_qz_dif"])
        # print(r['DB'], r["System"], f"{r['HF_qz_dimer']:.6f}, {r['HF_qz_no_cp_D']:.6f}")
        # if r['id'] == 0:
        print(n)
        print(r)
        print()
        break
    for n, r in df_compare.iterrows():
        print(n, r['id'], r['DB'], r["System"], r["HF_qz_dif"], r['HF_qz_no_cp_dif_D'])
    return
