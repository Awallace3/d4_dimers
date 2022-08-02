import psi4
import pandas as pd
import numpy as np
from .tools import print_cartesians, np_carts_to_string
import os
from tqdm import tqdm


def run_sapt0_example():
    basis = "cc-pvdz"
    memory = "1 GB"
    geom = """
0 1
He 0.0 0.0 0.0
--
He 2.0 0.0 0.0
"""
    # write as input files with 6 threads and submit on hive
    print(geom)
    psi4.geometry(geom)
    psi4.set_memory(memory)
    psi4.core.set_num_threads(2)
    psi4.set_options(
        {
            "basis": basis,
            "freeze_core": "true",
            "guess": "sad",
            "scf_type": "df",
            "SAPT0_E10": "true",
            "SAPT0_E20IND": "true",
            "SAPT0_E20Disp": "false",
        }
    )
    wf = psi4.energy("sapt0")
    # psi4.core.print_variables()
    # print(wf * 627.509)


def write_psi4_sapt0(
    A: str,
    B: str,
    meth_basis_dir: str,
    memory: str = "4 GB",
    basis: str = "aug-cc-pvdz",
    in_file: str = "d",
) -> []:
    """
    run_sapt0 computes sapt0 without the dispersion term
    """
    b = basis_labels(meth_basis_dir)
    if not os.path.exists(meth_basis_dir):
        os.mkdir(meth_basis_dir)
    os.chdir(meth_basis_dir)

    geom = "0 1\n%s--\n0 1\n%s" % (A, B)
    with open("%s.dat" % (in_file), "w") as f:
        f.write("memory %s\n" % memory)
        f.write("molecule mol {\n%s}\n\n" % geom)
        f.write(
            """
set {
basis %s
freeze_core true
guess sad
scf_type df
}
"""
            % basis
        )
        f.write("\nenergy('hf', bsse_type='cp')")
    #     with open("%s.pbs" % in_file, "w") as f:
    #         f.write(
    #             """
    # #PBS -N %s
    # #PBS -q hive-nvme-sas
    # #PBS -l nodes=1:ppn=6
    # #PBS -l mem=4gb
    # #PBS -l walltime=1000:00:00
    # """
    #             % path
    #         )

    os.chdir("..")
    return


def run_sapt0(A: str, B: str, memory: str = "4 GB", basis="jun-cc-pvdz"):
    """
    run_sapt0 computes sapt0 without the dispersion term
    """
    geom = "0 1\n%s--\n0 1\n%s" % (A, B)
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
            "SAPT0_E10": "true",
            "SAPT0_E20IND": "true",
            "SAPT0_E20Disp": "false",
        }
    )
    psi4.energy("sapt0")

    return


def create_pylauncher(
    jobs: [],
    data_dir: str,
    basis: str = "adz",
    name: str = "dimers",
    memory: str = "4gb",
    job_name: str = "my_psi4_jobs",
    cores: int = 6,  # cores and ppn should match
    queue: str = "hive",
    nodes: int = 10,  # how many for all jobs
    ppn: int = 6,
    walltime: str = "30:00:00",
) -> None:
    """
    create_pylauncher creates pylauncher and psi4_jobs
    """

    l = """#!/usr/bin/env python
import pylauncher3

myjob = '%s'
pylauncher3.ClassicLauncher(myjob, cores=%d)
""" % (
        job_name,
        cores,
    )
    d = ""
    for i in jobs:
        d += "psi4 -n%d %s/%s\n" % (cores, data_dir, i)
    s = """#!/bin/bash
#PBS -N %s
#PBS -q %s
#PBS -j oe
#PBS -l nodes=%d:ppn=%d	    #number of cores and cores per node required
#PBS -l pmem=%s	    #memory per core
#PBS -l walltime=%s
#PBS -m p
#PBS -V

module load anaconda3
conda activate psi4       #psi4 is my env w/ psi4 and paramiko
module load pylauncher/3.0

cd $PBS_O_WORKDIR
echo "Starting job."
python ./launcher.py
echo "Ending job."
""" % (
        name,
        queue,
        nodes,
        ppn,
        memory,
        walltime,
    )
    with open("launcher.py", "w") as f:
        f.write(l)
    with open(job_name, "w") as f:
        f.write(d)
    with open("submit.pbs", "w") as f:
        f.write(s)
    return


def basis_labels(
    basis: str,
    method: str = "HF",
) -> (str, str):
    """
        basis_labels converts basis to psi4 input basis
    ["tz", "atz", "jtz"]
    """
    if basis == "adz":
        return "aug-cc-pvdz", "%s_%s" % (method, basis)
    elif basis == "jdz":
        return "jun-cc-pvdz", "%s_%s" % (method, basis)
    elif basis == "dz":
        return "cc-pvdz", "%s_%s" % (method, basis)
    elif basis == "tz":
        return "cc-pvtz", "%s_%s" % (method, basis)
    elif basis == "atz":
        return "aug-cc-pvtz", "%s_%s" % (method, basis)
    elif basis == "jtz":
        return "jun-cc-pvtz", "%s_%s" % (method, basis)
    else:
        return basis, "%s_%s" % (method, basis)


class DirectoryAlreadyExists(Exception):
    """Already created calc directory"""

    def __init__(self, name):
        self.name = name
        super().__init__(self.name)


def expand_opt_df_jobs(
    df,
    columns_to_add: list = [
        "HF_dz",
        "HF_dt",
        "HF_adz",
        "HF_adt",
        "HF_jdz",
        "HF_jdt",
    ],
    prefix: str = "",
    replace_HF: bool = False,
) -> pd.DataFrame:
    """
    expand_opt_df_jobs adds columns with nan
    """
    if replace_HF:
        df = replace_hf_int_HF_jdz(df)
    columns_to_add = ["%s%s" % (prefix, i) for i in columns_to_add]
    for i in columns_to_add:
        if i not in df:
            df[i] = np.nan
    # print(df.columns.values)
    return df


def create_hf_binding_energies_jobs(
    df_p: "base1.pkl",
    bases: [],
    data_dir: str = "calc",
    in_file: str = "dimer",
    memory: str = "4gb",
    nodes: int = 5,
    cores: int = 4,
    walltime: str = "30:00:00",
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
    # df.loc[[0, 1, 2, 3, 4, 5]]
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
