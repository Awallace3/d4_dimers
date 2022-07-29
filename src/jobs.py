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
    psi4.core.print_variables()
    print(wf * 627.509)


def write_psi4_sapt0(
    A: str,
    B: str,
    memory: str = "4 GB",
    basis: str = "cc-pVDZ",
    in_file: str = "d",
    path: str = "tmp",
    jobs: [] = [],
) -> []:
    """
    run_sapt0 computes sapt0 without the dispersion term
    """
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)

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
SAPT0_E10 true
SAPT0_E20IND true
SAPT0_E20Disp false
}
"""
            % basis
        )
        f.write("\nenergy('sapt0')")
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
    jobs.append("%s/%s" % (path, in_file))
    # return jobs


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
    name: str = "dimers",
    memory: str = "4gb",
    job_name: str = "my_psi4_jobs",
    cores: int = 4,
    queue: str = "hive-nvme-sas",
    nodes: int = 1,
    ppn: int = 6,
    walltime: str = "4000:00:00",
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
        d += "psi4 -n%d %s\n" % (cores, i)
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


def create_hf_binding_energies_jobs(
    df: pd.DataFrame,
    basis_set: str,
    out_df: str = "calc.pkl",
) -> None:
    """
    run_hf_binding_energies uses psi4 to calculate monA, monB, and dimer energies with HF
    with a specified basis set.

    The inputted df will be saved to out_df after each computation finishes.
    """
    def_dir = os.getcwd()
    data_dir = "calc_%s" % basis_set
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    int_dir = os.getcwd()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)

    method = "hf/%s" % basis_set
    print(method)
    jobs = []
    for idx, item in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Creating Inputs",
        ascii=True,
    ):
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
        # run_sapt0(mA, mB, basis=basis_set)
        p = "%d_%s" % (idx, item["DB"])
        write_psi4_sapt0(
            mA,
            mB,
            basis=basis_set,
            path=p,
            jobs=jobs,
        )
    os.chdir(int_dir)
    create_pylauncher(jobs)
    os.chdir(def_dir)
    return
