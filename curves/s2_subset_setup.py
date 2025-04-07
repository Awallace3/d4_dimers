from hrcl_jobs_psi4 import jobspec
from hrcl_jobs_psi4 import psi4_inps
import hrcl_jobs
import numpy as np
from qm_tools_aw import tools
import os
import pandas as pd
from pprint import pprint as pp
import qcelemental as qcel

# get absolute path to current directory
path = os.path.abspath(os.getcwd())

DB_NAME = "./db/s2.db"
BASE_PKL = "./data/S2/s2_df_base.pkl"
OUT_PKL = './data/S2/s2_df_out.pkl'

output_vars = [
 'CUSTOM SCS-MP2 CORRELATION ENERGY',
 'CUSTOM SCS-MP2 TOTAL ENERGY',
 'HF TOTAL ENERGY',
 'MP2 CORRELATION ENERGY',
 'MP2 DOUBLES ENERGY',
 'MP2 OPPOSITE-SPIN CORRELATION ENERGY',
 'MP2 SAME-SPIN CORRELATION ENERGY',
 'MP2 SINGLES ENERGY',
 'MP2 TOTAL ENERGY',
 'SAPT ALPHA',
 'SAPT CT ENERGY',
 'SAPT DISP ENERGY',
 'SAPT DISP20 ENERGY',
 'SAPT DISP21 ENERGY',
 'SAPT DISP22(SDQ) ENERGY',
 'SAPT DISP22(T) ENERGY',
 'SAPT DISP30 ENERGY',
 'SAPT ELST ENERGY',
 'SAPT ELST10,R ENERGY',
 'SAPT ELST12,R ENERGY',
 'SAPT ELST13,R ENERGY',
 'SAPT ENERGY',
 'SAPT EST.DISP22(T) ENERGY',
 'SAPT EXCH ENERGY',
 'SAPT EXCH-DISP20 ENERGY',
 'SAPT EXCH-DISP30 ENERGY',
 'SAPT EXCH-IND-DISP30 ENERGY',
 'SAPT EXCH-IND20,R ENERGY',
 'SAPT EXCH-IND22 ENERGY',
 'SAPT EXCH-IND30,R ENERGY',
 'SAPT EXCH10 ENERGY',
 'SAPT EXCH10(S^2) ENERGY',
 'SAPT EXCH11(S^2) ENERGY',
 'SAPT EXCH12(S^2) ENERGY',
 'SAPT EXCHSCAL',
 'SAPT EXCHSCAL1',
 'SAPT EXCHSCAL3',
 'SAPT HF TOTAL ENERGY',
 'SAPT HF(2) ALPHA=0.0 ENERGY',
 'SAPT HF(2) ENERGY',
 'SAPT HF(3) ENERGY',
 'SAPT IND ENERGY',
 'SAPT IND-DISP30 ENERGY',
 'SAPT IND20,R ENERGY',
 'SAPT IND22 ENERGY',
 'SAPT IND30,R ENERGY',
 'SAPT MP2 CORRELATION ENERGY',
 'SAPT MP2(2) ENERGY',
 'SAPT MP2(3) ENERGY',
 'SAPT MP4 DISP',
 'SAPT TOTAL ENERGY',
 'SAPT0 DISP ENERGY',
 'SAPT0 ELST ENERGY',
 'SAPT0 EXCH ENERGY',
 'SAPT0 IND ENERGY',
 'SAPT0 TOTAL ENERGY',
 'SAPT2 DISP ENERGY',
 'SAPT2 ELST ENERGY',
 'SAPT2 EXCH ENERGY',
 'SAPT2 IND ENERGY',
 'SAPT2 TOTAL ENERGY',
 'SAPT2+ DISP ENERGY',
 'SAPT2+ ELST ENERGY',
 'SAPT2+ EXCH ENERGY',
 'SAPT2+ IND ENERGY',
 'SAPT2+ TOTAL ENERGY',
 'SAPT2+(3) DISP ENERGY',
 'SAPT2+(3) ELST ENERGY',
 'SAPT2+(3) EXCH ENERGY',
 'SAPT2+(3) IND ENERGY',
 'SAPT2+(3) TOTAL ENERGY',
 'SAPT2+(3)(CCD) ELST ENERGY',
 'SAPT2+(3)(CCD) EXCH ENERGY',
 'SAPT2+(3)(CCD) IND ENERGY',
 'SAPT2+(3)(CCD)DMP2 ELST ENERGY',
 'SAPT2+(3)(CCD)DMP2 EXCH ENERGY',
 'SAPT2+(3)(CCD)DMP2 IND ENERGY',
 'SAPT2+(3)DMP2 DISP ENERGY',
 'SAPT2+(3)DMP2 ELST ENERGY',
 'SAPT2+(3)DMP2 EXCH ENERGY',
 'SAPT2+(3)DMP2 IND ENERGY',
 'SAPT2+(3)DMP2 TOTAL ENERGY',
 'SAPT2+(CCD) ELST ENERGY',
 'SAPT2+(CCD) EXCH ENERGY',
 'SAPT2+(CCD) IND ENERGY',
 'SAPT2+(CCD)DMP2 ELST ENERGY',
 'SAPT2+(CCD)DMP2 EXCH ENERGY',
 'SAPT2+(CCD)DMP2 IND ENERGY',
 'SAPT2+3 DISP ENERGY',
 'SAPT2+3 ELST ENERGY',
 'SAPT2+3 EXCH ENERGY',
 'SAPT2+3 IND ENERGY',
 'SAPT2+3 TOTAL ENERGY',
 'SAPT2+3(CCD) ELST ENERGY',
 'SAPT2+3(CCD) EXCH ENERGY',
 'SAPT2+3(CCD) IND ENERGY',
 'SAPT2+3(CCD)DMP2 ELST ENERGY',
 'SAPT2+3(CCD)DMP2 EXCH ENERGY',
 'SAPT2+3(CCD)DMP2 IND ENERGY',
 'SAPT2+3DMP2 DISP ENERGY',
 'SAPT2+3DMP2 ELST ENERGY',
 'SAPT2+3DMP2 EXCH ENERGY',
 'SAPT2+3DMP2 IND ENERGY',
 'SAPT2+3DMP2 TOTAL ENERGY',
 'SAPT2+DMP2 DISP ENERGY',
 'SAPT2+DMP2 ELST ENERGY',
 'SAPT2+DMP2 EXCH ENERGY',
 'SAPT2+DMP2 IND ENERGY',
 'SAPT2+DMP2 TOTAL ENERGY',
 'SCS-MP2 CORRELATION ENERGY',
 'SCS-MP2 TOTAL ENERGY',
 'SCS-SAPT0 ELST ENERGY',
 'SCS-SAPT0 EXCH ENERGY',
 'SCS-SAPT0 IND ENERGY',
 'SSAPT0 DISP ENERGY',
 'SSAPT0 ELST ENERGY',
 'SSAPT0 EXCH ENERGY',
 'SSAPT0 IND ENERGY',
 'SSAPT0 TOTAL ENERGY',
]

def setup_s2_db():
    df = pd.read_pickle(BASE_PKL)
    print(df)
    # convert datatype of ['monAs', 'monBs', 'charges'] to np.integer
    df["id"] = [i for i in range(0, len(df))]
    df["monAs"] = df["monAs"].apply(lambda x: np.array(x, dtype=np.int32))
    df["monBs"] = df["monBs"].apply(lambda x: np.array(x, dtype=np.int32))
    df["charges"] = df["charges"].apply(lambda x: np.array(x, dtype=np.int32))
    df["Geometry"] = df["Geometry"].apply(lambda x: np.array(x, dtype=np.float64))
    sql = hrcl_jobs.pgsql.convert_to_sql(df, "main")
    # create db
    con, cur = hrcl_jobs.sqlt.establish_connection(DB_NAME)
    print("Connected")
    # sql = sql.split("),\n")[0]
    sql_cmds = sql.split(";")
    for sql in sql_cmds:
        print(sql)
        cur.execute(sql)
    con.commit()
    hrcl_jobs.sqlt.create_update_table(
        DB_NAME,
        'main',
        {f'"{k} adz"': 'DOUBLE PRECISION' for k in output_vars},
    )
    hrcl_jobs.sqlt.create_update_table(
        DB_NAME,
        'main',
        {f'"{k} atz"': 'DOUBLE PRECISION' for k in output_vars},
    )
    return


def generate_input_files(basis_sets=['aug-cc-pvdz', 'aug-cc-pvtz']):
    con, cur = hrcl_jobs.sqlt.establish_connection(DB_NAME)
    print("Connected")
    ids = cur.execute("SELECT id FROM main").fetchall()
    ids = [i[0] for i in ids]
    for id in ids:
        for bs in basis_sets:
            print(id, bs)
            js = hrcl_jobs.sqlt.collect_id_into_js(
                cur,
                headers=jobspec.sapt0_js_headers(),
                mem="100gb",
                extra_info={
                    "level_theory": [f"sapt2+3(ccd)dmp2/{bs}"],
                    "options": {
                        "maxiter": 100,
                        "E_CONVERGENCE": 8,
                        "D_CONVERGENCE": 8,
                        "freeze_core": "True",
                        "guess": "sad",
                        "scf_type": "df",
                    },
                    "num_threads": 10,
                    "function_call": f"psi4.energy('sapt2+3dmp2/{bs}')",
                    "out": {"path": "s2", "version": "1"},
                },
                dataclass_obj=jobspec.sapt0_js,
                id_value=id,
                id_label="id",
                table="main",
            )
            js.geometry = hrcl_jobs.pgsql.convert_sql_array_to_numpy(js.geometry)
            js.geometry[:, 1:] *= qcel.constants.bohr2angstroms
            js.monAs = hrcl_jobs.pgsql.convert_sql_array_to_numpy(js.monAs)
            js.monBs = hrcl_jobs.pgsql.convert_sql_array_to_numpy(js.monBs)
            js.charges = hrcl_jobs.pgsql.convert_sql_array_to_numpy(js.charges)
            job_dir = psi4_inps.generate_job_dir(js, js.extra_info["level_theory"][0], 0)
            job_dir = path + "/" + job_dir
            print(job_dir)
            js.extra_info[
                "sbatch_file"
            ] = f"""#!/bin/bash
#SBATCH -JS2-{id}
#SBATCH -Ahive-cs207
#SBATCH -N1 --ntasks=1 --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH -t96:00:00
#SBATCH -phive
#SBATCH -oslurm-%j.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu

cd {job_dir}
. "/storage/home/hhive1/awallace43/data/miniconda3/etc/profile.d/conda.sh"
conda activate /storage/hive/project/chem-sherrill/awallace43/.conda/envs/p4dev19

export PATH=/storage/hive/project/chem-sherrill/awallace43/gits/psi4_amw/build_dlpno_ccsd_t_weak_pairs/stage/bin:$PATH 
export PYTHONPATH=/storage/hive/project/chem-sherrill/awallace43/gits/psi4_amw/build_dlpno_ccsd_t_weak_pairs/stage/lib:$PYTHONPATH
export SCRATCH=${{TMPDIR}}
export PSI_SCRATCH=${{TMPDIR}}

python3 p4.py
    """
            psi4_inps.create_psi4_input_file(js)
    return

def abbreviate_basis_set(bs):
    if bs == 'aug-cc-pvdz':
        return 'adz'
    elif bs == 'aug-cc-pvtz':
        return 'atz'
    else:
        return bs


def gather_results(basis_sets=['aug-cc-pvdz', 'aug-cc-pvtz']):
    con, cur = hrcl_jobs.sqlt.establish_connection(DB_NAME)
    ids = [i[0] for i in cur.execute("SELECT id FROM main").fetchall()]
    for bs in basis_sets:
        bs_abbrev = abbreviate_basis_set(bs)
        for id in ids:
            datapath = f"./s2/{id}/sapt2+3_ccd_dmp2_{bs.replace('-', '_')}_1/vars.json"
            if not os.path.exists(datapath):
                continue
            print(datapath)
            qcvars = tools.json_to_dict(datapath)
            for key, value in qcvars.items():
                if key in output_vars:
                    cmd = f"""UPDATE main SET "{key} {bs_abbrev}" = {value} WHERE id = {id}"""
                    cur.execute(cmd)
    con.commit()
    hrcl_jobs.sqlt.table_to_df_pkl(DB_NAME, 'main', OUT_PKL, id_label='id')
    df = pd.read_pickle(OUT_PKL)
    return df


def main():
    # setup_s2_db()
    # generate_input_files()
    # return
    df = gather_results(['aug-cc-pvdz', 'aug-cc-pvtz'])
    df = pd.read_pickle(OUT_PKL)
    print(df)
    pd.set_option("display.max_rows", None)
    print(df[["system_id", "SAPT0 TOTAL ENERGY adz"]])
    # count NaNs in SAPT2+3(CCD)DMP2 TOTAL ENERGY
    # print(df["SAPT2+3(CCD)DMP2 TOTAL ENERGY"].isna().sum())
    return


if __name__ == "__main__":
    main()
