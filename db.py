import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hrcl_jobs
import hrcl_jobs_psi4

DB_PATH = "db/schr.db"

def run_parallel() -> None:
    """
    run_parallel
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    id_list = []

    if rank == 0:
        con, cur = hrcl_jobs.sqlt.establish_connection(DB_PATH)
        id_list = hrcl_jobs.sqlt.return_id_list(cur, "SAPT0_adz", "main", "main_id", ["NULL"])
        print(f"{id_list = }")

    level_theory = ["SAPT0/aug-cc-pVDZ"]

    headers_sql=["main_id", "Geometry", "monAs", "monBs"]
    output_columns=["SAPT0_adz"]
    table = "main"
    hrcl_jobs.parallel.ms_sl(
        id_list=id_list,
        n_procs=n_procs,
        db_path=DB_PATH,
        table=table,
        level_theory=level_theory,
        run_js_job=hrcl_jobs_psi4.psi4_inps.run_mp_js_grimme_components,
        js_obj=hrcl_jobs.jobspec.grimme_js,
        headers_sql=headers_sql,
        output_columns=output_columns,
        id_label="main_id",
    )
    return

def main():
    run_parallel()
    return


if __name__ == "__main__":
    main()