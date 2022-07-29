import pandas as pd
from src.setup import gather_data3, read_master_regen, expand_opt_df
from src.optimization import optimization, opt_cross_val
from src.jobs import create_hf_binding_energies_jobs, run_sapt0_example
from src.harvest import ssi_bfdb_data

# TODO: write mae on testing set
# TODO: check on max_e specifically
# TODO: look at HF damping parameters in f90

"""
dftd4/src/dftd4/param.f90

    case(p_hf)
      param = dftd_param ( & ! (SAW190103)
         &  s6=1.0000_wp, s8=1.61679827 _wp, a1=0.44959224 _wp, a2=3.35743605 _wp )
      !  Fitset: MD= -0.02597 MAD= 0.34732 RMSD= 0.49719
]
"""

def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # gather_data3(output_path="opt5.pkl")
    # run_sapt0_example()
    # df = pd.read_pickle("base.pkl")
    df = pd.read_pickle("opt5.pkl")
    basis_set = "cc-pvdz"
    # create_hf_binding_energies_jobs(df, basis_set)
    # df = df.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # df = expand_opt_df(df, replace_HF=True)
    # pd.to_pickle(df, "base.pkl")

    # ssi_bfdb_data(df)
    params = [1.61679827, 0.44959224, 3.35743605]
    opt_cross_val(df, nfolds=5, start_params=params)
    # read_master_regen()

    return


if __name__ == "__main__":
    main()
