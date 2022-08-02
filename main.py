import pandas as pd
from src.setup import gather_data3, read_master_regen, expand_opt_df
from src.optimization import optimization, opt_cross_val, HF_only, find_max_e
from src.jobs import create_hf_binding_energies_jobs, run_sapt0_example
from src.harvest import ssi_bfdb_data, harvest_data

"""
dftd4/src/dftd4/param.f90
    case(p_hf)
      param = dftd_param ( & ! (SAW190103)
         &  s6=1.0000_wp, s8=1.61679827 _wp, a1=0.44959224 _wp, a2=3.35743605 _wp )
      !  Fitset: MD= -0.02597 MAD= 0.34732 RMSD= 0.49719
]
"""


def analyze_max_errors(
    df,
    count: int = 5,
) -> None:
    """
    analyze_max_errors looks at largest max errors
    """
    basis_set = "jdz"
    hf_key = "HF_%s" % basis_set
    params = [0.57791015, 0.67223747, 0.98740773]
    find_max_e(df, params, hf_key, count)
    basis_set = "adz"
    hf_key = "HF_%s" % basis_set
    params = [0.5971246, 0.64577916, 1.17106229]
    find_max_e(df, params, hf_key, count)


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # gather_data3(output_path="opt5.pkl")

    # df = pd.read_pickle("base.pkl")
    # df = ssi_bfdb_data(df)
    df = pd.read_pickle("base1.pkl")
    # TODO: Show analyze_max_errors in comparison to jeff.out
    # analyze_max_errors(df, 10)

    # basis_set = "dz"
    # df = harvest_data(df, basis_set)
    # pd.to_pickle(df, "base1.pkl")

    # df = pd.read_pickle("base1.pkl")
    # basis_set = "adz"
    # hf_key = "HF_%s" % basis_set
    # params = [1.61679827, 0.44959224, 3.35743605]
    # opt_cross_val(df, nfolds=5, start_params=params, hf_key=hf_key)

    # read_master_regen()

    bases = ["tz", "atz", "jtz"]
    create_hf_binding_energies_jobs(
        "base1.pkl",
        bases,
        "calc",
        "dimer",
        "4gb",
        6,
        6,
        "99:00:00",
    )

    return


if __name__ == "__main__":
    main()
