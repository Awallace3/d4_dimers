import pandas as pd
from src.setup import (
    gather_data3,
    gather_data4,
    read_master_regen,
    mol_testing,
    gather_data2_testing,
    gather_data2_testing_mol,
    reorganize_carts_to_split_middle,
)
from src.optimization import optimization, opt_cross_val, HF_only, find_max_e
from src.jobs import create_hf_binding_energies_jobs, run_sapt0_example
from src.harvest import ssi_bfdb_data, harvest_data
from src.compare import error_stats_method, analyze_diffs

"""
dftd4/src/dftd4/param.f90
    case(p_hf)
      param = dftd_param ( & ! (SAW190103)
         &  s6=1.0000_wp, s8=1.61679827 _wp, a1=0.44959224 _wp, a2=3.35743605 _wp )
      !  Fitset: MD= -0.02597 MAD= 0.34732 RMSD= 0.49719
]
"""

# TODO: run psi4 calc on 1466 -> [hf-d4]


def get_params():
    return {
        # "HF_jdz": [0.57791015, 0.67223747, 0.98740773],
        # "HF_adz": [0.5971246, 0.64577916, 1.17106229],
        # "HF_jdz": [1.61679827, 0.44959224, 3.35743605],
        # "HF_adz": [1.61679827, 0.44959224, 3.35743605],
        "HF_jdz": [1.61679827, 0.44959224, 3.35743605],
        # "HF_adz": [0.59489596, 0.47592698, 1.95290271],
        # "HF_dz": [0.50755271, 0.24018135, 2.82780484],
    }


def analyze_max_errors(
    df,
    count: int = 5,
) -> None:
    """
    analyze_max_errors looks at largest max errors
    """
    params_dc = get_params()
    for k, v in params_dc.items():
        find_max_e(df, v, k, count)
    return


# TODO: Show analyze_max_errors in comparison to jeff.out
# TODO: error is in the C6s before dimer splits...


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # gather_data3(output_path="opt.pkl", from_master=True)
    # gather_data4(output_path="opt4.pkl", from_master=False)
    # df = pd.read_pickle("base.pkl")
    # df = ssi_bfdb_data(df)

    # gather_data2_testing()

    # compute_values(7265)
    df = pd.read_pickle("opt4.pkl")

    # analyze_max_errors(df)
    # analyze_diffs(df, get_params(), hf_cols=["HF_jdz"])
    # analyze_diffs(df, params_dc, hf_cols=["HF_jdz", "HF_adz", "HF_dz"])

    # s0, mae_s0, rmse_s0, max_e_s0 = error_stats_method(df, method="SAPT0")

    # basis_set = "dz"
    # df = harvest_data(df, basis_set)
    # pd.to_pickle(df, "base1.pkl")

    # df = pd.read_pickle("base1.pkl")

    basis_set = "jdz"
    hf_key = "HF_%s" % basis_set
    params = [1.61679827, 0.44959224, 3.35743605]
    opt_cross_val(df, nfolds=5, start_params=params, hf_key=hf_key)

    # read_master_regen()

    # bases = ["tz", "atz", "jtz"]
    # bases = ["atz"]
    # create_hf_binding_energies_jobs(
    #     "base1.pkl",
    #     bases,
    #     "calc",
    #     "dimer",
    #     "4gb",
    #     10,
    #     6,
    #     "99:00:00",
    # )
    return


def compute_values(id=1466) -> None:
    """
    compute_values ...
    """
    df = pd.read_pickle("opt.pkl")
    mol = df.loc[id]
    gather_data2_testing_mol(mol)
    return


if __name__ == "__main__":
    main()
