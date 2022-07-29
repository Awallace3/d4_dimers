import pandas as pd
from src.setup import gather_data3
from src.optimization import optimization, opt_cross_val
from src.jobs import create_hf_binding_energies_jobs, run_sapt0_example


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # gather_data3(output_path="opt5.pkl")

    # run_hf_binding_energies(df, basis_set)
    # run_sapt0_example()

    df = pd.read_pickle("opt5.pkl")
    basis_set = "cc-pvdz"
    create_hf_binding_energies_jobs(df, basis_set)


    # df = df.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # params = [3.02227550, 0.47396846, 4.49845309]
    # opt_cross_val(df, nfolds=5, start_params=params)

    return


if __name__ == "__main__":
    main()
