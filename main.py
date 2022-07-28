import pandas as pd
from src.setup import gather_data3
from src.optimization import optimization, opt_cross_val


def main():
    """
    Computes best parameters for SAPT0-D4
    """
    # gather_data3(output_path="opt5.pkl")
    df = pd.read_pickle("opt5.pkl")
    df = df.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    params = [0.96678026, 0.22123128, 4.15329578]
    # optimization(df, params)
    opt_cross_val(df)

    return


if __name__ == "__main__":
    main()
