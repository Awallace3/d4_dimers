import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_methods(df, m1="Benchmark", m2="qtp_00_adz_saptdft", arr=True) -> None:
    """
    compare_methods
    """
    print(m1, "versus", m2)
    df = df.replace({None: np.nan})
    df = df[df[m2].notna()]
    if arr:
        df["dif"] = df.apply(lambda r: r[m1] - r[m2][0], axis=1)
    else:
        df["dif"] = df.apply(lambda r: r[m1] - r[m2], axis=1)
    mean = df["dif"].mean()
    max_error = df["dif"].abs().max()
    MAD = df["dif"].mad()
    RMSE = (df["dif"] ** 2).mean() ** 0.5
    print(
        f"{mean = } kcal/mol\n{max_error = } kcal/mol\n{MAD = } kcal/mol\n{RMSE = } kcal/mol\n"
    )
