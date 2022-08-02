import pandas as pd
import numpy as np
from .tools import print_cartesians
import matplotlib.pyplot as plt
from .optimization import compute_bj_opt


def error_stats_method(
    df: pd.DataFrame,
    method: str = "SAPT0",
    count: int = 5,
    verbose: bool = True,
) -> (pd.DataFrame, float, float, float,):
    """
    error_stats_SAPT ...
    """
    df[f"diff_{method}"] = df.apply(lambda r: r["Benchmark"] - (r[method]), axis=1)
    df[f"diff_{method}_abs"] = df[f"diff_{method}"].abs()
    mae = df[f"diff_{method}_abs"].sum() / len(df[f"diff_{method}"])
    rmse = (df[f"diff_{method}"] ** 2).mean() ** 0.5
    max_e = df[f"diff_{method}_abs"].max()

    df = df.sort_values(by=[f"diff_{method}_abs"], ascending=False)
    df = df.reset_index(drop=False)
    if verbose:
        print(f"Method = {method}")
        print("        1. MAE  = %.4f" % mae)
        print("        2. RMSE = %.4f" % rmse)
        print("        3. MAX  = %.4f" % max_e)
        print(
            df[
                [
                    "index",
                    "DB",
                    "Benchmark",
                    method,
                    f"diff_{method}",
                    f"diff_{method}_abs",
                ]
            ].head(10)
        )
        for i in range(count):
            print(f"\nMol {i}")
            print(df.iloc[i])
            print("\nCartesians")
            print_cartesians(df.iloc[i]["Geometry"])
        print()
    return df, mae, rmse, max_e


def get_diffs(
    df,
    params,
    hf_col,
) -> pd.DataFrame:
    """
    get_diffs acquires differences for method
    """
    print(hf_col, params)
    df["d4_%s" % hf_col] = df.apply(
        lambda row: compute_bj_opt(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            row["C6s"],
            row["C8s"],
            row["monAs"],
            row["monBs"],
        ),
        axis=1,
    )
    df[f"diff_{hf_col}"] = df.apply(
        lambda r: r["Benchmark"] - (r[hf_col] + r[f"d4_{hf_col}"]), axis=1
    )
    df[f"diff_{hf_col}_abs"] = df[f"diff_{hf_col}"].abs()
    mae = df[f"diff_{hf_col}_abs"].sum() / len(df[f"diff_{hf_col}"])
    rmse = (df[f"diff_{hf_col}"] ** 2).mean() ** 0.5
    max_e = df[f"diff_{hf_col}_abs"].max()
    return df, mae, rmse, max_e


def analyze_diffs(df, params_dc, hf_cols: [] = ["HF_jdz", "HF_adz"],) -> None:
    """
    analyze_diffs collects diffs and produces plots
    """
    stats = np.zeros((len(hf_cols) + 1, 3))
    for n, i in enumerate(hf_cols):
        df, mae, rmse, max_e = get_diffs(df, params_dc[i], i)
        stats[n, :] = mae, rmse, max_e
    df, mae, rmse, max_e = error_stats_method(df, verbose=False)
    stats[len(hf_cols)] = np.array([ mae, rmse, max_e ])
    print("level\tMAE\tRMSE\tMax")
    for n, i in enumerate(stats):
        if n < len(hf_cols):
            print("%s\t%.4f\t%.4f\t%.4f" % (hf_cols[n], i[0],i[1],i[2]))
        else:
            print("SAPT0\t%.4f\t%.4f\t%.4f" % (i[0],i[1],i[2]))


