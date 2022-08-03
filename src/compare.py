import pandas as pd
import numpy as np
from .tools import print_cartesians
import matplotlib.pyplot as plt
# from .optimization import compute_bj_opt
from .setup import compute_bj_pairs, compute_bj_from_dimer_AB

pd.set_option("display.max_columns", None)

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
            ].head(30)
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
    # d4_energies = []
    # for idx, item in df.iterrows():
    #     e = compute_bj_from_dimer_AB(
    #         params,
    #         item["Geometry"][:, 0],  # pos
    #         item["Geometry"][:, 1:],  # carts
    #         item["monAs"],
    #         item["monBs"],
    #         item["C6s"],
    #     )
    #     d4_energies.append(e)
    #     if idx == 1466:
    #         print(idx, e, item[hf_col], e + item[hf_col])
    # df["d4_%s" % hf_col] = d4_energies
    df["d4_%s" % hf_col] = df.apply(
        lambda row: compute_bj_pairs(
            params,
            row["Geometry"][:, 0],  # pos
            row["Geometry"][:, 1:],  # carts
            # row["C8s"],
            row["monAs"],
            row["monBs"],
            row["C6s"],
            mult_out=627.509,
        ),
        axis=1,
    )
    df["d4_%s_IE" % hf_col] = df.apply(
        lambda r: (r[hf_col] + r[f"d4_{hf_col}"]), axis=1
    )
    df[f"diff_{hf_col}"] = df.apply(
        lambda r: r["Benchmark"] - (r[f"d4_{hf_col}_IE"]), axis=1
    )
    df[f"diff_{hf_col}_abs"] = df[f"diff_{hf_col}"].abs()
    mae = df[f"diff_{hf_col}_abs"].sum() / len(df[f"diff_{hf_col}"])
    rmse = (df[f"diff_{hf_col}"] ** 2).mean() ** 0.5
    max_e = df[f"diff_{hf_col}_abs"].max()
    return df, mae, rmse, max_e


def analyze_diffs(
    df,
    params_dc,
    hf_cols: [] = ["HF_jdz", "HF_adz"],
) -> None:
    """
    analyze_diffs collects diffs and produces plots
    """
    print("HF_D4  [1.61679827, 0.44959224, 3.35743605]")
    stats = np.zeros((len(hf_cols) + 1, 3))
    for n, i in enumerate(hf_cols):
        df, mae, rmse, max_e = get_diffs(df, params_dc[i], i)
        stats[n, :] = mae, rmse, max_e
    df, mae, rmse, max_e = error_stats_method(df, verbose=False)
    stats[len(hf_cols)] = np.array([mae, rmse, max_e])
    print("\nlevel\tMAE\tRMSE\tMax")
    for n, i in enumerate(stats):
        if n < len(hf_cols):
            print("%s\t%.4f\t%.4f\t%.4f" % (hf_cols[n], i[0], i[1], i[2]))
        else:
            print("SAPT0\t%.4f\t%.4f\t%.4f" % (i[0], i[1], i[2]))

    print("SAPT-D3\t%.4f\t%.4f\t%.4f" % (0.496, 0.902, 15.273))

    for n, i in enumerate(hf_cols):
        print(f"\nSorted by {i} absolute difference")
        df = df.sort_values(f"diff_{i}_abs", ascending=False)
        print(
            df[
                [
                    "index",
                    "DB",
                    "Benchmark",
                    f"diff_{hf_cols[0]}_abs",
                    # f"diff_{hf_cols[1]}_abs",
                    # f"diff_{hf_cols[2]}_abs",
                    "diff_SAPT0_abs",
                ]
            ].head(10)
        )
        for i in range(3):
            print(f"\nMol {i}")
            print(df.iloc[i]["monAs"])
            print(df.iloc[i]["monBs"])
            print(
                df.iloc[i][
                    [
                        "index",
                        "DB",
                        "System",
                        "System #",
                        "Benchmark",
                        "HF_jdz",
                        # "HF_adz",
                        "d4_HF_jdz",
                        # "d4_HF_adz",
                        "d4_HF_jdz_IE",
                        # "d4_HF_adz_IE",
                        "SAPT0",
                        "diff_HF_jdz",
                        # "diff_HF_adz",
                        "diff_SAPT0",
                        "diff_HF_jdz_abs",
                        # "diff_HF_adz_abs",
                        "diff_SAPT0_abs",
                    ]
                ],
            )
            print("\nCartesians")
            print_cartesians(df.iloc[i]["Geometry"])
        print()
    df = df.sort_values("diff_SAPT0_abs", ascending=False)
    print(f"\nSorted by SAPT0 absolute difference")
    print(
        df[
            [
                "index",
                "DB",
                "Benchmark",
                f"diff_{hf_cols[0]}_abs",
                # f"diff_{hf_cols[1]}_abs",
                "diff_SAPT0_abs",
            ]
        ].head(10)
    )
    for i in range(3):
        print(f"\nMol {i}")
        print(
            df.iloc[i][
                [
                    "index",
                    "DB",
                    "System",
                    "System #",
                    "Benchmark",
                    "HF_jdz",
                    "HF_adz",
                    "d4_HF_jdz",
                    "d4_HF_adz",
                    "d4_HF_jdz_IE",
                    "d4_HF_adz_IE",
                    "SAPT0",
                    "diff_HF_jdz",
                    "diff_HF_adz",
                    "diff_SAPT0",
                    "diff_HF_jdz_abs",
                    "diff_HF_adz_abs",
                    "diff_SAPT0_abs",
                ]
            ],
        )
        print("\nCartesians")
        print_cartesians(df.iloc[i]["Geometry"])
