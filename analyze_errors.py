import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from src import locald4
from src import paramsTable
from qm_tools_aw import tools
import qcelemental as qcel

def get_vdw_radii():
    print("{")
    for i in range(1, 119):
        try:
            v = qcel.vdwradii.get(i, units='bohr')
        except:
            v = 0.00
        print(f"  {v},")
    print("}")
    return


def TT_errors():
    df = pd.read_pickle("./plots/basis_study.pkl")
    df['contact (A)'] = df.apply(lambda r: f"{tools.closest_intermolecular_contact_dimer(r['Geometry'], r['monAs'], r['monBs']):.2f}", axis=1)
    print(df)
    # pp(df.columns.values.tolist())
    df["-D4(TT) unopt"] = df.apply(
        lambda row: locald4.compute_disp_2B_TT_ATM_TT_dimer(
            # row, *paramsTable.param_lookup("2B_TT_START4")
            row, *paramsTable.param_lookup("SAPT0_adz_3_IE_TT")
        ),
        axis=1,
    )
    df["SAPT0-D4(TT) unopt"] = df.apply(
        lambda r: r["SAPT0_adz_3_IE"] + r["-D4(TT) unopt"], axis=1
    )
    df["d4_TT_diff"] = df.apply(
        lambda r: r["Benchmark"] - r["SAPT0-D4(TT) unopt"],
        axis=1,
    )
    df["d4_TT_diff abs"] = df.apply(
        lambda r: abs(r["Benchmark"] - r["SAPT0-D4(TT) unopt"]),
        axis=1,
    )
    df["d4_BJ_diff"] = df.apply(
        lambda r: (r["SAPT0_adz_3_IE_ADZ_d4_diff"]),
        axis=1,
    )
    df["target"] = df.apply(
        lambda r: (r["Benchmark"] - r["SAPT0_adz_3_IE"]),
        axis=1,
    )
    df["d4_BJ_diff abs"] = df.apply(
        lambda r: abs(r["SAPT0_adz_3_IE_ADZ_d4_diff"]),
        axis=1,
    )
    df['d4_BJ'] = df['-D4 (SAPT0_adz_3_IE)']
    df['d4_TT'] = df['-D4(TT) unopt']
    df.sort_values("d4_TT_diff abs", inplace=True, ascending=False)
    pd.set_option("display.max_rows", None)
    print(df[['d4_TT_diff', 'd4_TT_diff abs', 'd4_BJ_diff', 'd4_BJ_diff abs']].describe())
    rmse_tt = np.sqrt(np.mean(df["d4_TT_diff"] ** 2))
    rmse_bj = np.sqrt(np.mean(df["d4_BJ_diff"] ** 2))
    print(f"RMSE TT: {rmse_tt:.2f}, RMSE BJ: {rmse_bj:.2f}")
    # print(df[["d4_TT_diff", "d4_BJ_diff", 'Benchmark', 'DB',]])
    # print(df[["d4_TT", "d4_BJ", 'Benchmark', 'DB', 'contact (A)']])
    # print(df[["d4_TT", "d4_BJ", 'target', 'DB', 'contact (A)']])
    print(df[["d4_TT", 'd4_BJ', 'target', 'DB', 'contact (A)', 'System']])

    tools.print_cartesians(df.iloc[0]['Geometry'])
    return


def main():
    # get_vdw_radii()
    TT_errors()
    return


if __name__ == "__main__":
    main()
