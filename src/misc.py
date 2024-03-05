import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qm_tools_aw import tools
from . import water_data
from . import setup
from . import locald4
from . import paramsTable
from . import optimization
from . import constants
import qcelemental as qcel
import os

ang_to_bohr = constants.Constants().g_aatoau()
hartree_to_kcal_mol = qcel.constants.conversion_factor("hartree", "kcal / mol")

def find_charged_systems(df):
    c = np.array([[0, 1] for i in range(3)])
    cnt_charged = 0
    ids = []
    for i, r in df.iterrows():
        if not np.array_equal(r["charges"], c):
            print(i, r["charges"])
            cnt_charged += 1
            ids.append(i)
    print(f"Found {cnt_charged} charged systems")
    return ids


def print_geom_by_id(df, id):
    tools.print_cartesians(df.loc[id]["Geometry"], symbols=True)
    return


def test_water_dftd4_2_body_and_ATM():
    params = paramsTable.paramsDict()["pbe"]
    print(params)
    df = pd.read_pickle(data_pkl)
    row = df.iloc[3014]
    charges = row["charges"]
    geom = row["Geometry"]
    ma = row["monAs"]
    mb = row["monBs"]
    pos, carts = geom[:, 0], geom[:, 1:]
    d4C6s, d4C8s, pairs, d4e_dimer = locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_dimer = }")
    d4C6s, d4C8s, pairs, d4e_monA = locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[ma], carts[ma], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monA = }")
    d4C6s, d4C8s, pairs, d4e_monB = locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[mb], carts[mb], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monB = }")
    IE = d4e_dimer - d4e_monA - d4e_monB
    # print(f"{IE = }")
    ed4_2_body_IE = locald4.compute_bj_dimer_f90(params, row)
    ed4_2_body_IE /= hartree_to_kcalmol
    print(f"{ed4_2_body_IE = }")
    d4C6s, d4C8s, pairs, d4e_dimer_ATM = locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges[0], dftd4_bin=dftd4_bin, p=params, s9=1.0
    )
    d4C6s, d4C8s, pairs, d4e_monA_ATM = locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[ma], carts[ma], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monA_ATM = }")
    d4C6s, d4C8s, pairs, d4e_monB_ATM = locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[mb], carts[mb], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monB_ATM = }")
    IE_ATM = d4e_dimer_ATM - d4e_monA_ATM - d4e_monB_ATM
    print(f"{IE_ATM = }")
    print(f"{d4e_dimer_ATM = }")
    print(f"{d4e_dimer_ATM - d4e_dimer = }")
    IE_diff_ATM_2_body = IE_ATM - ed4_2_body_IE
    print(f"{IE_diff_ATM_2_body = }")

    assert IE_ATM != IE


def regenerate_D4_data(df, df_path):
    df = setup.generate_D4_data(df)
    df.to_pickle(df_path)
    return


def sensitivity_analysis(df):
    """
    Computes the RMSE for dataset for each parameter and
    see how sensitive these are to parameter changes by
    orders of magnitude.

    RESULT:
    The parameters are not very sensitive to changes in the
    parameters up to 1e-2. Hence, should report to 1e-3 to ensure
    that RMSE stays below a change of 1e-4.
    """
    basis = "SAPT0_adz_3_IE"
    params = paramsTable.get_params(basis + "_2B")[1:4]
    print(basis, params)
    rmse = optimization.compute_int_energy_DISP(params, df, hf_key=basis)
    starting_rmse = rmse
    rmse_diff = abs(starting_rmse - rmse)
    values = [
        1e-6,
        -1e-6,
        1e-5,
        -1e-5,
        1e-4,
        -1e-4,
        1e-3,
        -1e-3,
        1e-2,
        -1e-2,
        0.1,
        -0.1,
    ]
    for i in range(len(params)):
        print("\n\nNext Parameter:\n\n")
        for v in values:
            params = paramsTable.get_params(basis + "_2B")[1:4]
            params[i] += v
            rmse = optimization.compute_int_energy_DISP(params, df, hf_key=basis)
            rmse_diff = abs(starting_rmse - rmse)
            print(f"{i} {v} {rmse_diff}")
            if rmse_diff > 1e-4:
                print(f"BREAKING HERE")
                break
    return


def examine_ATM_TT(df):
    params = paramsTable.get_params("SAPT0_adz_3_IE_2B")

    params_2B, params_ATM = params[0], params[1]
    # params_ATM[-1] = 1.0
    print(params_2B)
    print(params_ATM)
    r = df.iloc[0]
    # ATM = locald4.compute_disp_2B_BJ_ATM_TT_dimer(r, params_2B, params_ATM)
    # return
    df["d4_2B"] = df.apply(
        lambda r: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(r, params_2B, params_ATM),
        axis=1,
    )
    params_ATM[-1] = 1.0
    df["ATM_TT"] = df.apply(
        lambda r: locald4.compute_disp_2B_BJ_ATM_TT_dimer(r, params_2B, params_ATM),
        axis=1,
    )
    df["ATM_CHG"] = df.apply(
        lambda r: locald4.compute_disp_2B_BJ_ATM_CHG_dimer(r, params_2B, params_ATM)
        - r["d4_2B"],
        axis=1,
    )
    df["ATM_diff"] = df["ATM_TT"] - df["ATM_CHG"]
    df["target"] = df["Benchmark"] - (df["SAPT0_adz_3_IE"] + df["d4_2B"])
    for n, r in df.iterrows():
        line = f"{n} {r['target']:.6f} {r['ATM_TT']:.6f} {r['ATM_CHG']:.6f} {r['ATM_diff']:.6f}"
        print(line)
    print("CHG:")
    optimization.compute_int_energy_stats_DISP(
        params, df, "SAPT0_adz_3_IE", print_results=True
    )
    print("TT:")
    optimization.compute_int_energy_stats_DISP_TT(
        params, df, "SAPT0_adz_3_IE", print_results=True
    )
    return

def gather_data(version="schr"):
    # Gather data
    if version == "schr":
        setup.gather_data6(
            output_path="data/d4.pkl",
            from_master=True,
            HF_columns=[
                "HF_dz",
                "HF_jdz",
                "HF_adz",
                "HF_tz",
                "HF_jdz_dftd4",
                "HF_atz",
                "HF_jtz",
            ],
            overwrite=True,
        )
    elif version == "grimme":
        grimme_setup.combine_data_with_new_df()
    elif version == "grimme_paper":
        grimme_setup.read_grimme_dftd4_paper_HF_energies()
    else:
        raise ValueError(f"version {version} not recognized")
    return


def total_bases():
    return [
        "HF_dz",
        "HF_jdz",
        "HF_adz",
        "HF_tz",
        "HF_atz",
        "HF_jdz_no_cp",
        "HF_dz_no_cp",
        "HF_qz",
        "HF_qz_no_cp",
        "HF_qz_no_df",
        "HF_qz_conv_e_4",
        "pbe0_adz_saptdft_ndisp",
    ]


def make_bohr(geometry, ang_to_bohr_convert):
    if ang_to_bohr_convert:
        return np.hstack(
            (np.reshape(geometry[:, 0], (-1, 1)), ang_to_bohr * geometry[:, 1:])
        )
    else:
        return np.hstack(
            (np.reshape(geometry[:, 0], (-1, 1)), 1 / ang_to_bohr * geometry[:, 1:])
        )


def make_geometry_bohr_column(i):
    df, selected = df_names(i)
    tools.print_cartesians(df.iloc[0]["Geometry"])
    df["Geometry_bohr"] = df.apply(lambda x: make_bohr(x["Geometry"], True), axis=1)
    print()
    tools.print_cartesians(df.iloc[0]["Geometry_bohr"])
    print(df.columns.values)
    df.to_pickle(selected)
    return


def grimme_test_atm(df_names_inds=[3, 4]) -> None:
    """
    grimme_test_atm
    """

    for n, i in enumerate(df_names_inds):
        hf_qz_no_cp = "HF_qz_no_cp"
        if i == 4:
            hf_qz_no_cp = "HF_qz"
        df, _ = df_names(i)
        df[hf_qz_no_cp].dropna(inplace=True)
        df["dftd4_ie"] = df.apply(lambda r: r["d4Ds"] - r["d4As"] - r["d4Bs"], axis=1)
        df["diff"] = df.apply(
            lambda r: r["Benchmark"] - (r[hf_qz_no_cp] + r["dftd4_ie"]),
            axis=1,
        )
        print(df[["diff", "Benchmark", hf_qz_no_cp, "dftd4_ie"]].describe())
        # root mean square error of diff
        RMSE = np.sqrt(np.mean(df["diff"] ** 2))
        print(f"{RMSE = :.4f}\n\n")
        if n == 0:
            df1 = df
        elif n == 1:
            df2 = df

    df1["diff_diff"] = df1.apply(
        lambda r: -(r["diff"] - df2.loc[r.name]["diff"]), axis=1
    )
    df1["HF_qz_diff"] = df1.apply(
        lambda r: r["HF_qz_no_cp"] - df2.loc[r.name]["HF_qz"], axis=1
    )
    print(df1[["diff_diff", "HF_qz_diff"]].describe())
    print("HF_qz_df1\tHF_qz_df2\tHF_qz_diff")
    for n in range(len(df1)):
        print(
            df1.iloc[n]["HF_qz_no_cp"], df2.iloc[n]["HF_qz"], df1.iloc[n]["HF_qz_diff"]
        )
    return
def compute_ie_differences(df_num=0):
    df, selected = df_names(df_num)
    params = paramsTable.paramsDict()["HF"]
    if False:
        d4_dimers, d4_mons, d4_diffs = [], [], []
        r4r2_ls = r4r2.r4r2_vals_ls()
        for n, row in df.iterrows():
            print(n)
            ma = row["monAs"]
            mb = row["monBs"]
            charges = row["charges"]
            geom_bohr = row["Geometry_bohr"]
            C6s_dimer = row["C6s"]
            C6s_mA = row["C6_A"]
            C6s_mB = row["C6_B"]

            d4_dimer, d4_mons_individually = locald4.compute_bj_with_different_C6s(
                geom_bohr,
                ma,
                mb,
                charges,
                C6s_dimer,
                C6s_mA,
                C6s_mB,
                params,
            )
            diff = d4_dimer - d4_mons_individually

            d4_dimers.append(d4_dimer)
            d4_mons.append(d4_mons_individually)
            d4_diffs.append(diff)

        df["d4_C6s_dimer"] = d4_dimers
        df["d4_C6s_monomers"] = d4_mons
        df["d4_C6s_diff"] = d4_diffs
        df["d4_C6s_diff_abs"] = abs(df["d4_C6s_diff"])
        print(
            df[
                ["d4_C6s_dimer", "d4_C6s_monomers", "d4_C6s_diff", "d4_C6s_diff_abs"]
            ].describe()
        )
        df.to_pickle(selected)
    df["d4_C6s_dimer_IE"] = pd.to_numeric(
        df["Benchmark"] - (df["HF_adz"] + df["d4_C6s_dimer"])
    )
    df["d4_C6s_monomer_IE"] = pd.to_numeric(
        df["Benchmark"] - (df["HF_adz"] + df["d4_C6s_monomers"])
    )
    print(df[["d4_C6s_dimer_IE", "d4_C6s_monomer_IE"]].describe())
    investigate_pre = [
        "d4_C6s_dimer",
        "d4_C6s_monomers",
        "d4_C6s_diff",
        "d4_C6s_diff_abs",
        "HF_adz",
        "DB",
        # "monAs",
        # "monBs",
        "Benchmark",
    ]
    if True:
        investigate = []
        for i in investigate_pre:
            if i in df.columns.values:
                investigate.append(i)

        print(df[investigate].describe())
        print()
        df["d4_C6s_diff_abs"] = abs(df["d4_C6s_diff"])
        df2 = df.sort_values(by="d4_C6s_diff_abs", inplace=False, ascending=False)

        def print_rows(df, break_after=9000):
            print("i", end="      ")
            for l in investigate:
                if len(l) > 8:
                    print(f"{l[:8]}", end="   ")
                else:
                    while len(l) < 8:
                        l += " "
                    print(f"{l}", end="   ")
            print()
            cnt = 0
            for n, r in df2.iterrows():
                print(n, end="\t")
                for l in investigate:
                    if type(r[l]) != float:
                        print(f"{r[l]}   ", end="    ")
                    else:
                        print(f"{r[l]:.4f}", end="    ")
                print()
                cnt += 1
                if cnt == break_after:
                    return

        print_rows(df)
        cnt = 0
        for n, r in df2.iterrows():
            print(n, r["charges"][0], "angstrom")
            tools.print_cartesians_dimer(
                r["Geometry"], r["monAs"], r["monBs"], r["charges"]
            )
            cnt += 1
            if cnt > 3:
                break
    if True:
        idx = 3014
        r = df.iloc[idx]
        print("FINAL:", idx)
        print(r["charges"][0], "angstrom")
        tools.print_cartesians_dimer(
            r["Geometry"], r["monAs"], r["monBs"], r["charges"]
        )
        # print(r['charges'][0], 'bohr')
        # tools.print_cartesians_dimer(r["Geometry_bohr"], r['monAs'], r['monBs'], r['charges'])

    return df


def charge_comparison():
    df, selected = df_names(0)
    print(df.columns.values)
    def_charge = [[0, 1] for i in range(3)]
    cnt_correct = 0
    cnt_wrong = 0
    for n, r in df.iterrows():
        line = f"{n} {r['charges']} {r['HF INTERACTION ENERGY']:.4f} {r['HF_jdz']:.4f}"
        e_diff = abs(r["HF INTERACTION ENERGY"] - r["HF_jdz"])

        if n < 0:
            print(line)
        elif not np.all(r["charges"] == def_charge):
            if e_diff < 0.001:
                cnt_correct += 1
            else:
                cnt_wrong += 1
                print(line)
    print(cnt_correct, cnt_wrong)
    return


def sum_IE(vals):
    if vals is not None:
        return sum(vals[1:4])
    else:
        return np.nan


def total_IE(vals):
    if vals is not None:
        return vals[0]
    else:
        return np.nan


def merge_SAPT0_results_into_df():
    df, selected = df_names(6)
    print(df)
    df2 = pd.read_pickle("data/schr_sapt0.pkl")
    print(df2.columns.values)
    copy_SAPT0_cols = [
        "id",
        "SAPT0_dz",
        "SAPT0_jdz",
        "SAPT0_adz",
        "SAPT0_tz",
        "SAPT0_mtz",
        "SAPT0_jtz",
        "SAPT0_atz",
    ]
    for i in copy_SAPT0_cols:
        df[i] = df2[i]
    for i in copy_SAPT0_cols:
        print(df[i])

    for i in [
        j
        for j in df.columns.values
        if "SAPT0_" in j
        if j not in ["SAPT0", "SAPT0_aqz"]
        if "_IE" not in j
    ]:
        print(f'"{i}_3_IE",')
        df[i + "_3_IE"] = df.apply(lambda r: sum_IE(r[i]), axis=1)
        df[i + "_IE"] = df.apply(lambda r: total_IE(r[i]), axis=1)
    df.to_pickle(selected)
    return


def merge_SAPTDFT_results_into_df():
    df, selected = df_names(6)
    df2 = pd.read_pickle("data/schr_saptdft.pkl")
    print(df2.columns.values)
    copy_SAPT0_cols = [
        "id",
        "SAPT_DFT_adz",
        "SAPT_DFT_atz",
    ]
    for i in copy_SAPT0_cols:
        df[i] = df2[i]
    for i in copy_SAPT0_cols:
        print(df[i])
    for i in [
        j
        for j in df.columns.values
        if "SAPT_DFT_" in j
        if j not in ["SAPT0", "SAPT0_aqz"]
        if "_IE" not in j
    ]:
        print(f'"{i}_3_IE",')
        df[i + "_3_IE"] = df.apply(lambda r: sum_IE(r[i]), axis=1)
        df[i + "_IE"] = df.apply(lambda r: total_IE(r[i]), axis=1)
    print(df.columns.values)
    df.to_pickle(selected)
    return


def main():
    # water_data.water_data_collect()
    return


if __name__ == "__main__":
    main()
