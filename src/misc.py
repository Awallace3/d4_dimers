import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qm_tools_aw import tools
from . import water_data
from . import setup


def find_charged_systems():
    df = pd.read_pickle("data/d4.pkl")
    c = np.array([[0, 1] for i in range(3)])
    cnt_charged = 0
    for i, r in df.iterrows():
        if not np.array_equal(r["charges"], c):
            print(i, r["charges"])
            cnt_charged += 1
    print(f"Found {cnt_charged} charged systems")
    return


def print_geom_by_id(df, id):
    tools.print_cartesians(df.loc[id]["Geometry"], symbols=True)
    return



def test_water_dftd4_2_body_and_ATM():
    params = src.paramsTable.paramsDict()["pbe"]
    print(params)
    df = pd.read_pickle(data_pkl)
    row = df.iloc[3014]
    charges = row["charges"]
    geom = row["Geometry"]
    ma = row["monAs"]
    mb = row["monBs"]
    pos, carts = geom[:, 0], geom[:, 1:]
    d4C6s, d4C8s, pairs, d4e_dimer = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_dimer = }")
    d4C6s, d4C8s, pairs, d4e_monA = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[ma], carts[ma], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monA = }")
    d4C6s, d4C8s, pairs, d4e_monB = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[mb], carts[mb], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monB = }")
    IE = d4e_dimer - d4e_monA - d4e_monB
    # print(f"{IE = }")
    ed4_2_body_IE = src.locald4.compute_bj_dimer_f90(params, row)
    ed4_2_body_IE /= hartree_to_kcalmol
    print(f"{ed4_2_body_IE = }")
    d4C6s, d4C8s, pairs, d4e_dimer_ATM = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos, carts, charges[0], dftd4_bin=dftd4_bin, p=params, s9=1.0
    )
    d4C6s, d4C8s, pairs, d4e_monA_ATM = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        pos[ma], carts[ma], charges[0], dftd4_bin=dftd4_bin, p=params
    )
    print(f"{d4e_monA_ATM = }")
    d4C6s, d4C8s, pairs, d4e_monB_ATM = src.locald4.calc_dftd4_c6_c8_pairDisp2(
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

def main():
    # water_data.water_data_collect()
    return


if __name__ == "__main__":
    main()
