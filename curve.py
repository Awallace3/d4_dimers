import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import locald4
from src import paramsTable
from pprint import pprint as pp
import qcelemental as qcel

def compute_d4_ie():
    df = pd.read_pickle("./curves/curves.pkl")
    print(df)
    pp(df.columns.tolist())
    params_2b, params_3b = paramsTable.get_params("SAPT_DFT_OPT_END3")
    print(params_2b)
    # df['SAPT(PBE0)-D4/aDZ (S)'] = df.apply(
    #     lambda r: locald4.compute_bj_dimer_DFTD4(
    #         params_2b, r['Geometry'][:,0], r['Geometry'][:,1:],
    #         r['monAs'], r['monBs'], r['charges'], 
    #         dftd4_bin="/home/amwalla3/miniconda3/envs/d4dimersR/bin/dftd4"
    #     ),
    #     axis=1,
    # )
    def df_geom_bohr(df):
        df['coords'] = df['Geometry'].apply(
            lambda x: np.array(x[:, 1:]) * qcel.constants.conversion_factor("angstrom", "bohr")
        )
        df['Geometry_bohr'] = df.apply(
                lambda r: np.concatenate((r['Geometry'][:, 0].reshape(-1, 1), r['coords']), axis=1),
            axis=1,
        )
        return df

    df = df_geom_bohr(df)

    df['C6s'] = df.apply(
        # TODO
        lambda r: locald4.calc_dftd4_c6_c8_pairDisp2(
            # r['Geometry'][:,0], r['Geometry'][:,1:],
            r['Geometry'][:,0], r['Geometry'][:,1:],
            r['charges'][0],
            dftd4_bin="/home/amwalla3/miniconda3/envs/d4dimersR/bin/dftd4"
        )[0],
        axis=1,
    )
        
    params_2b, params_3b = paramsTable.get_params("SAPT_DFT_pbe0_adz_3_IE_supra")
    # Need (I), compute C6s and then compute
    df['SAPT(PBE0)-D4/aDZ (I)'] = df.apply(
        # TODO
        lambda r: locald4.compute_disp_2B_BJ_dimer_supra(
            r,
            params_2B=params_2b, 
            params_ATM=params_3b,
        ),
        axis=1,
    )
    print(df[['system_id', 'SAPT(PBE0)-D4/aDZ (I)']])
    df.to_pickle("./curves/curves_d4.pkl")
    return


def main():
    pd.set_option('display.max_rows', None)
    compute_d4_ie()
    df = pd.read_pickle("./curves/curves_d4.pkl")
    h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")
    pp(df.columns.tolist())
    df['SAPT_PBE0_DISP'] = df['SAPT_LP_DFT_RP__adz'].apply(lambda x: x[-1])
    print(df[['system_id', 'SAPT(PBE0)-D4/aDZ (I)', 'SAPT_PBE0_DISP']])
    return


if __name__ == "__main__":
    main()
