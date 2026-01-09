import pandas as pd
import src
import subprocess, os
from qm_tools_aw import tools
from pprint import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

ang_to_bohr = src.constants.Constants().g_aatoau()

def plot_ie_curve(
        df,
        sapt_col,
        disp_col=None,
        db='NBC10',
        system_num=0,
        R_max=1.75,
    ):
    # df_sys = df[(df['DB'] == db) & (df['System #'] == system_num)]
    df_sys = df[(df['DB'] == db) & (df['System Label'] == system_num)]
    # df_sys = df[(df['DB'] == db)]
    print(df_sys['System Label'])
    import matplotlib.pyplot as plt
    from qm_tools_aw import tools
    df_sys.sort_values(by='R', inplace=True)
    df_sys.reset_index(drop=True, inplace=True)
    df_sys['elst'] = df_sys[sapt_col].apply(lambda x: x[1])
    df_sys['exch'] = df_sys[sapt_col].apply(lambda x: x[2])
    df_sys['indu'] = df_sys[sapt_col].apply(lambda x: x[3])
    df_sys['disp'] = df_sys[sapt_col].apply(lambda x: x[4])
    df_sys = df_sys[df_sys['R'] <= R_max]
    # pd.set_option('display.max_columns', None)
    df_sys['total'] = df_sys.apply(lambda x: x['elst'] + x['exch'] + x['indu'] + x['disp'], axis=1)
    print(df_sys[['R', 'system_id', 'total', 'Benchmark']])
    if db.lower() == 'nbc10':
        df_sys['distance'] = df_sys['R']
    else:
        df_sys['distance'] = df_sys['R']

    fig = plt.figure(dpi=400)
    plt.xlabel('Percent of Equilibrium Distance', fontsize=16)
    plt.ylabel('Interaction Energy (kcal/mol)', fontsize=16)
    if not os.path.exists(f"plots/{db}"):
        os.makedirs(f"plots/{db}")
    plt.plot(df_sys['distance'], df_sys['Benchmark'], label='Ref.', color='grey', linestyle='--', linewidth=3.0)
    plt.legend(fontsize=16)
    y_max = max(df_sys['exch']) if max(df_sys['exch']) > max(df_sys['elst']) else max(df_sys['elst'])
    y_min = min(df_sys['disp']) if min(df_sys['disp']) < min(df_sys['elst']) else min(df_sys['elst'])
    y_max = y_max + 0.05 * y_max
    y_min = y_min + 0.05 * y_min
    # plt.ylim([y_min, y_max])
    plt.ylim([-5, 5])
    plt.savefig(f'plots/{db}/system_{system_num}_ie_curve_benchmark.png')
    plt.plot(df_sys['distance'], df_sys['elst'], linewidth=2.0, label='Elst', color='red')
    plt.plot(df_sys['distance'], df_sys['exch'], linewidth=2.0, label='Exch', color='green')
    plt.plot(df_sys['distance'], df_sys['indu'], linewidth=2.0, label='Indu', color='blue')
    plt.plot(df_sys['distance'], df_sys['disp'], linewidth=2.0, label='Disp', color='orange')
    plt.plot(df_sys['distance'], df_sys['total'],linewidth=2.0,  label='Total', color='black')
    plt.legend(fontsize=16)
    plt.savefig(f'plots/{db}/system_{system_num}_ie_curve.png')
    tools.print_cartesians(df_sys.iloc[0]["Geometry"])
    print()
    tools.print_cartesians(df_sys.iloc[len(df_sys) - 1]["Geometry"])
    return

def single_curve():
    df_name = "plots/ddft_study.pkl"
    # df_name = "plots/basis_study.pkl"
    # df_name = "dfs/los_saptdft_atz.pkl"
    # df_name = "dfs/schr_dft2.pkl"
    if not os.path.exists(df_name):
        print("Cannot find ./plots/basis_study.pkl, creating it now...")
        subprocess.call("cat plots/basis_study-* > plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("tar -xzf plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("rm plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("mv basis_study.pkl plots/basis_study.pkl", shell=True)
    df = pd.read_pickle(df_name)
    print(df['DB'].unique())
    # for i in range(1, 11):
    # print(f"Plotting system {i}")
    plot_ie_curve(
        df,
        sapt_col='SAPT0_adz',
        db='s66x8',
        system_num="27_Benzene-Pyridine",
    )
    return 

def ar_ar_setup():
    el1 = 18.0 # Argon
    el1 = 1.0 # Hydrogen
    el1 = 6.0 # Carbon
    el2 = el1
    el2 = 1.0
    ar_ar_geom = np.array([[el1, 0.0, 0.0, 0.0], [el2, 0.0, 0.0, 0.0]])
    geoms = []
    for i in np.linspace(0.5, 12.0, 50):
        cp_geom = ar_ar_geom.copy()
        cp_geom[-1, -1] = i
        geoms.append(cp_geom)
    c = np.array([[
        0, 1
    ] for i in range(3)])
    data = {
        "Geometry": geoms,
        "monAs": [np.array([0]) for i in range(len(geoms))],
        "monBs": [np.array([1]) for i in range(len(geoms))],
        "charges": [c for i in range(len(geoms))],
    }
    df = pd.DataFrame(data)
    print(df)
    c6_dimers, c6_monAs, c6_monBs, t6_1s, t6_2s, t8_1s, t8_2s, dist = [], [], [], [], [], [], [], []
    e1s, e2s = [], []
    saptdft_d4_params, _ = src.paramsTable.generate_2B_ATM_param_subsets(src.paramsTable.get_params("SAPT_DFT_pbe0_adz_3_IE"))
    # saptdft_d4_params, _ = src.paramsTable.generate_2B_ATM_param_subsets(src.paramsTable.get_params("SAPT_DFT_OPT_END3"))
    hfd4_params, _ = src.paramsTable.generate_2B_ATM_param_subsets(src.paramsTable.get_params("HF_ATM"))
    print(f"{saptdft_d4_params = }")
    print(f"{hfd4_params = }")
    for n, row in df.iterrows():
        ma = row["monAs"]
        mb = row["monBs"]
        charges = row["charges"]
        geom = row["Geometry"]

        cD = geom[:, 1:]
        pD = geom[:, 0]
        tools.print_cartesians_pos_carts(pD, cD)
        # pA, cA = pD[ma], cD[ma, :]
        # pB, cB = pD[mb], cD[mb, :]
        # C6s_dimer, C6s_mA, C6s_mB = src.locald4.calc_dftd4_c6_for_d_a_b(
        #     cD, pD, pA, cA, pB, cB, charges, p=saptdft_d4_params, s9=0.0,
        #     dftd4_bin = "/home/awallace43/.local/bin/dftd4",
        # )
        # C6s_dimer, _, _, df_c_e = src.locald4.calc_dftd4_c6_c8_pairDisp2(
        #     pD,
        #     cD,
        #     charges[0],
        #     p=saptdft_d4_params,
        #     dftd4_bin = "/home/awallace43/.local/bin/dftd4",
        # )
        # print(f"{C6s_dimer = }")
        # c6_monAs.append(C6s_mA)
        # c6_monBs.append(C6s_mB)
        # Actual Ar-Ar C6 values
        # C6s_dimer = np.array([[64.87588244, 64.87588244],
        #        [64.87588244, 64.87588244]])
        v = 1
        C6s_dimer = np.array([
           [v, v],
           [v, v]
        ])
        c6_dimers.append(C6s_dimer)
        distances = np.linalg.norm(geom[0, 1:] - geom[1, 1:])
        t6_1, t8_1, energies = src.locald4.compute_bj_terms(geom[:, 0], geom[:, 1:], C6s_dimer, params=saptdft_d4_params)
        print(f"{t6_1 = } {t8_1 = }")
        t6_2, t8_2, energies = src.locald4.compute_bj_terms(geom[:, 0], geom[:, 1:], C6s_dimer, params=hfd4_params)
        t6_1s.append(t6_1[1])
        t6_2s.append(t6_2[1])
        t8_1s.append(t8_1[1])
        t8_2s.append(t8_2[1])
        e1s.append(sum(energies) * 627.509)
        e2s.append(sum(energies) * 627.509)
        dist.append(distances)
        print(n, row)
        print(f"{dist[-1] = } {t6_1s[-1] = } {t6_2s[-1] = } {t8_1s[-1] = } {t8_2s[-1] = }")
        # break
    df["C6_dimer"] = c6_dimers
    # df["C6_mA"] = c6_monAs
    # df["C6_mB"] = c6_monBs
    df["t6_1"] = t6_1s
    df["t6_2"] = t6_2s
    df["t8_1"] = t8_1s
    df["t8_2"] = t8_2s
    df["energy_1"] = e1s
    df["energy_2"] = e2s
    df['p1'] = [saptdft_d4_params for i in range(len(df))]
    df['p2'] = [hfd4_params for i in range(len(df))]
    df["distance"] = dist
    print(df)
    df.to_pickle("ar_ar_c6.pkl")
    return df


def plot_ar_ar_BJ_damping_function():
    """
    We don't actually need C6's because they get divided out...
    C8 = -3C6 * sqrt(QA * QB)
    R_0^{AB} = sqrt(C8 / C6) = sqrt(-3 * sqrt(QA * QB))
    meaning only atomic number matters for the damping function
    """
    df = pd.read_pickle("ar_ar_c6.pkl")
    el1 = int(df['Geometry'].iloc[0][0, 0])
    el2 = int(df['Geometry'].iloc[0][1, 0])
    t6_1 = df["t6_1"]
    t6_2 = df["t6_2"]
    t8_1 = df["t8_1"]
    t8_2 = df["t8_2"]
    e1 = df["energy_1"]
    e2 = df["energy_2"]
    distance = df["distance"]
    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Primary y-axis for t6 and t8 values
    ax1.set_xlabel('Distance (Angstrom)', fontsize=12)
    ax1.set_ylabel('Damping Function', fontsize=12, color='black')

    # Plot t6 and t8 lines on primary y-axis
    line1 = ax1.plot(distance, t6_1, 'b-', linewidth=2, label=f't6_1 s8={df["p1"].iloc[0][1]:.2f}, a1={df["p1"].iloc[0][2]:.2f}, a2={df["p1"].iloc[0][3]:.2f}')
    line2 = ax1.plot(distance, t6_2, 'b--', linewidth=2, label=f't6_2 s8={df["p2"].iloc[0][1]:.2f}, a1={df["p2"].iloc[0][2]:.2f}, a2={df["p2"].iloc[0][3]:.2f}')
    line3 = ax1.plot(distance, t8_1, 'r-', linewidth=2, label='t8_1')
    line4 = ax1.plot(distance, t8_2, 'r--', linewidth=2, label='t8_2')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create secondary y-axis for e1 and e2 values
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Dispersion Energy (kcal/mol)', fontsize=12, color='green')

    # Plot e1 and e2 lines on secondary y-axis
    # line5 = ax2.plot(distance, e1, 'g-', linewidth=2, label='e1')
    # line6 = ax2.plot(distance, e2, 'g--', linewidth=2, label='e2')
    # ax2.tick_params(axis='y', labelcolor='green')

    # Add title
    plt.title(f'{el1}-{el2} Pairwise Damping Function', fontsize=14)

    # Add grid (only for primary y-axis)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a single legend for all lines
    # lines = line1 + line2 + line3 + line4 + line5 + line6
    lines = line1 + line2 + line3 + line4 
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10)

    # annotate note that 1s come from SAPT_DFT_D4 and 2s come from HF_D4
    ax1.annotate('params 1s from SAPT_DFT_D4', xy=(0.1, 0.65), xycoords='axes fraction', fontsize=12, color='black')
    ax1.annotate('params 2s from HF_D4', xy=(0.1, 0.60), xycoords='axes fraction', fontsize=12, color='black')

    # Improve appearance
    plt.tight_layout()
    plt.savefig("plots/ar_ar_bj_damping_function.png")
    return


def main():
    # ar_ar_setup()
    # plot_ar_ar_BJ_damping_function()
    single_curve()
    return

if __name__ == "__main__":
    main()
