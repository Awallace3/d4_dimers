import pandas as pd
import src
import subprocess, os
from qm_tools_aw import tools
from pprint import pprint as pp

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

def main():
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

if __name__ == "__main__":
    main()
