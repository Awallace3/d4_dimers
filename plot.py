import pandas as pd
import src
import subprocess, os

def main():
    df_name = "plots/basis_study.pkl"
    # df_name = "dfs/schr_dft2.pkl"
    if not os.path.exists(df_name):
        print("Cannot find ./plots/basis_study.pkl, creating it now...")
        subprocess.call("cat plots/basis_study-* > plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("tar -xzf plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("rm plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("mv basis_study.pkl plots/basis_study.pkl", shell=True)
    df = pd.read_pickle(df_name)
    print('Read df from ./plots/basis_study.pkl')
    # df = src.plotting.plotting_setup_dft(
    #     (df, df_name),
    #     build_df=False,
    # )
    df = src.plotting.plotting_setup(
        (df, df_name),
        False,
    )
    return
    # return
    df = src.plotting.plot_basis_sets_d4(
        df,
        False,
    )
    df = src.plotting.plot_basis_sets_d3(
        df,
        False,
    )
    return


if __name__ == "__main__":
    main()
