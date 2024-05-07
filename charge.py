import pandas as pd
import src
import subprocess, os

def investigations():
    df_name = "plots/basis_study.pkl"
    # df_name = "dfs/schr_dft2.pkl"
    if not os.path.exists(df_name):
        print("Cannot find ./plots/basis_study.pkl, creating it now...")
        subprocess.call("cat plots/basis_study-* > plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("tar -xzf plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("rm plots/basis_study.pkl.tar.gz", shell=True)
        subprocess.call("mv basis_study.pkl plots/basis_study.pkl", shell=True)
    df = pd.read_pickle(df_name)
    df_charged = src.plotting.get_charged_df(df)
    df_charged['id'] = df_charged.index
    df_charged.reset_index(drop=True, inplace=True)
    df_charged['size'] = df.apply(lambda x: len(x['Geometry']), axis=1)
    print(df_charged['size'].describe())
    id_max = df_charged['size'].idxmax()
    print(f"Max size molecule: {df_charged['size'][id_max]} {id_max = }")
    id_min = df_charged['size'].idxmin()
    print(f"Min size molecule: {df_charged['size'][id_min]} {id_min = }")
    pd.set_option('display.max_rows', None)
    print(df_charged[['charges', 'size', 'id']])
    min_row = df_charged.iloc[id_min]
    print(min_row['Geometry'])
    print(min_row['charges'])
    # Cation case: 7867
    # Anion case: 7898
    # Cation and Anion case: 7871
    return

def main():
    investigations()
    return


if __name__ == "__main__":
    main()

