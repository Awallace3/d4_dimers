import src
import os
import subprocess


def main():
    df_name = "./plots/ddft_study.pkl"
    if not os.path.exists(df_name):
        print("Cannot find ./plots/ddft_study.pkl, creating it now...")
        subprocess.call(
            "cat plots/ddft_study-* > plots/ddft_study.pkl.tar.gz", shell=True
        )
        subprocess.call("tar -xzf plots/ddft_study.pkl.tar.gz", shell=True)
        subprocess.call("rm plots/ddft_study.pkl.tar.gz", shell=True)
        subprocess.call("mv ddft_study.pkl plots/ddft_study.pkl", shell=True)
    df = src.plotting.plotting_setup_dft_ddft(
        # df_name,
        "./plots/LoS.pkl",
        build_df=False,
        df_out="./plots/LoS_ddft.pkl",
        original_plot=False,
    )
    # return
    print(df)
    from pprint import pprint as pp

    pp(df.columns.to_list())
    # return
    # src.plotting.plot_LoS_saptdft(
    #     df,
    #     presentation=True,
    # )
    src.plotting_saptdft.plot_LoS_saptdft(
        df,
        presentation=True,
    )
    return


if __name__ == "__main__":
    main()
